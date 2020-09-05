# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the adversarial patch attack `AdversarialPatch`. This attack generates an adversarial patch that
can be printed into the physical world with a common printer. The patch can be used to fool image classifiers.

| Paper link: https://arxiv.org/abs/1712.09665
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import (
    ClassifierMixin,
    ClassifierNeuralNetwork,
    ClassifierGradients,
)
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)

class ZO_AdaMM(object):
    """
    Black Box optimizer
    """
    
    def __init__(self, model, input_shape, time_stp = 1, beta_1=None, beta_2=None, m=None, v=None, v_hat=None, device='cpu'):
        import  art.attacks.evasion.adversarial_patch.zoo_adamm_Utils as util
        
        self.v_init = 1e-7
        dim = np.prod(list(input_shape))
        self.delta_adv = np.zeros((1,dim))
        self.model = model        
        
        # initialize the best solution & best loss
        self.best_adv_img = []  # successful adv image in [-0.5, 0.5]
        self.best_delta = []    # best perturbation
#         self.best_distortion = (0.5 * self.dim) ** 2 # threshold for best perturbation
        self.total_loss = [] ## I: max iters
        self.l2s_loss_all = []
        self.attack_flag = False
        self.first_flag = True ## record first successful attack
        
            
        if beta_1==None:
            self.beta_1 = 0.9
        
        if beta_2==None:
            self.beta_2 = 0.9
            
        if m==None:
            self.m = np.zeros((1,dim))
        if v==None:
            self.v = self.v_init * np.ones((1,dim))
        
        if v_hat==None:
            self.v_hat = self.v_init * np.ones((1,dim))
            
    def step(self, origImage, target_label, timesteps=1, alpha=None):
        import  art.attacks.evasion.adversarial_patch.zoo_adamm_Utils as util
        
        self.origImage=origImage
        self.orgShape=self.origImage.shape
        self.dim = np.prod(self.orgShape)
        self.origImage_vec = self.origImage.reshape(1,self.dim)
        self.target_label = target_label  
        self.timesteps = timesteps
        self.args={"maxiter":timesteps,
           "init_const" : 10,
           "kappa" : 1e-10,
           "decay_lr" : True,
           "exp_code" : 50,
           "q" : 10,
           "mu":0.05,
           "lr_idx" : 0,
           "lr" : 3e-4,
           "constraint" : 'cons',
           "decay_lr" : True,
           "arg_targeted_attack" : True} 
        
        if alpha==None:
            self.alpha=np.random.uniform(1e-3,1e-2,size=(self.timesteps,))
        
        mu = self.args["mu"]
        base_lr = self.args["lr"]
        for t in range(timesteps):
            if self.args["constraint"] == 'uncons':
                # * 0.999999 to avoid +-0.5 return +-infinity 
                self.w_ori_img_vec = np.arctanh(2 * (self.origImage_vec) * 0.999999)  # in real value, note that orig_img_vec in [-0.5, 0.5]
                self.w_img_vec = np.arctanh(2 * (np.clip(self.origImage_vec + self.delta_adv,-0.5,0.5)) * 0.999999) 
            else:
                self.w_ori_img_vec = self.origImage_vec.copy()
                self.w_img_vec = np.clip(self.origImage_vec + self.delta_adv,0.,255.)
                
            if self.args["decay_lr"]:
                base_lr = self.args["lr"]/np.sqrt(t+1)

            ## Total loss evaluation
            if self.args["constraint"] == 'uncons':
                self.total_loss[t], self.l2s_loss_all[t] = self.function_evaluation_uncons(self.w_img_vec, self.args["kappa"], self.target_label, self.args["init_const"], self.model, 
                                                                                           self.origImage, self.args["arg_targeted_attack"])

            else:
                total_loss, l2s_loss_all = self.function_evaluation_cons(self.w_img_vec, self.args["kappa"], self.target_label, self.args["init_const"], self.model, 
                                                                                         self.origImage, self.args["arg_targeted_attack"])

            self.total_loss.append(total_loss)
            self.l2s_loss_all.append(l2s_loss_all)
            ## gradient estimation w.r.t. w_img_vec
            self.grad_est = self.gradient_estimation_v2(self.args["mu"], self.args["q"], self.w_img_vec, self.dim, self.args["kappa"], self.target_label, self.args["init_const"], self.model,
                                                   self.origImage, self.args["arg_targeted_attack"], self.args["constraint"])
    
            
            
            self.m = self.beta_1 * self.m + (1 - self.beta_1) * self.grad_est
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(self.grad_est) ### vt
            self.v_hat = np.maximum(self.v_hat,self.v)
            
            self.delta_adv = self.delta_adv - base_lr * self.m /np.sqrt(self.v_hat)
            if self.args["constraint"] == 'cons':
                tmp = self.delta_adv.copy()
                V_temp = np.sqrt(self.v_hat)
                self.delta_adv = self.projection_box(tmp, self.origImage_vec, V_temp, 0., 255.)
            
            self.w_img_vec = self.w_ori_img_vec + self.delta_adv
            
#             adv_img_vec = self.w_img_vec.copy()
#             adv_img = np.resize(adv_img_vec, self.origImage.shape)
            
            
#             attack_prob, _, _ = util.model_prediction(self.model, adv_img)
#             target_prob = attack_prob[0, self.target_label]
#             attack_prob_tmp = attack_prob.copy()
#             attack_prob_tmp[0, self.target_label] = 0
#             other_prob = np.amax(attack_prob_tmp)
#             if self.args["arg_targeted_attack"]:

#                 self.best_adv_img = adv_img
#                 self.best_distortion = self.distortion(adv_img, self.origImage)
#                 self.best_delta = adv_img - self.origImage
#                 self.best_iteration = t + 1
#                 adv_class = np.argmax(attack_prob)
#                 self.attack_flag = True
#                 ## Record first attack
#                 if (self.first_flag):
#                     self.first_flag = False  ### once gets into this, it will no longer record the next sucessful attack
#                     self.first_adv_img = adv_img
#                     self.first_distortion = self.distortion(adv_img, self.origImage)
#                     self.first_delta = adv_img - self.origImage
#                     self.first_class = adv_class
#                     first_iteration = t + 1
        
    def function_evaluation_cons(self, x, kappa, target_label, const, model, orig_img, arg_targeted_attack):
        # x is in [-0.5, 0.5]
        import  art.attacks.evasion.adversarial_patch.zoo_adamm_Utils as util
        import tensorflow as tf
        img_vec = x.copy()
        img = np.resize(img_vec, orig_img.shape)
        orig_prob, orig_class, orig_prob_str = util.model_prediction(model, img)
#         tmp = orig_prob.copy()
#         tmp[0, target_label] = 0
# #         print(np.log(np.amax(tmp) + 1e-10) - np.log(orig_prob[0, target_label] + 1e-10))
        
# #         print(np.log(orig_prob[0, target_label] + 1e-10))
#         if arg_targeted_attack:  # targeted attack, target_label is false label
#             Loss1 =  np.max([np.log(np.amax(tmp) + 1e-10) - np.log(orig_prob[0, target_label] + 1e-10), -kappa])
#         else:  # untargeted attack, target_label is true label
#             Loss1 =  np.max([np.log(orig_prob[0, target_label] + 1e-10) - np.log(np.amax(tmp) + 1e-10), -kappa])

#         Loss2 = np.linalg.norm(img - orig_img) ** 2 ### squared norm
# #         print(Loss2)
        print(orig_prob.max())
        Loss2 = 0
        Loss_per_example = tf.keras.losses.categorical_crossentropy(
            y_true=target_label, y_pred=orig_prob, from_logits=False, label_smoothing=0
        )
        Loss1 = tf.reduce_mean(Loss_per_example)
        return Loss1.numpy() + Loss2, Loss2
    
    
    def gradient_estimation_v2(self,mu,q,x,d,kappa,target_label,const,model,orig_img,arg_targeted_attack,arg_cons):
        import  art.attacks.evasion.adversarial_patch.zoo_adamm_Utils as util
        # x is img_vec format in real value: w
        # m, sigma = 0, 100 # mean and standard deviation
        sigma = 100
        # ## generate random direction vectors
        # U_all_new = np.random.multivariate_normal(np.zeros(d), np.diag(sigma*np.ones(d) + 0), (q,1))


        f_0, ignore =self.function_evaluation_cons(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)

        grad_est=0
        for i in range(q):
            u = np.random.normal(0, sigma, (1,d))
            u_norm = np.linalg.norm(u)

            # ui = U_all_new[i, 0].reshape(-1)
            # u = ui / np.linalg.norm(ui)
            # u = np.resize(u, x.shape)

            f_tmp, ignore = self.function_evaluation_cons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
            # gradient estimate
            # if arg_mode == "ZO-M-signSGD":
            #     grad_est=grad_est+ np.sign(u*(f_tmp-f_0))
            # else:
            print(f_tmp-f_0)
            grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/mu
        return grad_est
        #grad_est=grad_est.reshape(q,d)
        #return d*grad_est.sum(axis=0)/q
    

    def projection_box(self, a_point, X, Vt, lb, up):
        import  art.attacks.evasion.adversarial_patch.zoo_adamm_Utils as util
        ## X \in R^{d \times m}
        #d_temp = a_point.size
        VtX = np.sqrt(Vt)*X

        min_VtX = np.min(VtX, axis=0)
        max_VtX = np.max(VtX, axis=0)

        Lb = lb * np.sqrt(Vt) - min_VtX
        Ub = up * np.sqrt(Vt) - max_VtX

        a_temp = np.sqrt(Vt)*a_point
        z_proj_temp = np.multiply(Lb, np.less(a_temp, Lb)) + np.multiply(Ub, np.greater(a_temp, Ub)) \
                      + np.multiply(a_temp, np.multiply( np.greater_equal(a_temp, Lb), np.less_equal(a_temp, Ub)))
        #delta_proj = np.diag(1/np.diag(np.sqrt(Vt)))*z_proj_temp
        delta_proj = 1/np.sqrt(Vt)*z_proj_temp
        #print(delta_proj)
        return delta_proj.reshape(a_point.shape)
    

    def projection_box_eps(self, a_point, X, Vt, lb, up, eps = 16/256):
        import  art.attacks.evasion.adversarial_patch.zoo_adamm_Utils as util
        ## X \in R^{d \times m}
        #d_temp = a_point.size
        #X = np.reshape(X, (X.shape[0]*X.shape[1],X.shape[2]))
        min_VtX = np.min(X, axis=0)
        max_VtX = np.max(X, axis=0)

        Lb = np.maximum(-eps, lb - min_VtX)
        Ub = np.minimum( eps, up - max_VtX)
        z_proj_temp = np.clip(a_point,Lb,Ub)
        return z_proj_temp.reshape(a_point.shape)
    
    def distortion(self, a, b):
        return np.linalg.norm(a - b) 
    
    def zero_grad(self):
        return self.__init__(self.inputVar,self.func,**self.args)
    
    def returnInput(self):
        return self.origImage
        
    def returnloss(self):
        return self.total_loss[-1]
            
    def gradFn(self):
#         print(self.grad_est.shape)
        return self.grad_est.reshape(self.orgShape)
    

class AdversarialPatchTensorFlowV2(EvasionAttack):
    """
    Implementation of the adversarial patch attack.

    | Paper link: https://arxiv.org/abs/1712.09665
    """

    attack_params = EvasionAttack.attack_params + [
        "rotation_max",
        "scale_min",
        "scale_max",
        "learning_rate",
        "max_iter",
        "batch_size",
        "patch_shape",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: Union[ClassifierNeuralNetwork, ClassifierGradients],
        rotation_max: float = 22.5,
        scale_min: float = 0.1,
        scale_max: float = 1.0,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        patch_shape: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Create an instance of the :class:`.AdversarialPatchTensorFlowV2`.

        :param classifier: A trained classifier.
        :param rotation_max: The maximum rotation applied to random patches. The value is expected to be in the
               range `[0, 180]`.
        :param scale_min: The minimum scaling applied to random patches. The value should be in the range `[0, 1]`,
               but less than `scale_max`.
        :param scale_max: The maximum scaling applied to random patches. The value should be in the range `[0, 1]`, but
               larger than `scale_min.`
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape (width, height, nb_channels).
                            Currently only supported for `TensorFlowV2Classifier`. For classifiers of other frameworks
                            the `patch_shape` is set to the shape of the image samples.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        super(AdversarialPatchTensorFlowV2, self).__init__(estimator=classifier)
        self.rotation_max = rotation_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.image_shape = classifier.input_shape
        self._check_params()

        if self.image_shape[2] not in [1, 3]:
            raise ValueError("Color channel need to be in last dimension.")

        if self.patch_shape is not None:
            if self.patch_shape[2] not in [1, 3]:
                raise ValueError("Color channel need to be in last dimension.")
            if self.patch_shape[0] != self.patch_shape[1]:
                raise ValueError("Patch height and width need to be the same.")
        if not (self.estimator.postprocessing_defences is None or self.estimator.postprocessing_defences == []):
            raise ValueError(
                "Framework-specific implementation of Adversarial Patch attack does not yet support "
                + "postprocessing defences."
            )

        mean_value = (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2.0 + self.estimator.clip_values[
            0
        ]
        initial_value = np.ones(self.patch_shape) * mean_value
        self._patch = tf.Variable(
            initial_value=initial_value,
            shape=self.patch_shape,
            dtype=tf.float32,
            constraint=lambda x: tf.clip_by_value(x, self.estimator.clip_values[0], self.estimator.clip_values[1]),
        )

        self._train_op = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate, momentum=0.0, nesterov=False, name="SGD"
        )
        
        
        self.v_init = 1e-7
        self.dim = (self.batch_size,)+self.image_shape
        self.beta_1 = 0.9
        self.beta_2 = 0.9
        self.m = np.zeros(self.dim)
        self.v = self.v_init * np.ones(self.dim)
        self.v_hat = self.v_init * np.ones(self.dim)
        self.u = tf.convert_to_tensor(np.random.normal(0,1,self.dim), dtype = tf.float32)
        self.mu = 0.05
        self.zoo_lr = 3e-2
        self.zoo_timesteps = 1

    def _zootrain_step(self, images: Optional[np.ndarray] = None, target: Optional[np.ndarray] = None) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]
        for t in range(self.zoo_timesteps):
            if target is None:
                target = self.estimator.predict(x=images)
                self.targeted = False
            else:
                self.targeted = True

            with tf.GradientTape() as tape:
                tape.watch(self._patch)
                loss1 = self._loss(images, target)
                loss2 = self._loss(images + self.u*self.mu, target)
            
#             loss = (loss2+loss1)/2
            gradients = (np.prod(list(self.dim)) / self.mu) * (loss2.numpy() - loss1.numpy()) * self.u

            self.m = self.beta_1*self.m + (1-self.beta_1)*gradients
            self.v = self.beta_2*self.v + (1-self.beta_2)*(gradients**2)
            self.v_hat = np.maximum(self.v_hat, self.v)
            images = images   -  (self.zoo_lr * np.sqrt(self.v_hat) * self.m)
            
            

        self._train_op.apply_gradients(zip(gradients, [self._patch]))
#         print(self._patch.shape)
#         print('Max', np.array(gradients).max())
#         print('min ',np.array(gradients).min())
#         print('loss', loss1+loss2)
        return loss1+loss2
    
    def _train_step(self, images: Optional[np.ndarray] = None, target: Optional[np.ndarray] = None) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        if target is None:
            target = self.estimator.predict(x=images)
            self.targeted = False
        else:
            self.targeted = True

        with tf.GradientTape() as tape:
            tape.watch(self._patch)
            loss = self._loss(images, target)

        gradients = tape.gradient(loss, [self._patch])

        if not self.targeted:
            gradients = [-g for g in gradients]          
            

        self._train_op.apply_gradients(zip(gradients, [self._patch]))
        return loss

    def _probabilities(self, images: "tf.Tensor") -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        patched_input = self._random_overlay(images, self._patch)

        patched_input = tf.clip_by_value(
            patched_input, clip_value_min=self.estimator.clip_values[0], clip_value_max=self.estimator.clip_values[1],
        )

        probabilities = self.estimator._predict_framework(patched_input)

        return probabilities

    def _loss(self, images: "tf.Tensor", target: "tf.Tensor") -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        probabilities = self._probabilities(images)

        self._loss_per_example = tf.keras.losses.categorical_crossentropy(
            y_true=target, y_pred=probabilities, from_logits=False, label_smoothing=0
        )

        loss = tf.reduce_mean(self._loss_per_example)

        return loss

    def _get_circular_patch_mask(self, nb_images: int, sharpness: int = 40) -> "tf.Tensor":
        """
        Return a circular patch mask.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        diameter = self.image_shape[0]

        x = np.linspace(-1, 1, diameter)
        y = np.linspace(-1, 1, diameter)
        x_grid, y_grid = np.meshgrid(x, y, sparse=True)
        z_grid = (x_grid ** 2 + y_grid ** 2) ** sharpness

        image_mask = 1 - np.clip(z_grid, -1, 1)
        image_mask = np.expand_dims(image_mask, axis=2)
        image_mask = np.broadcast_to(image_mask, self.image_shape)
        image_mask = tf.stack([image_mask] * nb_images)
        return image_mask

    def _random_overlay(self, images: np.ndarray, patch: np.ndarray, scale: Optional[float] = None) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]
        import tensorflow_addons as tfa

        nb_images = images.shape[0]
        image_mask = self._get_circular_patch_mask(nb_images=nb_images)
        image_mask = tf.cast(image_mask, images.dtype)
        patch = tf.cast(patch, images.dtype)
        padded_patch = tf.stack([patch] * nb_images)
        transform_vectors = list()

        for i in range(nb_images):
            if scale is None:
                im_scale = np.random.uniform(low=self.scale_min, high=self.scale_max)
            else:
                im_scale = scale
            padding_after_scaling = (1 - im_scale) * self.image_shape[0]
            x_shift = np.random.uniform(-padding_after_scaling, padding_after_scaling)
            y_shift = np.random.uniform(-padding_after_scaling, padding_after_scaling)
            phi_rotate = float(np.random.uniform(-self.rotation_max, self.rotation_max)) / 90.0 * (math.pi / 2.0)

            # Rotation
            rotation_matrix = np.array(
                [[math.cos(-phi_rotate), -math.sin(-phi_rotate)], [math.sin(-phi_rotate), math.cos(-phi_rotate)],]
            )

            # Scale
            xform_matrix = rotation_matrix * (1.0 / im_scale)
            a0, a1 = xform_matrix[0]
            b0, b1 = xform_matrix[1]

            x_origin = float(self.image_shape[0]) / 2
            y_origin = float(self.image_shape[1]) / 2

            x_origin_shifted, y_origin_shifted = np.matmul(xform_matrix, np.array([x_origin, y_origin]))

            x_origin_delta = x_origin - x_origin_shifted
            y_origin_delta = y_origin - y_origin_shifted

            a2 = x_origin_delta - (x_shift / (2 * im_scale))
            b2 = y_origin_delta - (y_shift / (2 * im_scale))

            transform_vectors.append(np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32))

        image_mask = tfa.image.transform(image_mask, transform_vectors, "BILINEAR")
        padded_patch = tfa.image.transform(padded_patch, transform_vectors, "BILINEAR")
        inverted_mask = 1 - image_mask

        return images * inverted_mask + padded_patch * image_mask

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        import tensorflow as tf  # lgtm [py/repeated-import]

        y = check_and_transform_label_format(labels=y, nb_classes=self.estimator.nb_classes)

        shuffle = kwargs.get("shuffle", True)
        if shuffle:
            ds = (
                tf.data.Dataset.from_tensor_slices((x, y))
                .shuffle(10000)
                .batch(self.batch_size)
                .repeat(math.ceil(self.max_iter / (x.shape[0] / self.batch_size)))
            )
        else:
            ds = (
                tf.data.Dataset.from_tensor_slices((x, y))
                .batch(self.batch_size)
                .repeat(math.ceil(self.max_iter / (x.shape[0] / self.batch_size)))
            )

        i_iter = 0
        
            
        for images, target in tqdm(ds):

            if i_iter >= self.max_iter:
                break
            if not self.estimator._zooAdamm:
                loss = self._train_step(images=images, target=target)
            else:
                loss = self._zootrain_step(images=images, target=target)

            if divmod(i_iter, 10)[1] == 0:
                logger.info("Iteration: {} Loss: {}".format(i_iter, loss))

            i_iter += 1

        return (
            self._patch.numpy(),
            self._get_circular_patch_mask(nb_images=1).numpy()[0],
        )

    def apply_patch(self, x: np.ndarray, scale: float, patch_external: Optional[np.ndarray] = None) -> np.ndarray:
        """
        A function to apply the learned adversarial patch to images.

        :param x: Instances to apply randomly transformed patch.
        :param scale: Scale of the applied patch in relation to the classifier input shape.
        :param patch_external: External patch to apply to images `x`.
        :return: The patched samples.
        """
        patch = patch_external if patch_external is not None else self._patch
        return self._random_overlay(images=x, patch=patch, scale=scale).numpy()

    def reset_patch(self, initial_patch_value: np.ndarray) -> None:
        """
        Reset the adversarial patch.

        :param initial_patch_value: Patch value to use for resetting the patch.
        """
        initial_value = np.ones(self.patch_shape) * initial_patch_value
        self._patch.assign(np.ones(shape=self.patch_shape) * initial_value)
