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


from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export

class Adam(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the Adam algorithm.
  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.
  According to
  [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
  the method is "*computationally
  efficient, has little memory requirement, invariant to diagonal rescaling of
  gradients, and is well suited for problems that are large in terms of
  data/parameters*".
  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use, The
      learning rate. Defaults to 0.001.
    beta_1: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use. The
      exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use, The
      exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon: A small constant for numerical stability. This epsilon is
      "epsilon hat" in the Kingma and Ba paper (in the formula just before
      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
      1e-7.
    amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
      the paper "On the Convergence of Adam and beyond". Defaults to `False`.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Adam"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.
  Usage:
  >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> # The first step is `-learning_rate*sign(grad)`
  >>> var1.numpy()
  9.9
  Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.
  Notes:
  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.
  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               name='Adam',
               **kwargs):
    super(Adam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self.amsgrad = amsgrad

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')
    for var in var_list:
      self.add_slot(var, 'v')
    if self.amsgrad:
      for var in var_list:
        self.add_slot(var, 'vhat')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Adam, self)._prepare_local(var_device, var_dtype, apply_state)

    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
          (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
    apply_state[(var_device, var_dtype)].update(
        dict(
            lr=lr,
            epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t))

  def set_weights(self, weights):
    params = self.weights
    # If the weights are generated by Keras V1 optimizer, it includes vhats
    # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
    # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
    num_vars = int((len(params) - 1) / 2)
    if len(weights) == 3 * num_vars + 1:
      weights = weights[:len(params)]
    super(Adam, self).set_weights(weights)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    if not self.amsgrad:
      return training_ops.resource_apply_adam(
          var.handle,
          m.handle,
          v.handle,
          coefficients['beta_1_power'],
          coefficients['beta_2_power'],
          coefficients['lr_t'],
          coefficients['beta_1_t'],
          coefficients['beta_2_t'],
          coefficients['epsilon'],
          grad,
          use_locking=self._use_locking)
    else:
      vhat = self.get_slot(var, 'vhat')
      return training_ops.resource_apply_adam_with_amsgrad(
          var.handle,
          m.handle,
          v.handle,
          vhat.handle,
          coefficients['beta_1_power'],
          coefficients['beta_2_power'],
          coefficients['lr_t'],
          coefficients['beta_1_t'],
          coefficients['beta_2_t'],
          coefficients['epsilon'],
          grad,
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

    if not self.amsgrad:
      v_sqrt = math_ops.sqrt(v_t)
      var_update = state_ops.assign_sub(
          var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t])
    else:
      v_hat = self.get_slot(var, 'vhat')
      v_hat_t = math_ops.maximum(v_hat, v_t)
      with ops.control_dependencies([v_hat_t]):
        v_hat_t = state_ops.assign(
            v_hat, v_hat_t, use_locking=self._use_locking)
      v_hat_sqrt = math_ops.sqrt(v_hat_t)
      var_update = state_ops.assign_sub(
          var,
          coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

  def get_config(self):
    config = super(Adam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad,
    })
    return config

    

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
        self.dim = (1,)+self.image_shape
        self.beta_1 = 0.9
        self.beta_2 = np.random.uniform(1e-3,1e-2)

        self.mu = 1.0
        self.zoo_lr = np.random.uniform(1e-3,1e-2)
        self.zoo_timesteps = 10
        
        self._zootrain_op = Adam(learning_rate=self.learning_rate, beta_1 = self.beta_1, beta_2 = self.beta_2)
        
        
    def _zootrain_step(self, images: Optional[np.ndarray] = None, target: Optional[np.ndarray] = None) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]
        self.m = np.zeros(self.dim)
        self.v = self.v_init * np.ones(self.dim)
        self.v_hat = self.v_init * np.ones(self.dim)
        delta_adv = 0
        for t in range(self.zoo_timesteps):
            self.u = np.random.uniform(0,1,self.dim)
            if target is None:
                target = self.estimator.predict(x=images)
                self.targeted = False
            else:
                self.targeted = True


            loss = (self._zooloss(images , self.u * self.mu, target) - self._zooloss(images, [0], target))


            del_fx = (np.prod(self.dim) / self.mu) * (loss.numpy()) * self.u

            self.m = self.beta_1 * self.m + (1-self.beta_1) * del_fx
            self.v = self.beta_2 * self.v + (1-self.beta_2) * (del_fx**2)
            self.v_hat = np.maximum(self.v_hat, self.v)
            adaptive_lr = self.learning_rate / (np.sqrt(self.v_hat))
            gradients = (adaptive_lr * self.m)[0]

            delta_adv = delta_adv   -  gradients
            tmp = delta_adv.copy()
            V_temp = np.sqrt(self.v_hat.reshape(1,-1))
            X = images.numpy().reshape(images.shape[0], np.prod(images.shape[1:]))
            delta_adv = self.zooprojection_box(tmp, X, V_temp, 0, 255)
            self._patch = tf.clip_by_value(self._patch + delta_adv, clip_value_min=self.estimator.clip_values[0], clip_value_max=self.estimator.clip_values[1])
#             self._zootrain_op.apply_gradients(zip(gradients, [self._patch]))
#             print(self._patch)
#             print(gradients.min())
#             print(gradients.max())
#             print('patch', self._patch.numpy().reshape(1,-1))
#         print(loss)
        

        return loss

    
    def zooprojection_box(self, a_point, X, Vt, lb, up):
        ## X \in R^{d \times m}
        #d_temp = a_point.size
        VtX = np.sqrt(Vt)*X

        min_VtX = np.min(VtX, axis=0)
        max_VtX = np.max(VtX, axis=0)

        Lb = lb * np.sqrt(Vt) - min_VtX
        Ub = up * np.sqrt(Vt) - max_VtX

        a_temp = np.sqrt(Vt)*a_point.reshape(1,-1)
        z_proj_temp = np.multiply(Lb, np.less(a_temp, Lb)) + np.multiply(Ub, np.greater(a_temp, Ub)) \
                      + np.multiply(a_temp, np.multiply( np.greater_equal(a_temp, Lb), np.less_equal(a_temp, Ub)))
        #delta_proj = np.diag(1/np.diag(np.sqrt(Vt)))*z_proj_temp
        delta_proj = 1/np.sqrt(Vt)*z_proj_temp
        return delta_proj.reshape(a_point.shape)
    
    def _zooprobabilities(self, images: "tf.Tensor", changes) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        patched_input = self._random_overlay(images, tf.clip_by_value(self._patch + changes[0], clip_value_min=self.estimator.clip_values[0], clip_value_max=self.estimator.clip_values[1]))
        
        patched_input = tf.clip_by_value(
            patched_input, clip_value_min=self.estimator.clip_values[0], clip_value_max=self.estimator.clip_values[1],
        )

        probabilities = self.estimator._predict_framework(patched_input)

        return probabilities

    def _zooloss(self, images: "tf.Tensor", changes, target: "tf.Tensor") -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        probabilities = self._zooprobabilities(images, changes)

        self._loss_per_example = tf.keras.losses.categorical_crossentropy(
            y_true=target, y_pred=probabilities, from_logits=False, label_smoothing=0
        )

        loss = tf.reduce_mean(self._loss_per_example)

        return loss    

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
            np.clip(self._patch.numpy(),0,255),
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
            
        patch = patch_external if patch_external is not None else np.clip(self._patch.numpy(),0,255)
        return self._random_overlay(images=x, patch=patch, scale=scale).numpy()

    def reset_patch(self, initial_patch_value: np.ndarray) -> None:
        """
        Reset the adversarial patch.

        :param initial_patch_value: Patch value to use for resetting the patch.
        """
        initial_value = np.ones(self.patch_shape) * initial_patch_value
        self._patch.assign(np.ones(shape=self.patch_shape) * initial_value)
