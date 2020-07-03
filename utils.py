#!/usr/bin/env python
# coding: utf-8

# In[1]:



"""
@author: sumanyumuku98
"""

import os
import sys
import time
import torch
import numpy as np
import torchvision
# import torchvision.transforms as transforms
import scipy.linalg as linalg
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import Optimizer
# from torch.optim import SGD, Adam
from torch.autograd import Variable
import math
from scipy.ndimage.interpolation import rotate

# device='cuda' if torch.cuda.is_available() else 'cpu'
# print(device)



# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 80

TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' ' + msg)
    L.append(' | Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def submatrix(arr):
    x, y = np.nonzero(arr)
#     print(x)
#     print(y)
    # Using the smallest and largest x and y indices of nonzero elements, 
    # we can find the desired rectangular bounds.  
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1]


class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr
    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255
    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


def init_patch_circle(image_size, patch_size):
    image_size = image_size**2
    noise_size = int(image_size*patch_size)
    radius = int(math.sqrt(noise_size/math.pi))
    patch = np.zeros((1, 3, radius*2, radius*2))    
    for i in range(3):
        a = np.zeros((radius*2, radius*2))    
        cx, cy = radius, radius # The center of circle 
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 <= radius**2
        a[cy-radius:cy+radius, cx-radius:cx+radius][index] = np.random.rand()
        idx = np.flatnonzero((a == 0).all((1)))
        a = np.delete(a, idx, axis=0)
        patch[0][i] = np.delete(a, idx, axis=1)
    return patch, patch.shape


def circle_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image 
    x = np.zeros(data_shape)
   
    # get shape
    m_size = patch_shape[-1]
    
    for i in range(x.shape[0]):

        # random rotation
        rot = np.random.choice(360)
        for j in range(patch[i].shape[0]):
            patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False)
        
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
       
        # apply patch to dummy image  
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    return x, mask, patch.shape


def init_patch_square(image_size, patch_size):
    # get mask
    image_size = image_size**2
    noise_size = image_size*patch_size
    noise_dim = int(noise_size**(0.5))
    patch = np.random.rand(1,3,noise_dim,noise_dim)
    return patch, patch.shape


def square_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image 
    x = np.zeros(data_shape)
    
    # get shape
    m_size = patch_shape[-1]
    
    for i in range(x.shape[0]):

        # random rotation
        rot = np.random.choice(4)
        for j in range(patch[i].shape[0]):
            patch[i][j] = np.rot90(patch[i][j], rot)
        
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
       
        # apply patch to dummy image  
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    return x, mask


# In[3]:


# x[0][1]


# In[4]:


class ZO_AdaMM(object):
    """
    Black Box optimizer
    """
    
    def __init__(self,inputVar,func,timesteps=1,alpha=None,beta_1=None,beta_2=None,m=None,v=None,v_app=None):
        self.inputVar=inputVar
        self.orgShape=self.inputVar.shape
        
        if len(self.orgShape)!=0:
#             self.inputVar=torch.flatten(self.inputVar)
            self.dim=np.prod(self.orgShape)
            self.scalar=False
        else:
            self.scalar=True
        
        try:
            if not torch.is_tensor(self.inputVar):
                raise TypeError
        except TypeError:
            print("Input should be a tensor")
         
        #self.inputVar.requires_grad_()
        self.func=func
        
        self.args={'timesteps':timesteps,'alpha':alpha,'beta_1':beta_1,'beta_2':beta_2,
                   'm':m,'v':v,'v_app':v_app}
        
        self.timesteps=timesteps
        self.mu=1.0
        
        if alpha==None:
            self.alpha=np.random.uniform(1e-3,1e-2,size=(self.timesteps,))
            
        if beta_1==None:
            self.beta_1=np.random.uniform(1e-3,1e-2,size=(self.timesteps,))
        
        if beta_2==None:
            self.beta_2=np.random.uniform(0,1)
            
        if m==None:
            if not self.scalar:
                self.m=torch.zeros_like(self.inputVar).double()
            else:
                self.m=torch.tensor(0.0)
        if v==None:
            if not self.scalar:
                self.v=torch.zeros_like(self.inputVar).double()
            else:
                self.v=torch.tensor(0.0)
        
        if v_app==None:
            if not self.scalar:
                self.v_app=torch.zeros_like(self.inputVar).double()
            else:
                self.v_app=torch.tensor(0.0)
        
        if not self.scalar:
            self.grad=torch.zeros_like(self.inputVar).double()
        else:
            self.grad=torch.tensor(0.0)
        
    def step(self):
        for t in range(self.timesteps):
            if not self.scalar:
                u=torch.tensor(np.random.uniform(0,1,self.orgShape)).double()
            else:
                u=torch.tensor(np.random.uniform())
            
            if not self.scalar:
                g_t= (self.func(self.inputVar+self.mu*u)-self.func(self.inputVar))*(self.dim/self.mu)*u
            else:
                g_t= (self.func(self.inputVar+self.mu*u)-self.func(self.inputVar))*(1.0/self.mu)*u
            
            self.m=self.beta_1[t]*self.m + (1.0-self.beta_1[t])*g_t
            self.v=self.beta_2*self.v + (1.0-self.beta_2)*(g_t**2)
            
            self.v_app=torch.max(self.v_app,self.v)
            
            if not self.scalar:
                v_app_np= np.diag(self.v_app.flatten().numpy())
            else:
                v_app_np=self.v_app.numpy()
            
            
            if not self.scalar:
                self.grad= torch.tensor(np.matmul(self.alpha[t]*linalg.fractional_matrix_power(v_app_np,-0.5),
                                                  self.m.flatten().numpy()).reshape(self.orgShape))
            else:
                self.grad=torch.tensor(self.alpha[t]*np.power(v_app_np,-0.5)*self.m.numpy())
            
            self.inputVar=self.inputVar-self.grad
            
    
    def zero_grad(self):
        self.__init__(self.inputVar,self.func,**self.args)
    
    def returnInput(self):
        return self.inputVar
    
    def grad(self):
        return self.grad


# In[ ]:





# In[ ]:




