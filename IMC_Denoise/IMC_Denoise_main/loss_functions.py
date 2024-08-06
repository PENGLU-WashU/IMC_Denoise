# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import backend as K

def HF_regularization(Img_in):
    sum_axis = [1,2,3]
    dx,dy = tf.image.image_gradients(Img_in)
    dxx,dxy = tf.image.image_gradients(dx)
    dyx,dyy = tf.image.image_gradients(dy)
    HF = tf.reduce_sum(tf.abs(dxx),axis = sum_axis)+tf.reduce_sum(tf.abs(dyy),axis = sum_axis)+2**0.5*tf.reduce_sum(tf.abs(dxy),axis = sum_axis)
    return HF

def create_I_divergence(lambda_HF):
  def weighted_I_divergence(y_true, y_pred):  
    target, mask = tf.split(y_true, 2, axis = -1)
    loss1 = tf.multiply(target, tf.math.log(target+1e-15))
    loss2 = tf.multiply(target, tf.math.log(y_pred))
    I_divergence = tf.multiply((loss1 - loss2 - target + y_pred), mask)
    return tf.reduce_sum(I_divergence)/ tf.reduce_sum(mask) + lambda_HF*tf.reduce_mean(HF_regularization(y_pred))
  return weighted_I_divergence
  
def create_mse(lambda_HF):
  def weighted_mse(y_true, y_pred):
    target, mask = tf.split(y_true, 2, axis=-1) 
    loss_mask = tf.reduce_sum(tf.math.square(y_pred - target) * mask) / tf.reduce_sum(mask)
    loss = loss_mask  + lambda_HF*K.mean(HF_regularization(y_pred))
    return loss
  return weighted_mse