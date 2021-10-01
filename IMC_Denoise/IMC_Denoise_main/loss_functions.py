# -*- coding: utf-8 -*-

import tensorflow as tf

def create_weighted_binary_crossentropy():
  def weighted_ce(y_true, y_pred):
    target, mask = tf.split(y_true, 2, axis = -1)
    loss1 = tf.multiply(target, tf.math.log(y_pred + 1e-15))
    loss2 = tf.multiply(1.0 - target, tf.math.log(1.0 - y_pred + 1e-15))
    bce = tf.multiply(-(loss1 + loss2), mask)
    return tf.reduce_sum(bce)/ tf.reduce_sum(mask)
  return weighted_ce
  
def create_mse():
  def weighted_mse(y_true, y_pred):
    target, mask = tf.split(y_true, 2, axis=-1) 
    loss_mask = tf.reduce_sum(tf.math.square(y_pred - target) * mask) / tf.reduce_sum(mask)
    loss = loss_mask
    return loss
  return weighted_mse