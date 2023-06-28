import numpy as np 
from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras import * 
import os 

def random_crop(lr_img, hr_img, hr_crop_size=128):
    lr_crop_size = hr_crop_size
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w
    hr_h = lr_h

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


class CharBonnierLoss(keras.losses.Loss):
    def __init__(self):
        super(CharBonnierLoss, self).__init__()

    def __call__(self, true_y, pred_y):
        loss_val = tf.reduce_mean(tf.sqrt(tf.square(true_y - pred_y) + tf.square(1e-3)))
        return loss_val


class PSNR(keras.metrics.Metric):
    def __init__(self):
        super(PSNR, self).__init__()

    def call(self, true_y, pred_y):
        psnr_score =  tf.image.psnr(pred_y, true_y, max_val=255.0)
        return psnr_score


def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))

def psnr_enchancement(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)

def psnr_denoising(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)

def psnr_sr(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)

def l2_loss(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred) +  tf.square(1e-3)
    return tf.sqrt(tf.reduce_mean(squared_difference, axis=-1))


class LossFunctionNotExists(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message