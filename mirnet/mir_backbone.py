import time, os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as ans 
from tqdm import tqdm 
import shutil 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import Model
from tensorflow.keras import layers 
from tensorflow.keras.layers import *
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import *
from datetime import datetime
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense, Input, UpSampling2D, Conv2DTranspose, Conv2D, add, Add,\
                    Lambda, Concatenate, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, \
                    Add, LayerNormalization, Activation, LeakyReLU, SeparableConv2D, Softmax, MaxPooling2D
try:
    import tensorflow_addons as tfa 
except:
    !pip install tensorflow_addons
    import tensorflow_addons as tfa
    from tensorflow_addons.layers import InstanceNormalization
try:
    import gdown 
except:
    !pip install gdown --quiet
    import gdown
    
try: 
    from imutils import paths 
except:
    !pip install imutils  --quiet
    from imutils import paths
from collections import defaultdict
from tqdm import tqdm
import PIL
import glob
from PIL import Image
import cv2


class MIRNet(keras.Model):
    """
        this class, will be used as a backbone mirnet model class, which will be backbone
        model for all image restoration tasks.
        Methods:
            selective_kernel_feature_fusion(scope: public): builds a skff block.
            channel_attention(scope: public): builds a channel attention block.
            spatial_attention(scope: public): builds a spatial attention block.
            dual_attention_unit(scope: public): builds a dual_attention_unit block.
            downsampling(scope: public): builds a downsampling block.
            upsampling(scope: public): builds a upsampling block.
            multiscale_residual_block(scope: public): builds a mrb.
            recursive_residual_group(scope: public): builds a rrg.
            
        Attrs:
            num_rrg(dtype: int): number of recursive residual groups in model.
            num_mrb(dtype: int): number of num_mrb in model.
            num_channels(dtype: int): number of num_channels groups in model.
    """
    def __init__(self, num_rrg, num_mrb, num_channels):
        super(MIRNet, self).__init__()
        self.num_rrg = num_rrg 
        self.num_mrb = num_mrb 
        self.channels = num_channels
        
    def selective_kernel_feature_fusion(self, L1, L2, L3):
        """
            this method is used for adjusting the receptive field of the neurons
            dynamically using two methods fuse and select, instead of just concatinating.
            Params:
                L1(dtype: tf.Tensor): convolutional stream1.
                L2(dtype: tf.Tensor): convolutional stream2.
                L3(dtype: tf.Tensor): convolutional stream3.
                
            Return(type: tf.Tensor): 
                returns a recalibrated convolutional sreams.
        """
        n_channels = list(L1.shape)[-1]
        gap = GlobalAveragePooling2D()
        channel_downscaling_conv = Conv2D(filters=n_channels//8, kernel_size=(1,1))
        channel_upsampling_conv1 = Conv2D(filters=n_channels, kernel_size=(1, 1))
        channel_upsampling_conv2 = Conv2D(filters=n_channels, kernel_size=(1, 1))
        channel_upsampling_conv3 = Conv2D(filters=n_channels, kernel_size=(1, 1))

        # combining all three scale streams
        L = L1 + L2 + L3 
        # calculate the  channel-wise statistics
        s = gap(L)
        s =  tf.reshape(s, shape=(-1, 1, 1, n_channels))

        # applying channel upsampling to the z(feature vector) to get feature descriptor
        # v = feature descriptor, z = feature vector
        z =  channel_downscaling_conv(s)
        v1 = channel_upsampling_conv1(z)
        v2 = channel_upsampling_conv2(z)
        v3 = channel_upsampling_conv3(z)

        # applying the softmax to v1, v2, v3, to get a attention activation.
        # s = attention activation for a feature descriptor.
        s1 = Softmax()(v1)
        s2 = Softmax()(v2)
        s3 = Softmax()(v3)

        # adaptively recabirating the feature maps.
        L1 = s1 * L1 
        L2 = s3 * L2 
        L3 = s3 * L3 

        # Global descriptor.
        U = Add()([L1, L2, L3])
        return U

    def channel_attention(self, M: tf.Tensor)->tf.Tensor:
        """
            this method, is used to  exploits the inter-channel relationships of the convolutional feature
            maps by applying squeeze and excitation operations
            Params:
                M(type: tf.Tensor): input feature maps.
            Returns(type: tf.Tensor):
                returns the calibrated feature map.
        """
        # M = feature maps.
        n_channels =list(M.shape)[-1]
        gap = GlobalAveragePooling2D()

        # squeeze operation, to extract the feature descriptor, by encoding global context
        d = gap(M)
        d = tf.reshape(d, shape=(-1,1,1,n_channels))

        # excitation operation 
        conv_1 = Conv2D(filters=n_channels//8, kernel_size=(1, 1), activation="relu")(d)
        conv_2 = Conv2D(filters=n_channels, kernel_size=(1, 1))(conv_1)

        # sgimoid gating, to extract the d_hat(activation)
        d_hat = sigmoid(conv_2)

        # rescaling the feature map with activation d_hat
        return M * d_hat
    
    def spatial_attention(self, M: tf.Tensor)->tf.Tensor: 
        """
            this method, is used to  exploits the inter-spatial dependencies of the convolutional feature
            maps by applying conv and sigmoid gating and maxpooling.
            Params:
                M(type: tf.Tensor): input feature maps.
            Returns(type: tf.Tensor):
                returns the calibrated feature map.
        """
        # M = feature maps.
        gap = tf.reduce_max(M, axis=-1)
        gap = tf.expand_dims(gap, axis=-1)

        gmp = tf.reduce_mean(M, axis=-1)
        gmp = tf.expand_dims(gmp, axis=-1)

        # concat the gap output and maxpool output to generate a feature map f.
        f = Concatenate(axis=-1)([gap, gmp])

        # passing f to conv2 and sigmoid to get a spatial attention map.(f_hat)
        conv_out = Conv2D(filters=1, kernel_size=(1,1))(f)
        f_hat = sigmoid(conv_out)

        # recalibrating the feature map M, with the spatial attention map (f_hat)
        return M * f_hat
    
    def dual_attention_unit(self, X: tf.Tensor)->tf.Tensor:
         """
            this method, is used to extract the feature map of the input image, by 
            using channel and spatial attention
            Params:
                M(type: tf.Tensor): input image.
            Returns(type: tf.Tensor):
                returns the calibrated feature map.
        """
        n_channels = list(X.shape)[-1]

        # extract the feature maps (high-level features)
        M = Conv2D(n_channels, kernel_size=(3,3), padding='same')(X)
        M = ReLU()(M)
        M = Conv2D(n_channels, kernel_size=(3,3), padding='same')(M)

        # passing the feature map(M), to channel and spatial attention, to pass info across
        # both channels and spatial dimension of the feature tensor.(conv tensor)
        channel_rescaled_M = self.channel_attention(M)
        spatial_rescaled_M = self.spatial_attention(M)

        concat = Concatenate(axis=-1)([channel_rescaled_M, spatial_rescaled_M])
        conv_out = Conv2D(n_channels, kernel_size=(1,1))(concat)
        return Add()([X, conv_out])
    
    def downsampling(self, X: tf.Tensor)->tf.Tensor: 
        """
            this method, used for downsampling the feature maps using residual blocks(main
            and skip branch), uses the antialiasing downsampling.
            Params:
                Params:
                M(type: tf.Tensor): input image.
            Returns(type: tf.Tensor):
                returns the downsampled X.
        """
        n_channels = list(X.shape)[-1]

        #upper branch (main branch)
        upper_branch = Conv2D(filters=n_channels, kernel_size=(1,1))(X)
        upper_branch = ReLU()(upper_branch)
        upper_branch = Conv2D(filters=n_channels, kernel_size=(3,3), padding='same')(upper_branch)
        upper_branch = ReLU()(upper_branch)

        # antialiasing downsampling using maxpooling
        upper_branch = MaxPooling2D()(upper_branch)
        upper_branch = Conv2D(filters=n_channels * 2, kernel_size=(1,1))(upper_branch)

        # antialiasing downsampling in skip connection.
        skip_branch = MaxPooling2D()(X)
        skip_branch = Conv2D(filters=n_channels * 2, kernel_size=(1,1))(skip_branch)

        return Add()([skip_branch, upper_branch])
    
    def upsampling(self, X: tf.Tensor)->tf.Tensor: 
        """
            this method, used for upsampling the feature maps or image using 
            residual blocks(main and skip branch), uses the BiLinear upsampling.
            Params:
                X(type: tf.Tensor): input image.
            Returns(type: tf.Tensor):
                returns the upsampled X.
        """
        n_channels = list(X.shape)[-1]

        # upprt barch upsampling with bilinear upsampler.
        upper_branch = Conv2D(filters=n_channels, kernel_size=(1,1))(X)
        upper_branch = ReLU()(upper_branch)
        upper_branch = Conv2D(filters=n_channels, kernel_size=(3,3), padding='same')(upper_branch)
        upper_branch = ReLU()(upper_branch)

        # bilinear upsampling (upsampling convolution).
        upper_branch = UpSampling2D()(upper_branch)
        upper_branch = Conv2D(filters=n_channels//2, kernel_size=(1, 1))(upper_branch)

        # bilinear upsampling for skip connection branch
        skip_branch = UpSampling2D()(X)
        skip_branch = Conv2D(filters=n_channels // 2, kernel_size=(1,1))(skip_branch)

        return Add()([upper_branch, skip_branch])
    
    def multiscale_residual_block(self, X: tf.Tensor)->tf.Tensor:
        """
            this method used to implement the multiscale residual block, which will pass 
            the input image through multiple convolution streams, to get a different scaled 
            image representation.
            Params:
                X(type: tf.Tensor): input image.
            Returns(type: tf.Tensor):
                returns the preprocesed X.
        """
        # downsampled features(multi scale streams)
        level_1 = X
        level_2 = self.downsampling(level_1)
        level_3 = self.downsampling(level_2)

        # appling dual attention to downsampled features
        level_1_DAU = self.dual_attention_unit(level_1)
        level_2_DAU = self.dual_attention_unit(level_2)
        level_3_DAU = self.dual_attention_unit(level_3)

        # applying the skff to dau features.
        # level1_skff = l1_dau, us(l2_dau), us(us(l3_dau))
        # level2_skff = ds(l1_dau), l2_dau, us(l3_dau)
        # level3_skff = ds(ds(l1_dau)), ds(l2_dua), l3_dau
        level_1_SKFF = self.selective_kernel_feature_fusion(level_1_DAU, self.upsampling(level_2_DAU),
                                                                        self.upsampling(self.upsampling(level_3_DAU)))
        
        level_2_SKFF = self.selective_kernel_feature_fusion(self.downsampling(level_1_DAU), level_2_DAU,
                                                                                            self.upsampling(level_3_DAU))
        
        level_3_SKFF = self.selective_kernel_feature_fusion(self.downsampling(self.downsampling(level_1_DAU)),
                                                                                self.downsampling(level_2_DAU), level_3_DAU)

        # DAU2 
        level_1_DAU_2 = self.dual_attention_unit(level_1_SKFF)
        level_2_DAU_2 = self.upsampling((self.dual_attention_unit(level_2_SKFF)))
        level_3_DAU_2 = self.upsampling(self.upsampling(self.dual_attention_unit(level_3_SKFF)))

        # SKFF 2
        SKFF = self.selective_kernel_feature_fusion(level_1_DAU_2, level_2_DAU_2, level_3_DAU_2)
        conv = Conv2D(self.channels, kernel_size=(3, 3), padding="same")(SKFF)

        return Add()([X, conv])
    
    def recursive_residual_group(self, X: tf.Tensor)->tf.Tensor:
        """
            this method is used to implement the recursive residual group module,
            which uses multiple multi-scale conv blocks.
            Params:
                 X(type: tf.Tensor): input image.
            Returns(type: tf.Tensor):
                returns the preprocesed X.
        """
        conv_1 = Conv2D(self.channels, kernel_size=(3,3), padding='same')(X)
        for _ in range(self.num_mrb):
            conv_1 = self.multiscale_residual_block(conv_1)

        conv_2 = Conv2D(self.channels, kernel_size=(3,3), padding='same')(conv_1)

        return Add()([conv_2, X])
    
    def get_model(self, X): 
        """
            this method, will returns the output which is generated by the recursive
            residual block, which is mir backbone output.
            Params:
                X(type: tf.Tensor): input image.
            Returns(type: tf.Tensor):
                returns the preprocesed X.
        """
        X1 = Conv2D(self.channels, kernel_size=(3,3), padding='same')(X)

        for _ in range(self.num_rrg):
            X1 = self.recursive_residual_group(X1)

        conv = Conv2D(3, kernel_size=(3,3), padding='same')(X1)
        output = Add()([X, conv])

        return output
