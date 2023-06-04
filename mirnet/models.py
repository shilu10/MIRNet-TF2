from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras.layers import Input
from tensorflow.keras import *


def get_enchancement_model(num_rrg: int, num_mrb: int, num_channels: int)->keras.Model: 
    """
        this function, will creaate a low light enchancement model with MIRNET as a 
        backbone model.
        Params:
            num_rrg(dtype: int)      : used to specify number of recursive residual blocks
                                       in mirnet backbone.
            num_mrb(dtype: int)      : used to specify number of multi-scale stream block in
                                       inside the residual recursive group.
            num_channels(dtype: int) : used to specify the initial channels in the mirnet block.
        
        Returns(type: keras.Model)
            returns the functional keras model, for the low light enchancement.
    """
    mirnet_backbone = MIRNet(num_rrg=num_rrg,
                             num_mrb=num_mrb,
                             num_channels=num_channels
                            )
    inputs = Input(shape=(None, None, 3))
    model = Model(inputs, mirnet_backbone.get_model(inputs))
    
    return model

def get_denoising_model(num_rrg:int, num_mrb:int, num_channels: int)->keras.Model: 
    """
        this function, will creaate a image denoising model with MIRNET as a 
        backbone model.
        Params:
            num_rrg(dtype: int)      : used to specify number of recursive residual blocks
                                       in mirnet backbone.
            num_mrb(dtype: int)      : used to specify number of multi-scale stream block in
                                       inside the residual recursive group.
            num_channels(dtype: int) : used to specify the initial channels in the mirnet block.
        
        Returns(type: keras.Model)
            returns the functional keras model, for the image denoising.
    """
    mirnet_backbone = MIRNet(num_rrg=num_rrg,
                             num_mrb=num_mrb,
                             num_channels=num_channels
                            )
    inputs = Input(shape=(None, None, 3))
    model = Model(inputs, mirnet_backbone.get_model(inputs))
    
    return model

def get_super_resolution_model(num_rrg:int,
                num_mrb:int, num_channels: int, scale_factor: float)->keras.Model: 
    """
        this function, will creaate a image super resolution model with MIRNET as a 
        backbone model.
        Params:
            num_rrg(dtype: int)      : used to specify number of recursive residual blocks
                                       in mirnet backbone.
            num_mrb(dtype: int)      : used to specify number of multi-scale stream block in
                                       inside the residual recursive group.
            num_channels(dtype: int) : used to specify the initial channels in the mirnet block.
            scale_factor(dtype:float): used to specify the output resolution scale size. 
        Returns(type: keras.Model)
            returns the functional keras model, for the image super resolution.
    """
    mirnet_backbone = MIRNet(num_rrg=num_rrg,
                             num_mrb=num_mrb,
                             num_channels=num_channels
                            )
    
    inputs = Input(shape=(None, None, 3))
    out = mirnet_backbone.get_model(inputs)
    
    out = Conv2D(64 * (scale_factor ** 2), 3, padding='same')(out)
    out = tf.nn.depth_to_space(out, scale_factor)
    hr = Conv2D(3, kernel_size=(1,1))(out)

    model = Model(inputs, outputs=hr)
    
    return model
