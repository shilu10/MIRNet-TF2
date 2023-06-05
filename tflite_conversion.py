import tensorflow as tf 
from tensorflow.keras.models import load_model
import argparse
from mirnet import get_enchancement_model, get_denoising_model, get_super_resolution_model

parser = argparse.ArgumentParser()

parser.add_argument('--input_shape', type=tuple, default=(128, 128, 3))
parser.add_argument('--scale_factor', type=int, default=4)
parser.add_argument('--saved_model_path', type=str, default="checkpoint/saved/super_resolution/best_model.h5")
parser.add_argument('--scale_factor', type=int, default=4)


SCALE = 4
INPUT_SHAPE=(512, 512, 3)

MODEL_PATH = "./saved/models/interp_esr.h5"
TFLITE_MODEL_PATH = './saved/models/esrgan.tflite'

def main():

    trained_model = load_model(MODEL_PATH, custom_objects={'tf': tf})
    weights = trained_model.get_weights()
    
    model = rrdb_net(input_shape=INPUT_SHAPE,scale_factor=SCALE)
    model.set_weights(weights)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)













if __name__ == '__main__':
    main()