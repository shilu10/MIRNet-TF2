# # Hight Resolution image shape is 96

# supressing tensorflow warning, info, or things.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from tensorflow.keras.models import load_model
import argparse
from mirnet import get_enhancement_model, get_denoising_model, get_super_resolution_model

parser = argparse.ArgumentParser()

parser.add_argument('--scale_factor', type=int, default=4)
parser.add_argument('--saved_model_path', type=str, default="checkpoint/super_resolution/best_model.h5")
parser.add_argument('--tflite_model_path', type=str, default='checkpoint/super_resolution/best_model.tflite')
parser.add_argument('--mode', type=str, default='super_resolution')
parser.add_argument('--num_rrg', type=int, default=3)
parser.add_argument('--num_mrb', type=int, default=1)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--optimize', type=bool, default=False)
parser.add_argument('--quantize', type=str, default="default")


args = parser.parse_args()

def main():
    if args.mode == "super_resolution":
        model = get_super_resolution_model(
            num_rrg=args.num_rrg,
            num_mrb=args.num_mrb,
            num_channels=args.num_channels,
            scale_factor=args.scale_factor
        )

    if args.mode == "denoise":
        model = get_denoising_model(
            num_rrg=args.num_rrg,
            num_mrb=args.num_mrb,
            num_channels=args.num_channels
        )

    else:
        model = get_enhancement_model(
            num_rrg=args.num_rrg,
            num_mrb=args.num_mrb,
            num_channels=args.num_channels
        )

    model.load_weights(args.saved_model_path)

    # trained_model = load_model(args.saved_model_path, custom_objects={'tf': tf})
    # weights = trained_model.get_weights()
    # model.set_weights(weights)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.optimize:
        
        if args.quantize == 'default':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        elif args.quantize == 'fp16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            

    tflite_model = converter.convert()
    with open(args.tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print("Tflite model is saved at: ", args.tflite_model_path)

if __name__ == '__main__':
    main()