# supressing tensorflow warning, info, or things.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
import os, shutil 
import cv2
from PIL import Image 
import PIL 
import numpy as np 
from imutils import paths 
from mirnet import get_enhancement_model
import argparse
from tensorflow.keras.preprocessing.image import img_to_array
from utils import get_lowres_image
import time, glob
import tdqm 
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()

parser.add_argument('--test_path', type=str, default="test/LIME/")
parser.add_argument('--plot_results', type=bool, default=False)
parser.add_argument('--checkpoint_filepath', type=str, default="checkpoint/enhancement/")
parser.add_argument('--num_rrg', type=int, default=3)
parser.add_argument('--num_mrb', type=int, default=2)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--summary', type=bool, default=False)
parser.add_argument('--store_model_summary', type=bool, default=False)
parser.add_argument('--file_extension', type=str, default='bmp')
parser.add_argument('--mode', type=str, default="enhancement")

args = parser.parse_args()

def test(model):

    lowlight_test_images_path = args.test_path
    test_files = glob.glob(lowlight_test_images_path + f"*.{args.file_extension}")

    for test_file in tqdm.tqdm(test_files, total=len(test_files)):
        
        filename = test_file.split("/")[-1]
        lr_img = cv2.imread(test_file)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        
        # for resizing specific model data to specific dim.
        lr_img = get_lowres_image(lr_img, mode=args.mode)

        inputs = img_to_array(img_lowlight)
        inputs = np.expand_dims(inputs, axis=0)
        t = time.time()

        enhanced_image = model.predict(inputs, verbose=False)
        enhanced_image = enhanced_image[0]
        print("Time taken for inference: ", time.time() - t)

        if args.plot_results:
            plt.figure()
            plt.subplot(121)
            plt.imshow(original_img)
            
            plt.subplot(122)
            plt.imshow(enhanced_image)

            plt.show()
        
        save_file_dir = lowlight_test_images_path.replace('test', 'results')
        save_file_path = save_file_dir + filename
        cv2.imwrite(save_file_path, cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    model = get_enchancement_model(
        num_rrg=args.num_rrg,
        num_mrb=args.num_mrb,
        num_channels=args.num_channels
    )
    
    model.load_weights(args.checkpoint_filepath + 'best_model.h5')

    if args.summary:
        model.summary()

    if args.store_model_summary:
        tf.keras.utils.plot_model(to_file="mirnet_enhancement.png")

    test(model)
