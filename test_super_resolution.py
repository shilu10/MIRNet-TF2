import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
import os, shutil 
import cv2
from PIL import Image 
import PIL 
import numpy as np 
from imutils import paths 
from glob import glob 
from mirnet import get_super_resolution_model
import argparse
from tensorflow.keras.preprocessing.image import img_to_array


parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, default="test/super_resolution")
parser.add_argument('--plot_results', type=bool, default=False)
parser.add_argument('--checkpoint_filepath', type=str, default="checkpoint/saved/super_resolution/")
parser.add_argument('--num_rrg', type=int, default=3)
parser.add_argument('--num_mrb', type=int, default=2)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--scale_factor', type=int, default=4)
parser.add_argument('--summary', type=bool, default=False)
parser.add_argument('--store_model_summary', type=bool, default=False)
parser.add_argument('--file_extension', type=str, default='png')

args = parser.parse_args()

def test(model):

    lowlight_test_images_path = args.test_path

    for test_file in glob.glob(lowlight_test_images_path + f"*.{args.file_extension}"):
        filename = test_file.split("/")[-1]
        data_lowlight_path = test_file
        original_img = Image.open(data_lowlight_path)
        original_size = (np.array(original_img).shape[1], np.array(original_img).shape[0])
        original_img = cv2.resize(original_img, (256, 256))

        img_lowlight = cv2.imread(data_lowlight_path)
        img_lowlight = cv2.resize(img_lowlight, (256, 256))
        
        y = img_to_array(img_lowlight)
        inputs = np.expand_dims(y, axis=0)
        t = time.time()
        out = model.predict(inputs, verbose=False)
        print(time.time() - t)

        out_img_y = out[0]
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1], 3))

        enhanced_image = PIL.Image.fromarray(np.uint8(out_img_y))

        if args.plot_results:
            plt.figure()
            plt.subplot(121)
            plt.imshow(original_img/255.0)
            
            plt.subplot(122)
            plt.imshow(enhanced_image/255.0)
        
        save_file_dir = lowlight_test_images_path.replace('test', 'results')
        save_file_path = save_file_dir + "/" + filename
        enhanced_image.save(fsave_file_path)


if __name__ == '__main__':
    model = get_super_resolution_model(
            num_rrg=args.num_rrg,
            num_mrb=args.num_mrb,
            num_channels=args.num_channels,
            scale_factor=args.scale_factor
        )

    model.load_weights(args.checkpoint_filepath + '/best_model.h5')

    if args.summary:
        model.summary()

    if args.store_model_summary:
        tf.keras.utils.plot_model(to_file="mirnet_super_resolution.png")

    test(model)
