import gdown 
import os 
import shutil 
from imutils import paths 
import glob 
import glob 
import numpy as np 
from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras import *
from .utils import random_rotate, random_crop, random_flip



class UnsuuportedFileExtension(Exception):
    def __init__(self, message):
        self.message = message

        
class InitializationErro(Exception):
    def __init__(self, message):
        self.message = message
        

class SIDDDataLoader:
    def __init__(self, dname):
        assert dname in ["sidd"], "given dataset name is not valid, supported datasets are ['lol']"  
        #assert type(resize_shape) == int, 'Unknown dtype for resize shape, needed Int' 
        #assert type(batch_size) == int, 'Unknown dtype for batch_size, needed Int' 
        self.dname = dname 
    
    def __image_files(self):
        try:
            with open('SIDD_Small_sRGB_Only/Scene_Instances.txt') as f:
                instances = f.read()

            instances = instances.split('\n')
            path = 'SIDD_Small_sRGB_Only/Data/'

            noisy_images_path = []
            gt_images_path = []

            for f in instances:
                images_path = path + f + '/'

                for g in os.listdir(images_path):
                    image_path = images_path + g

                    if 'NOISY' in image_path:
                        noisy_images_path.append(image_path)
                    else:
                        gt_images_path.append(image_path)

            return noisy_images_path, gt_images_path
        
        except Exception as err:
            return err
            
    def __noisy_image_files(self):
        try:
            num_train = 150
            
            noisy_images_path, _ = self.__image_files()
            train_noisy_data_path = noisy_images_path[: num_train]
            val_noisy_data_path = noisy_images_path[num_train+1: ]

            return train_noisy_data_path, val_noisy_data_path
        
        except Exception as err:
            return err
    
    def __gt_image_files(self):
        try:
            num_train = 150
            
            _, gt_images_path = self.__image_files()
            train_noisy_data_path = gt_images_path[: num_train]
            val_noisy_data_path = gt_images_path[num_train+1: ]

            return train_noisy_data_path, val_noisy_data_path
        
        except Exception as err:
            return err
    
    def __train_tf_dataset(self):
        try: 
            noisy_train_files, _ = self.__noisy_image_files()
            gt_train_files, _ = self.__gt_image_files()
            
            tf_dataset = tf.data.Dataset.from_tensor_slices((noisy_train_files, gt_train_files)) 
            return tf_dataset
        
        except Exception as err:
            return err 
    
    def __val_tf_dataset(self):
        try: 
            _, noisy_val_files = self.__noisy_image_files()
            _, gt_val_files = self.__gt_image_files()
            tf_dataset = tf.data.Dataset.from_tensor_slices((noisy_val_files, gt_val_files)) 
            return tf_dataset 
        
        except Exception as err:
            return err
    
    def initialize(self):
        try: 
            if self.dname == "sidd":
                SIDD_DATA_PATH = 'https://competitions.codalab.org/my/datasets/download/a26784fe-cf33-48c2-b61f-94b299dbc0f2'

                if not os.path.exists('SIDD_Small_sRGB_Only/Data'):
                    os.system(f'wget {SIDD_DATA_PATH}')
                    os.system(f'unzip -q a26784fe-cf33-48c2-b61f-94b299dbc0f2')

                if (os.path.exists("a26784fe-cf33-48c2-b61f-94b299dbc0f2") and not os.path.exists("SIDD_Small_sRGB_Only/")):
                    os.system(f'unzip -q a26784fe-cf33-48c2-b61f-94b299dbc0f2')
                    
        except Exception as err:
            print(err, "rr")
            return err 
        
    def __read_img(self, img_fpath): 
        try: 
            raw = tf.io.read_file(img_fpath)
            image = tf.image.decode_png(raw, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image
            
        except Exception as err:
            return err
        
    def __load_data(self, lr_img_path, hr_img_path):
        try: 
            lr_img = self.__read_img(lr_img_path)
            hr_img = self.__read_img(hr_img_path)

            return lr_img, hr_img
        
        except Exception as err:
            return err
        
    def __create_tf_dataset(self, tf_ds, batch_size, transform):
        if transform:
           
            tf_ds = tf_ds.map(lambda lr, hr: random_crop(lr, hr), num_parallel_calls=tf.data.AUTOTUNE)
            tf_ds = tf_ds.map(random_flip, num_parallel_calls=tf.data.AUTOTUNE)
            tf_ds = tf_ds.map(random_rotate, num_parallel_calls=tf.data.AUTOTUNE)

            tf_ds = tf_ds.batch(batch_size, drop_remainder=True)
        tf_ds = tf_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return tf_ds
        
    def get_dataset(self, subset, batch_size, transform=True):
        assert subset in ("train", 'val'), "unsupported split type"
        try:
            if subset == "train":
                tf_ds = self.__train_tf_dataset()
                tf_ds = tf_ds.map(self.__load_data, num_parallel_calls=tf.data.AUTOTUNE).cache()
                tf_ds = self.__create_tf_dataset(tf_ds, batch_size, transform)
                return tf_ds
            
            else:
                tf_ds = self.__val_tf_dataset()
                tf_ds = tf_ds.map(self.__load_data, num_parallel_calls=tf.data.AUTOTUNE).cache()
                tf_ds = self.__create_tf_dataset(tf_ds, batch_size, transform)
                return tf_ds
                
        except Exception as err:
            print(err)
            raise InitializationErro('DataLoader, has not been initialize, use .initalize method')
