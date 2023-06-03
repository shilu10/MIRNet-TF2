class UnsuuportedFileExtension(Exception):
    def __init__(self, message):
        self.message = message

        
class InitializationErro(Exception):
    def __init__(self, message):
        self.message = message
        

class DataLoader:
    def __init__(self, dname):
        assert dname in ["lol"], "given dataset name is not valid, supported datasets are ['lol']"  
        #assert type(resize_shape) == int, 'Unknown dtype for resize shape, needed Int' 
        #assert type(batch_size) == int, 'Unknown dtype for batch_size, needed Int' 
        self.dname = dname 
        
    def __lr_image_path(self):
        try:
            train_lr_data_path = os.path.join("lol_dataset", "our485", "low") 
            val_lr_data_path = os.path.join("lol_dataset", "eval15", 'low')
            return train_lr_data_path, val_lr_data_path
        
        except Exception as err:
            return err 
        
    def __hr_image_path(self):
        try: 
            train_hr_data_path = os.path.join("lol_dataset", "our485", "high") 
            val_hr_data_path = os.path.join("lol_dataset", "eval15", 'high') 
            return train_hr_data_path, val_hr_data_path
        
        except Exception as err:
            return err
    
    def __lr_image_files(self):
        try:
            train_lr_data_path, val_lr_data_path = self.__lr_image_path()
            files = list(paths.list_images(train_lr_data_path))
            files_val = list(paths.list_images(val_lr_data_path))
            return files, files_val
        
        except Exception as err:
            return err
    
    def __hr_image_files(self):
        try:
            train_hr_data_path, val_hr_data_path = self.__hr_image_path() 
            files = list(paths.list_images(train_hr_data_path))
            files_val = list(paths.list_images(val_hr_data_path))
            return files, files_val
        
        except Exception as err:
            return err
    
    def __train_tf_dataset(self):
        try: 
            lr_train_files, _ = self.__lr_image_files()
            hr_train_files, _ = self.__lr_image_files()
            tf_dataset = tf.data.Dataset.from_tensor_slices((lr_train_files, hr_train_files)) 
            return tf_dataset
        
        except Exception as err:
            return err 
    
    def __val_tf_dataset(self):
        try: 
            _, lr_val_files = self.__lr_image_files()
            _, hr_val_files = self.__lr_image_files()
            tf_dataset = tf.data.Dataset.from_tensor_slices((lr_val_files, hr_val_files)) 
            return tf_dataset 
        
        except Exception as err:
            return err
    
    def initialize(self):
        try: 
            if self.dname == "lol":
                LOL_DATA_URL = 'https://drive.google.com/uc?id=1DdGIJ4PZPlF2ikl8mNM9V-PdVxVLbQi6'
                if not os.path.exists('lol_dataset'):
                    gdown.download(LOL_DATA_URL, quiet=True)
                    os.system(f'unzip -q lol_dataset')

                if (os.path.exists("lol_dataset.zip") and not os.path.exists("lol_dataset")):
                    os.system(f'unzip -q lol_dataset')
                    
        except Exception as err:
            return err 
        
        else:
            self.train_data_path = os.path.join("lol_dataset", "our485", "low")
            self.val_data_path = os.path.join("lol_dataset", "eval15", 'low')
        
    def __read_img(self, img_fpath): 
        try: 
            raw = tf.io.read_file(img_fpath)
            image = tf.image.decode_png(raw)
            image = tf.cast(image, dtype=tf.float32) / 255.0
            return image
            
        except Exception as err:
            return err
        
    def __load_data(self, lr_img_path, hr_img_path, transform):
        try: 
            lr_img = self.__read_img(lr_img_path)
            hr_img = self.__read_img(hr_img_path)

            return lr_img, hr_img
        
        except Exception as err:
            return err
        
    def get_dataset(self, subset, batch_size, transform=True):
        assert subset in ("train", 'val'), "unsupported split type"
        try:
            if subset == "train":
                tf_ds = self.__train_tf_dataset()
                tf_ds = tf_ds.map(self.__load_data, num_parallel_calls=tf.data.AUTOTUNE).cache()
                if transform:
                    tf_ds = tf_ds.map(random_crop, num_parallel_calls=tf.data.AUTOTUNE).cache()
                    tf_ds = tf_ds.map(random_flip, num_parallel_calls=tf.data.AUTOTUNE).cache()
                    tf_ds = tf_ds.map(random_rotate, num_parallel_calls=tf.data.AUTOTUNE).cache()
                tf_ds = tf_ds.shuffle(buffer_size=50)
                tf_ds = tf_ds.batch(batch_size)
                tf_ds = tf_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                return tf_ds
            
            else:
                tf_ds = self.__val_tf_dataset()
                tf_ds = tf_ds.map(self.__load_data, num_parallel_calls=tf.data.AUTOTUNE).cache()
                if transform:
                    tf_ds = tf_ds.map(random_crop, num_parallel_calls=tf.data.AUTOTUNE).cache()
                    tf_ds = tf_ds.map(random_flip, num_parallel_calls=tf.data.AUTOTUNE).cache()
                    tf_ds = tf_ds.map(random_rotate, num_parallel_calls=tf.data.AUTOTUNE).cache()
                    
                tf_ds = tf_ds.shuffle(buffer_size=5)
                tf_ds = tf_ds.batch(batch_size)
                tf_ds = tf_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                return tf_ds
                
        except Exception as err:
            raise InitializationErro('DataLoader, has not been initialize, use .initalize method')
