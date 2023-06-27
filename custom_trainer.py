import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import *
from tensorflow.keras.layers import * 
from datetime import datetime
from collections import *
from tqdm import tqdm



class Trainer:
    def __init__(self, model, loss_func, metric_func, optimizer, ckpt, ckpt_manager, epochs):
        self.model = model 
        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metric_func = metric_func

        self.metric_tracker = keras.metrics.Mean() 
        self.val_metric_tracker = keras.metrics.Mean() 
        
        self.loss_tracker = keras.metrics.Mean()
        self.val_loss_tracker = keras.metrics.Mean()
        
        log_dir = 'loss/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '/train'
        self.train_writer = tf.summary.create_file_writer(log_dir)
        
        log_dir = 'loss/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '/val'
        self.val_writer = tf.summary.create_file_writer(log_dir)
    
    #@tf.function
    def train_step(self, train_batch):

        source_img_batch, target_img_batch = train_batch
        with tf.GradientTape() as tape: 
            pred_image_batch = self.model(source_img_batch)
            #loss_val = self.loss_func(pred_image_batch, target_img_batch)
            loss_val = tf.reduce_mean(tf.sqrt(tf.square(target_img_batch - pred_image_batch) + tf.square(1e-3)))

        params = self.model.trainable_variables
        grads = tape.gradient(loss_val, params)
        
        self.optimizer.apply_gradients(zip(grads, params))
        self.loss_tracker.update_state(loss_val)

        # train psnr metric tracker.
        train_psnr = self.metric_func(pred_image_batch, target_img_batch)
        self.metric_tracker.update_state(train_psnr)

        train_result = {
            "loss": self.loss_tracker.result(),
            "psnr": self.metric_tracker.result()
        }
        
        return train_result
    
    def test_step(self, val_batch):

        source_img_batch, target_img_batch = val_batch
        pred_image_batch = self.model(source_img_batch) 
        
        loss_val = self.loss_func(pred_image_batch, target_img_batch)
        self.val_loss_tracker.update_state(loss_val)
        
        # psnr metric tracker.
        val_psnr = self.metric_func(pred_image_batch, target_img_batch)
        self.val_metric_tracker.update_state(val_psnr)

        val_result = {
            "loss": self.val_loss_tracker.result(),
            "psnr": self.val_metric_tracker.result()
        }
        
        return val_result
    
    def compute_loss(self, original, enchanced, curve_params):
        illumination_loss = 200 * self.illumination_smoothness_loss_func(curve_params)
        spatial_constancy_loss = tf.reduce_mean(
            self.spatial_consistency_loss_func(enchanced, original)
        )
        color_constancy_loss = 5 * tf.reduce_mean(self.color_constancy_loss_func(enchanced))
        exposure_loss = 10 * tf.reduce_mean(self.exposure_loss_func(enchanced)) 
        
        return spatial_constancy_loss, exposure_loss, illumination_loss, color_constancy_loss
    
    def load_weights(self, filepath):
        pass 
    
    def save_weights(self, filepath):
        pass 
    
    def train(self, train_ds, val_ds):
        history = defaultdict(list)
        
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        
        for epoch in range(self.epochs): 
            print(f"Epoch: {epoch}: ")
            for step, training_batch in tqdm(enumerate(train_ds), total=len(train_ds)): 
                train_result = self.train_step(training_batch)
                train_loss = train_result["loss"].numpy()
                train_psnr = train_result["psnr"].numpy()

            for step, val_batch in enumerate(val_ds):
                val_result = self.test_step(val_batch)
                val_loss = val_result["loss"].numpy()
                val_psnr = val_result['psnr'].numpy()
        
            self.ckpt.epoch.assign_add(1)
            history["train_loss"].append(train_loss) 
            history["val_loss"].append(val_loss)
            

            with self.train_writer.as_default(step=epoch):
                tf.summary.scalar('train_loss', train_loss)
                tf.summary.scalar('train_psnr', train_psnr)

            with self.val_writer.as_default(step=epoch):
                tf.summary.scalar('val_loss', val_loss)
                tf.summary.scalar('train_psnr', val_psnr)
            

            print(f'train_loss: {train_loss}, train_psnr: {train_psnr} \n')
            print(f'val_loss: {val_loss}, val_psnr: {val_psnr}  \n')
            
            #reset states of training step
            self.loss_tracker.reset_states()
            self.metric_tracker.reset_states()
            
            # reset states of the val step
            self.val_loss_tracker.reset_states()
            self.val_metric_tracker.reset_states()
               
            if epoch %2 == 0: 
                save_path = self.ckpt_manager.save()
                
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.epoch), save_path))
            
        return self.model
    
    def compute_psnr(self):
        pass 