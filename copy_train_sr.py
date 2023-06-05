import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from mirnet import get_super_resolution_model
import argparse
from utils import charbonnier_loss, CharBonnierLoss, psnr_sr, PSNR
from dataloaders import SRDataLoader



def train(custom_training=True, epochs=1):
    train_loader = SRDataLoader(
            scale=4,            
            downgrade=args.downgrade,
            subset='train'
        )      
                         
    train_ds = train_loader.dataset(
            batch_size=args.batch_size,        
            random_transform=True, 
            repeat_count=None
        )  

    val_loader = SRDataLoader(
            scale=args.scale_factor,            
            downgrade=args.downgrade,
            subset='val'
        )      
                         
    val_ds = val_loader.dataset(
            batch_size=args.batch_size,        
            random_transform=False, 
            repeat_count=None
        )  

    model = get_super_resolution_model(
            num_rrg=args.num_rrg,
            num_mrb=args.num_mrb,
            num_channels=args.num_channels,
            scale_factor=args.scale_factor
        )

    

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="val_psnr_sr",
            patience=10,
            mode='max'
        )

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            "checkpoint/saved/" + "/best_model.h5",
            monitor="val_psnr_sr",
            mode="max",
            save_best_only=True,
            period=1
        )

    reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(
            monitor='val_psnr_sr',
            factor=0.5,
            patience=5,
            verbose=1,
            epsilon=1e-7,
            mode='max'
        )

 

    loss_func = tf.keras.metrics.MeanSquaredError()
    
    
    if custom_training:
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
            epoch=tf.Variable(1)
        )

        manager = tf.train.CheckpointManager(
            checkpoint,
            directory="checkpoint/sr/",
            max_to_keep=5
        )

        status = checkpoint.restore(manager.latest_checkpoint)
        trainer = Trainer(
                    model=model,
                    loss_func=loss_func,
                    metric_func=psnr_sr,
                    optimizer=optimizer,
                    ckpt=checkpoint,
                    ckpt_manager=manager,
                    epochs=epochs
                )

        trainer.train(train_ds, val_ds)

    else:
        model.compile(
                optimizer=optimizer,
                loss=loss_func,
                metrics=[psnr_sr]
            )

        model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[early_stopping_callback, model_checkpoint_callback, reduce_lr_loss]
            )

if __name__ == '__main__':
    train()
