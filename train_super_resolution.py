import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from mirnet import get_super_resolution_model
import argparse
from utils import charbonnier_loss, CharBonnierLoss, psnr_sr, PSNR
from dataloaders import SRDataLoader
import os
import shutil, glob 
import sys 


parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--loss_function', type=str, default="l2")
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--checkpoint_filepath', type=str, default="checkpoint/saved/super_resolution/")
parser.add_argument('--num_rrg', type=int, default=3)
parser.add_argument('--num_mrb', type=int, default=2)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--summary', type=bool, default=False)
parser.add_argument('--store_model_summary', type=bool, default=False)
parser.add_argument('--scale_factor', type=int, default=4)
parser.add_argument('--downgrade', type=str, default="bicubic")

args = parser.parse_args()

def train():
    train_loader = SRDataLoader(
            scale=args.scale_factor,            
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
            subset='valid'
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

    if args.summary:
        model.summary()

    if args.store_model_summary:
        tf.keras.utils.plot_model(to_file="mirnet_super_resolution.png")

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="val_psnr_sr",
            patience=10,
            mode='max'
        )

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            args.checkpoint_filepath + "/best_model.h5",
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

    if args.loss_function == "charbonnier":
        loss_func = charbonnier_loss

    if args.loss_function == "l1":
        loss_func = tf.keras.losses.MeanAbsoluteError()

    else:
        loss_func = tf.keras.losses.MeanSquaredError()

    if args.use_custom_trainer:
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
            epoch=tf.Variable(1)
        )

        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=args.checkpoint_filepath,
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
                    epochs=args.n_epochs
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
                epochs=args.n_epochs,
                callbacks=[early_stopping_callback, model_checkpoint_callback, reduce_lr_loss]
            )

if __name__ == '__main__':
    train()
