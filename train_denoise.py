import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from mirnet import get_denoising_model
import argparse
from utils import charbonnier_loss, CharBonnierLoss, psnr_denoising, PSNR
from dataloaders import SIDDDataLoader
from custom_trainer import Trainer
import os
import shutil, glob 
import sys 


parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--loss_function', type=str, default="l1")
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--checkpoint_filepath', type=str, default="checkpoint/saved/denoise/")
parser.add_argument('--num_rrg', type=int, default=3)
parser.add_argument('--num_mrb', type=int, default=2)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--summary', type=bool, default=False)
parser.add_argument('--store_model_summary', type=bool, default=False)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--use_custom_trainer', type=bool, default=False)

args = parser.parse_args()

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataloader = SIDDDataLoader("sidd")
    dataloader.initialize()
    train_ds = dataloader.get_dataset(
                    subset="train",
                    batch_size=args.batch_size,
                    transform=True
                )

    val_ds = dataloader.get_dataset(
                    subset="val",
                    batch_size=args.batch_size,
                    transform=False
                )

    model = get_denoising_model(
            num_rrg=args.num_rrg,
            num_mrb=args.num_mrb,
            num_channels=args.num_channels
        )

    if args.summary:
        model.summary()

    if args.store_model_summary:
        tf.keras.utils.plot_model(to_file="mirnet_denoise.png")

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="val_psnr_denoising",
            patience=10,
            mode='max'
        )

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            args.checkpoint_filepath+"/best_model.h5",
            monitor="val_psnr_denoising",
            mode="max",
            save_best_only=True,
            period=1
        )

    reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(
            monitor='val_psnr_denoising',
            factor=0.5,
            patience=5,
            verbose=1,
            epsilon=1e-7,
            mode='max'
        )

    if args.loss_function == "charbonnier":
        loss_func = charbonnier_loss

    if args.loss_function == "l1":
        loss_func = tf.keras.metrics.MeanAbsoluteError()

    else:
        loss_func = tf.keras.metrics.MeanSquaredError()

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
                    metric_func=psnr_enchancement,
                    optimizer=optimizer,
                    ckpt=checkpoint,
                    ckpt_manager=manager,
                    epochs=args.epcohs
                )

        trainer.train(train_ds, val_ds)
        
    
    else: 
        model.compile(
                optimizer=optimizer,
                loss=loss_func,
                metrics=[psnr_denoising]
            )

        model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.n_epochs,
                callbacks=[early_stopping_callback, model_checkpoint_callback, reduce_lr_loss]
            )
    


if __name__ == '__main__':
    train()
