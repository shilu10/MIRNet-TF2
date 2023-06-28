import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from mirnet.models import get_enchancement_model
import argparse
from utils import charbonnier_loss, CharBonnierLoss, psnr_enchancement, PSNR, l2_loss, LossFunctionNotExists
from dataloaders.lol_dataloader import LOLDataLoader
from custom_trainer import Trainer
import os
import shutil, glob 
import sys 


def train(gpu, 
      batch_size,
      num_rrg,
      num_mrb, 
      num_channels, 
      summary, 
      store_model_summary, 
      checkpoint_filepath, 
      loss_function, 
      use_custom_trainer,
      epochs):


    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    dataloader = LOLDataLoader("lol")
    dataloader.initialize()
    train_ds = dataloader.get_dataset(
                    subset="train",
                    batch_size=batch_size,
                    transform=True
                )

    val_ds = dataloader.get_dataset(
                    subset="val",
                    batch_size=batch_size,
                    transform=False
                )
    
    model = get_enchancement_model(
            num_rrg=num_rrg,
            num_mrb=num_mrb,
            num_channels=num_channels
        )

    if summary:
        model.summary()

    if store_model_summary:
        tf.keras.utils.plot_model(to_file="mirnet_enchancement.png")

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="val_psnr_enchancement",
            patience=10,
            mode='max'
        )

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath+"/best_model.h5",
            monitor="val_psnr_enchancement",
            mode="max",
            save_best_only=True,
            period=1
        )

    reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(
            monitor='val_psnr_enchancement',
            factor=0.5,
            patience=5,
            verbose=1,
            epsilon=1e-7,
            mode='max'
        )

    if loss_function == "charbonnier":
        loss_func = charbonnier_loss

    if loss_function == "l1":
        loss_func = tf.keras.losses.MeanAbsoluteError()

    if loss_function == "l2":
        loss_func = tf.keras.losses.MeanSquaredError()
    
    else:
        raise LossFunctionNotExists("given loss function is not supported")

    if use_custom_trainer:
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
            epoch=tf.Variable(1)
        )

        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=checkpoint_filepath,
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
                    epochs=epochs
                )

        trainer.train(train_ds, val_ds)

    else:
        model.compile(
                optimizer=optimizer,
                loss=loss_func,
                metrics=[psnr_enchancement]
            )

        model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[early_stopping_callback, model_checkpoint_callback, reduce_lr_loss]
            )

