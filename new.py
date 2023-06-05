import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
#from mirnet import get_enchancement_model
import argparse
from utils import charbonnier_loss, CharBonnierLoss, psnr_enchancement, PSNR
from dataloaders import LOLDataLoader
from mirnet import MIRNet


def train():
    dataloader = LOLDataLoader("lol")
    dataloader.initialize()
    train_ds = dataloader.get_dataset(
                    subset="train",
                    batch_size=2,
                    transform=True
                )

    val_ds = dataloader.get_dataset(
                    subset="val",
                    batch_size=1,
                    transform=False
                )

    mir_x = MIRNet(3, 2, 64)
    x = Input(shape=(None, None, 3))
    out = mir_x.get_model(x)
    model = Model(inputs=x, outputs=out)


    
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="val_psnr_enchancement",
            patience=10,
            mode='max'
        )

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            "cjc"+"/best_model.h5",
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

   

    model.compile(
            optimizer=optimizer,
            loss=charbonnier_loss,
            metrics=[psnr_enchancement]
        )

    model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1,
            callbacks=[early_stopping_callback, model_checkpoint_callback, reduce_lr_loss]
        )

