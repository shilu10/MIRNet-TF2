def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))

def psnr_enchancement(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


optimizer = keras.optimizers.Adam(learning_rate=1e-4)

early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_psnr_enchancement",
        patience=10,
        mode='max'
    )

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f'checkpoint/mirnet_enhancement',
        monitor="val_psnr_enchancement",
        mode="max",
        save_best_only=True,
        period=1
    )

reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(
        monitor='val_psnr_delight',
        factor=0.5, patience=5,
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
        epochs=20,
        callbacks=[early_stopping_callback, model_checkpoint_callback, reduce_lr_loss]
    )
