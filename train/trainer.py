from tensorflow.keras.callbacks import EarlyStopping

def train(model, X_train, y_train):

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop]
    )

    return history