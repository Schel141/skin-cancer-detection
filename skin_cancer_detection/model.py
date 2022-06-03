import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping


def initialize_model():

    model = Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1. / 255))
    model.add(layers.Conv2D(16, (3,3), input_shape=(75, 100, 3), padding='same', activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(32, (2,2), padding='same', activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu')) # intermediate layer
    model.add(layers.Dense(7, activation='softmax'))

    return model


def compile_model(model):
    model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy', # "sparse_" allows to avoid one-hot-encoding the target
    metrics = ['accuracy','Recall', 'Precision'])
    return model


es = EarlyStopping(patience = 20, restore_best_weights = True)

history = model.fit(X_train_stack, y_train,
                    validation_split = 0.3,
                    callbacks = [es],
                    epochs = 200,
                    batch_size = 32)
