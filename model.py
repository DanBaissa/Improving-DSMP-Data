# model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import SGD

def create_model(conv_size):
    model = Sequential()
    model.add(Conv2D(64, (conv_size, conv_size), activation='relu', padding='same', input_shape=(None, None, 1)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (conv_size, conv_size), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (conv_size, conv_size), activation='relu', padding='same'))

    # here come the decoding layers (upsampling and convolution)
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (conv_size, conv_size), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (conv_size, conv_size), activation='relu', padding='same'))

    # Compile the model with SGD optimizer and momentum
    model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9), loss='mean_squared_error')

    return model
