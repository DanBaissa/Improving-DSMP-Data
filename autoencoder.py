from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

def create_autoencoder(input_shape, conv_size):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(conv_size, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(conv_size * 2, (3, 3), activation='relu', padding='same')(pool1)
    encoded = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    conv3 = Conv2D(conv_size * 2, (3, 3), activation='relu', padding='same')(encoded)
    up1 = UpSampling2D((2, 2))(conv3)
    conv4 = Conv2D(conv_size, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(conv4)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)

    # Create model
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder
