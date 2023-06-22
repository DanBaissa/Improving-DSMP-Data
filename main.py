#main.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanAbsoluteError
import tensorflow as tf
from preprocess import preprocess_data, load_raster_data
from model import create_model
from tensorflow.python.client import device_lib
import os
import rasterio

def main(DSMP_dir, BM_dir, epochs, conv_size, output_location):
    # Check if TensorFlow is using the GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # TensorFlow can access a GPU
        print("GPU devices: ", gpus)
    else:
        # TensorFlow cannot access a GPU
        print("No GPU devices available.")

    # Load and preprocess the data
    DSMP_files = [DSMP_dir + '/' + f for f in os.listdir(DSMP_dir) if f.endswith('.tif')]
    BM_files = [BM_dir + '/' + f for f in os.listdir(BM_dir) if f.endswith('.tif')]
    X_train, X_test, y_train, y_test = preprocess_data(BM_files, DSMP_files)

    # Create the model
    model = create_model(conv_size) # pass conv_size to create_model function

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test, y_test))

    # Save the model
    model.save('trained_model.h5')

    # Plotting loss function
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Loss_Plot.png')

    # Comparing the old and predicted rasters
    DSMP_data = load_raster_data(DSMP_files[0])  # Load first DSMP file for comparison
    DSMP_data = np.expand_dims(DSMP_data, axis=(0, 3))  # Add dimensions to fit keras Conv2D input shape
    predicted_data = model.predict(DSMP_data)[0, :, :, 0]

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(DSMP_data[0, :, :, 0], cmap='gray')
    axes[0].set_title('Original DSMP')
    axes[1].imshow(predicted_data, cmap='gray')
    axes[1].set_title('Predicted DSMP')
    plt.savefig('Comparison.png')
    plt.show()


if __name__ == "__main__":
    main()
