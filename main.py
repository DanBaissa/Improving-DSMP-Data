# main.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess import preprocess_data, load_raster_data
from model import create_model
from tensorflow.python.client import device_lib

def main():
    # Check if TensorFlow is using the GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # TensorFlow can access a GPU
        print("GPU devices: ", gpus)
    else:
        # TensorFlow cannot access a GPU
        print("No GPU devices available.")

    # Load and preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data("Croped_BM_ETHIOPIA_dec_2013.tif", "2013_12_eth.tif")

    # Create the model
    model = create_model()

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Compute in-sample and out-of-sample "accuracy"
    epsilon = 0.1  # Define your own acceptable error range
    correct_predictions_insample = np.abs(np.squeeze(model.predict(X_train)) - y_train) < epsilon
    correct_predictions_outsample = np.abs(np.squeeze(model.predict(X_test)) - y_test) < epsilon
    accuracy_insample = np.mean(correct_predictions_insample)
    accuracy_outsample = np.mean(correct_predictions_outsample)
    print(f"In-sample accuracy: {accuracy_insample * 100:.2f}%")
    print(f"Out-of-sample accuracy: {accuracy_outsample * 100:.2f}%")

    # Use the model to make predictions on the full DSMP dataset
    BM_data = load_raster_data("BM_resampled.tif")
    BM_data = np.expand_dims(BM_data, axis=(0, 3))  # Add dimensions to fit keras Conv2D input shape
    DSMP_improved = model.predict(BM_data)[0, :, :, 0]

    # Visualize the predicted DSMP raster
    plt.imshow(DSMP_improved, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
