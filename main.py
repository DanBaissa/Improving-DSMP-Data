# main.py


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanAbsoluteError
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
    X_train, X_test, y_train, y_test = preprocess_data("2013_12_eth.tif", "Croped_BM_ETHIOPIA_dec_2013.tif")

    # Create the model
    model = create_model()

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Compute in-sample and out-of-sample mean absolute error
    # I should convert to MAPE
    mae = MeanAbsoluteError()
    MAE_insample = mae(y_train, np.squeeze(model.predict(X_train))).numpy()
    MAE_outsample = mae(y_test, np.squeeze(model.predict(X_test))).numpy()
    print(f"In-sample Mean Absolute Error: {MAE_insample:.4f}")
    print(f"Out-of-sample Mean Absolute Error: {MAE_outsample:.4f}")

    # Compute and print mean of predicted and true values
    mean_y_train = np.mean(y_train)
    mean_y_test = np.mean(y_test)
    mean_pred_train = np.mean(model.predict(X_train))
    mean_pred_test = np.mean(model.predict(X_test))
    print(f"Mean of true in-sample values: {mean_y_train:.4f}")
    print(f"Mean of predicted in-sample values: {mean_pred_train:.4f}")
    print(f"Mean of true out-of-sample values: {mean_y_test:.4f}")
    print(f"Mean of predicted out-of-sample values: {mean_pred_test:.4f}")

    # Use the model to make predictions on the full DSMP dataset
    BM_data = load_raster_data("BM_resampled.tif")
    BM_data = np.expand_dims(BM_data, axis=(0, 3))  # Add dimensions to fit keras Conv2D input shape
    DSMP_improved = model.predict(BM_data)[0, :, :, 0]

    # Visualize the predicted DSMP raster
    plt.imshow(DSMP_improved, cmap='gray')
    plt.show()

    # Take the logarithm of the data
    log_data = np.log1p(DSMP_improved)  # Apply logarithm to avoid taking log of zero or negative values

    # Plot the logarithm of the raster data
    plt.figure(figsize=(8, 8))
    plt.imshow(log_data, cmap='gray')
    plt.colorbar(label='Logarithm')
    plt.title('Logarithm of Improved DSMP Dataset')
    plt.show()


if __name__ == "__main__":
    main()