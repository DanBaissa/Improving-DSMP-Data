# predict_new_data.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import preprocess_data, load_raster_data, save_raster_data
import rasterio

def preprocess_new_data(DSMP_file):
    # Load the DSMP data
    DSMP = load_raster_data(DSMP_file)

    # Apply a threshold
    DSMP[DSMP > 1000] = 0

    # Normalize the data
    DSMP = (DSMP - np.min(DSMP)) / (np.max(DSMP) - np.min(DSMP))

    # Expand dimensions to fit keras Conv2D input shape
    DSMP = np.expand_dims(DSMP, axis=(0, 3))

    return DSMP

def predict_on_new_data():
    # Load the trained model
    model = load_model('my_model.h5')

    # Load and preprocess the new DSMP data
    DSMP_new = preprocess_new_data("new_DSMP_data.tif")

    # Use the model to make predictions
    DSMP_improved = model.predict(DSMP_new)

    # Save the improved DSMP data to a .TIF file
    with rasterio.open('new_DSMP_data.tif') as src:
        meta = src.meta
    meta.update(dtype=rasterio.float32)

    with rasterio.open("DSMP_improved.tif", 'w', **meta) as dst:
        dst.write(DSMP_improved.astype(rasterio.float32), 1)

    # Visualize the predicted DSMP raster
    plt.imshow(np.log1p(DSMP_improved.squeeze()), cmap='gray')
    plt.title('Improved DSMP Dataset Log Scale')
    plt.savefig('Improved.pdf', format='pdf')
    plt.show()

if __name__ == "__main__":
    predict_on_new_data()
