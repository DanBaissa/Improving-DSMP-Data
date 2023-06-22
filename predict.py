from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_raster_data
import rasterio
import os

def predict(DSMP_folder, output_folder, model_path):
    # Load the model
    model = load_model(model_path)

    # Load and preprocess the data
    DSMP_files = [os.path.join(DSMP_folder, filename) for filename in os.listdir(DSMP_folder)]
    for DSMP_file in DSMP_files:
        DSMP = load_raster_data(DSMP_file)

        DSMP = np.expand_dims(DSMP, axis=(0, 3))  # Add dimensions to fit keras Conv2D input shape
        DSMP_improved = model.predict(DSMP)[0, :, :, 0]

        # Save the improved raster
        with rasterio.open(DSMP_file) as src:
            meta = src.meta
        with rasterio.open(os.path.join(output_folder, 'improved_' + os.path.basename(DSMP_file)), 'w', **meta) as dest:
            dest.write(DSMP_improved, 1)
