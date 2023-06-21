# predict.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import load_raster_data
import rasterio


def predict():
    # Load the model
    model = load_model('trained_model.h5')

    # Load new DSMP data
    new_DSMP_file = "2013_12_eth.tif"  # replace with your new DSMP file
    DSMP_data = load_raster_data(new_DSMP_file)
    DSMP_data_reshaped = np.expand_dims(np.expand_dims(DSMP_data, axis=0), axis=3)  # Add dimensions to fit keras Conv2D input shape

    # Use the model to make predictions on the new DSMP data
    DSMP_improved = model.predict(DSMP_data_reshaped)[0, :, :, 0]

    # Obtain the metadata from the new DSMP data
    with rasterio.open(new_DSMP_file) as src:
        meta = src.meta

    # Update metadata to match the shape and datatype of DSMP_improved
    meta.update({
        'dtype': 'float32',
        'height': DSMP_improved.shape[0],
        'width': DSMP_improved.shape[1],
        'count': 1
    })

    # Create a new raster file with the updated metadata and write DSMP_improved to it
    with rasterio.open('improved_prediction.tif', 'w', **meta) as dest:
        dest.write(DSMP_improved, 1)

    # Plot the original and improved rasters side by side
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Display original raster
    ax[0].imshow(DSMP_data, cmap='gray')
    ax[0].set_title('Original DSMP Data')

    # Display improved raster
    ax[1].imshow(DSMP_improved, cmap='gray')
    ax[1].set_title('Improved DSMP Prediction')

    plt.savefig('Comparison.pdf', format='pdf')
    plt.show()


if __name__ == "__main__":
    predict()
