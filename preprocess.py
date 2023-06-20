# preprocess.py

import rasterio
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
import numpy as np

def load_raster_data(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)

    # Apply a threshold
    data[data > 3000] = 0

    # Apply a log transform (add a small constant to avoid log(0))
    data = np.log(data + 1)


    # Normalize the data
    data = data / np.max(data)

    return data

def resample_raster_to_match(source_path, target_path, destination_path):
    # Open the source and target datasets
    with rasterio.open(source_path) as source:
        with rasterio.open(target_path) as target:
            # Rescale factor is the ratio of target resolution to source resolution
            scale_x = source.res[0] / target.res[0]
            scale_y = source.res[1] / target.res[1]

            # The new shape of the dataset (rounding to nearest integer)
            new_shape = (round(source.height * scale_y), round(source.width * scale_x))

            # The transform for the new (resampled) dataset
            new_transform = rasterio.Affine.scale(scale_x, scale_y) * source.transform

            # The new dataset will be written to 'destination_path'
            with rasterio.open(destination_path, 'w',
                               driver=source.driver,
                               height=new_shape[0],
                               width=new_shape[1],
                               count=source.count,
                               dtype=source.dtypes[0],
                               crs=source.crs,
                               transform=new_transform) as dest:

                # Resample each band
                for i in range(1, source.count + 1):
                    # Read band from source, resample it, and write it to dest
                    resampled_band = source.read(i, out_shape=new_shape, resampling=Resampling.bilinear)
                    dest.write(resampled_band, i)

def preprocess_data(BM_file, DSMP_file, patch_size=16):
    # Resample the Black Marble data to match the DSMP data
    resample_raster_to_match(BM_file, DSMP_file, "BM_resampled.tif")

    # Load the resampled Black Marble data and the DSMP data
    DSMP = load_raster_data(DSMP_file)
    BM_resampled = load_raster_data("BM_resampled.tif")

    # If BM and DSMP are not the same size, trim them to the smallest common size
    min_height = min(DSMP.shape[0], BM_resampled.shape[0])
    min_width = min(DSMP.shape[1], BM_resampled.shape[1])
    DSMP = DSMP[:min_height, :min_width]
    BM_resampled = BM_resampled[:min_height, :min_width]

    # Assume that BM and DSMP now have the same resolution and can be directly used as X and y
    X = np.expand_dims(BM_resampled, axis=2)  # Add a dimension to fit keras Conv2D input shape
    y = np.expand_dims(DSMP, axis=2)  # Add a dimension to fit keras Conv2D input shape

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

