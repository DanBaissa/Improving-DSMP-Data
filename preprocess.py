# preprocess.py

import rasterio
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
import numpy as np
from skimage.util.shape import view_as_windows

def load_raster_data(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)
        # Apply a threshold
        data[data > 1000] = 0
        # Normalize the data
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

def load_original_raster_data(file_path):
    with rasterio.open(file_path) as src:
        return src

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

def preprocess_data(X_files, y_files, patch_size=16):
    X_data = [load_raster_data(file) for file in X_files]
    y_data = [load_raster_data(file) for file in y_files]

    # Find the smallest common size among all rasters
    min_height = min(data.shape[0] for data in X_data + y_data)
    min_width = min(data.shape[1] for data in X_data + y_data)

    # Trim all rasters to the smallest common size
    X_data = [data[:min_height, :min_width] for data in X_data]
    y_data = [data[:min_height, :min_width] for data in y_data]

    # Stack arrays along new third axis
    X = np.stack(X_data, axis=2)
    y = np.stack(y_data, axis=2)

    # Extract patches from the stacked arrays
    X_patches = view_as_windows(X, (patch_size, patch_size, X.shape[2]))
    y_patches = view_as_windows(y, (patch_size, patch_size, y.shape[2]))

    # Reshape patches into the format (num_patches, patch_size, patch_size, num_channels)
    X_patches = X_patches.reshape(-1, patch_size, patch_size, X.shape[2])
    y_patches = y_patches.reshape(-1, patch_size, patch_size, y.shape[2])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_patches, y_patches, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test