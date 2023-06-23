# preprocess.py

import rasterio
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
import numpy as np


def load_raster_data(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)

        # Apply a threshold
        data[data > 1000] = 0

        # Normalize the data
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        return data

def load_raster_data_min_max(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)

        # Apply a threshold
        data[data > 1000] = 0

        return data


def resample_raster_to_match(source_path, target_path, destination_path):
    with rasterio.open(source_path) as source:
        with rasterio.open(target_path) as target:
            scale_x = source.res[0] / target.res[0]
            scale_y = source.res[1] / target.res[1]

            new_shape = (round(source.height * scale_y), round(source.width * scale_x))
            new_transform = rasterio.Affine.scale(scale_x, scale_y) * source.transform

            with rasterio.open(destination_path, 'w',
                               driver=source.driver,
                               height=new_shape[0],
                               width=new_shape[1],
                               count=source.count,
                               dtype=source.dtypes[0],
                               crs=source.crs,
                               transform=new_transform) as dest:
                for i in range(1, source.count + 1):
                    resampled_band = source.read(i, out_shape=new_shape, resampling=Resampling.bilinear)
                    dest.write(resampled_band, i)


def preprocess_data(BM_files, DSMP_files, patch_size=16):
    X = []
    y = []

    for BM_file, DSMP_file in zip(BM_files, DSMP_files):
        resample_raster_to_match(BM_file, DSMP_file, "BM_resampled.tif")
        DSMP = load_raster_data(DSMP_file)
        BM_resampled = load_raster_data("BM_resampled.tif")

        min_height = min(DSMP.shape[0], BM_resampled.shape[0])
        min_width = min(DSMP.shape[1], BM_resampled.shape[1])
        DSMP = DSMP[:min_height, :min_width]
        BM_resampled = BM_resampled[:min_height, :min_width]

        y.append(np.expand_dims(BM_resampled, axis=2))
        X.append(np.expand_dims(DSMP, axis=2))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
