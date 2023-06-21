# preprocess.py

from rasterio import open
import numpy as np
from sklearn.model_selection import train_test_split


def load_raster_data(file_path):
    """
    Load raster data from the given file path.

    Parameters:
    file_path (str): The file path of the raster file to load.

    Returns:
    A numpy array containing the raster data.
    """
    with open(file_path) as f:
        data = f.read(1)

    return data


def preprocess_data(file_paths, labels_file_path):
    """
    Load and preprocess the data.

    Parameters:
    file_paths (list of str): List of file paths for the input rasters.
    labels_file_path (str): File path for the labels raster.

    Returns:
    Four numpy arrays: the training inputs, the testing inputs,
                       the training labels, and the testing labels.
    """
    # Load the input rasters
    inputs = [load_raster_data(file_path) for file_path in file_paths]

    # Stack the inputs along the depth dimension
    X = np.stack(inputs, axis=-1)

    # Load the labels
    y = load_raster_data(labels_file_path)

    # Apply a threshold
    X[X > 1000] = 0
    y[y > 1000] = 0

    # Take the logarithm of the data
    X = np.log1p(X)
    y = np.log1p(y)

    # Normalize the data
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Reshape the data to fit the model
    X = np.expand_dims(X, axis=0)
    y = np.expand_dims(y, axis=0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
