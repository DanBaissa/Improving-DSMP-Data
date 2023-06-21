import numpy as np
import rasterio
from matplotlib import pyplot as plt


def rescale_output_to_original(input_file, original_file):
    # Load the data
    with rasterio.open(input_file) as src:
        input_data = src.read(1)

    with rasterio.open(original_file) as src:
        original_data = src.read(1)

    # Find the min and max of the original data
    original_min = np.min(original_data)
    original_max = np.max(original_data)

    # Rescale the input data to match the original range
    rescaled_data = input_data * (original_max - original_min) + original_min

    return rescaled_data


if __name__ == "__main__":
    rescaled_data = rescale_output_to_original('Improved.tif', '2013_12_eth.tif')
    # Here you can further process or visualize the rescaled_data

    # Visualize the predicted DSMP raster
    plt.imshow(rescaled_data, cmap='gray')
    plt.title('Improved DSMP Dataset')
    plt.show()

    # Visualize the predicted DSMP raster
    plt.imshow(np.log1p(rescaled_data), cmap='gray')
    plt.title('Improved DSMP Dataset Log Scale')
    plt.show()