import numpy as np
import matplotlib.pyplot as plt
import rasterio

# Open the raster dataset using rasterio
dataset = rasterio.open('Croped_BM_ETHIOPIA_dec_2013.tif')

# Read the data as a numpy array
data = dataset.read(1)  # Assuming the raster has a single band

# Apply a threshold
data[data > 1000] = 0

# Plot the raster data
plt.figure(figsize=(8, 8))
plt.imshow(data, cmap='gray')
plt.colorbar(label='Original')
plt.title('Original Raster Dataset')
plt.show()

# Take the logarithm of the data
log_data = np.log1p(data)  # Apply logarithm to avoid taking log of zero or negative values

# Plot the logarithm of the raster data
plt.figure(figsize=(8, 8))
plt.imshow(log_data, cmap='gray')
plt.colorbar(label='Logarithm')
plt.title('Logarithm of Raster Dataset')
plt.show()

# Normalize the data
normalized_data = (log_data - np.min(log_data)) / (np.max(log_data) - np.min(log_data))

# Plot the normalized raster data
plt.figure(figsize=(8, 8))
plt.imshow(normalized_data, cmap='gray')
plt.colorbar(label='Normalized')
plt.title('Normalized Raster Dataset')
plt.show()

