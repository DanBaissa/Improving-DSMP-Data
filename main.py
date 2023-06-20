# main.py

import matplotlib.pyplot as plt
from preprocess import preprocess_data, load_raster_data
from model import create_model

def main():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data("Croped_BM_ETHIOPIA_dec_2013.tif", "2013_12_eth.tif")

    # Create the model
    model = create_model()

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Evaluate the model on the test data
    loss = model.evaluate(X_test, y_test)
    print(f"Out-of-sample MSE: {loss}")

    # Use the model to make predictions on the full DSMP dataset
    BM_data = load_raster_data("BM_resampled.tif")
    BM_data = np.expand_dims(BM_data, axis=(0, 3))  # Add dimensions to fit keras Conv2D input shape
    DSMP_improved = model.predict(BM_data)[0, :, :, 0]

    # Visualize the predicted DSMP raster
    plt.imshow(DSMP_improved, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()