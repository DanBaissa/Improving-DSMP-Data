# main.py

from preprocess import preprocess_data
from model import create_model

def main():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data("Croped_BM_ETHIOPIA_dec_2013.tif", "2013_12_eth.tif")

    # Create the model
    model = create_model()

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Evaluate the model
    model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
