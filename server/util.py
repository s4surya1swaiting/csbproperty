import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bed, bath):
    """
    Predict the price based on the input features.

    Args:
    - location (str): Location name (e.g., '1st Phase JP Nagar').
    - sqft (float): Total area in square feet.
    - bed (int): Number of bedrooms.
    - bath (int): Number of bathrooms.

    Returns:
    - float: Predicted price rounded to 2 decimal places.
    """
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    # Create an array of zeros for the input features
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bed  # Replacing 'bhk' with 'bed'
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    """
    Load model and column data from saved artifacts.
    """
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations

    # Load data columns
    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # First 3 columns are sqft, bath, bed

    # Load the model
    global __model
    if __model is None:
        with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("Loading saved artifacts...done")


def get_location_names():
    """
    Get the list of location names.
    """
    return __locations


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))  # Other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # Other location
