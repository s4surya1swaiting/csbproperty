import json
import numpy as np
import pickle

# Global variables to store model and data
__locations = None
__model = None
__data_columns = None

def load_saved_artifacts():
    """
    Loads saved artifacts (model, data columns, locations) into memory.
    """
    print("Loading saved artifacts...start")

    global __model
    global __data_columns
    global __locations

    # Load model
    with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
        __model = pickle.load(f)

    # Load column names
    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']

    # Load location names and convert to lowercase
    __locations = [location.lower() for location in __data_columns[3:]]

    print("Loading saved artifacts...done")

def get_location_names():
    """
    Returns a list of available location names.
    """
    return __locations

def get_estimated_price(location, total_sqft, bed, bath):
    """
    Predicts the price of a home based on the provided features.

    Parameters:
        location (str): Location name.
        total_sqft (float): Total area in square feet.
        bed (int): Number of bedrooms.
        bath (int): Number of bathrooms.

    Returns:
        float: Estimated home price or error message if location not found.
    """
    try:
        # Convert input location to lowercase
        location = location.lower()

        # Validate input
        if location not in __locations:
            raise ValueError(f"Location '{location}' is not available.")

        # Prepare the input data for prediction
        loc_index = __locations.index(location)
        x = np.zeros(len(__data_columns))  # Initialize input array

        # Set the values for total_sqft, bed, and bath
        x[__data_columns.index('total_sqft')] = total_sqft
        x[__data_columns.index('bed')] = bed
        x[__data_columns.index('bath')] = bath

        # Set the location (one-hot encoded)
        x[loc_index + 3] = 1  # Adjust location index to match the data column order

        # Predict price using the model
        predicted_price = __model.predict([x])[0]

        if predicted_price < 0:
            return "Error: Predicted price is negative. Please check the input data."

        return round(predicted_price, 2)

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error occurred: {str(e)}"
