from flask import Flask, request, jsonify
from flask_cors import CORS
import util  # Assuming util.py is in the same directory as server.py

app = Flask(__name__)

# Enable CORS for all domains (you can restrict it to specific domains)
CORS(app)

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    """
    Endpoint to fetch available location names.
    Returns:
        JSON: A dictionary containing a list of locations.
    """
    try:
        response = jsonify({
            'locations': util.get_location_names()
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify({'error': f"Failed to fetch location names. Details: {str(e)}"}), 500


@app.route('/predict_home_price', methods=['GET'])
def predict_home_price():
    """
    Endpoint to predict home price based on input features.
    Expects:
        - total_sqft (float): Total area in square feet.
        - location (str): Location name.
        - bhk (int): Number of bedrooms (formerly bed).
        - bath (int): Number of bathrooms.
        - balcony (int): Number of balconies (optional).
    Returns:
        JSON: Estimated price or error message.
    """
    try:
        # Parse input data from the URL query parameters
        location = request.args.get('location', '').strip()
        total_sqft = float(request.args.get('total_sqft', 0))
        bed = int(request.args.get('bhk', 0))  # Use 'bhk' as bedrooms
        bath = int(request.args.get('bath', 0))
        balcony = int(request.args.get('balcony', 0))  # Optional, adjust if needed

        # Validate inputs
        if not location or total_sqft <= 0 or bed <= 0 or bath <= 0:
            return jsonify({'error': 'Invalid input parameters. Please provide valid data.'}), 400

        # Get the estimated price
        estimated_price = util.get_estimated_price(location, total_sqft, bed, bath)

        # Return the prediction
        response = jsonify({
            'estimated_price': estimated_price
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except ValueError:
        return jsonify({'error': 'Invalid input format. Please check the input values.'}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction failed. Details: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    try:
        util.load_saved_artifacts()  # Make sure you have a utility function to load the model
        app.run(host='0.0.0.0', port=5000, debug=True)  # Accessible on local network
    except Exception as e:
        print(f"Failed to start the server. Details: {str(e)}")
