from flask import Flask, request, jsonify
import util

app = Flask(__name__)

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

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    """
    Endpoint to predict home price based on input features.
    Expects:
        - total_sqft (float): Total area in square feet.
        - location (str): Location name.
        - bed (int): Number of bedrooms.
        - bath (int): Number of bathrooms.
    Returns:
        JSON: Estimated price or error message.
    """
    try:
        # Parse input data
        data = request.form
        total_sqft = float(data.get('total_sqft', 0))
        location = data.get('location', '').strip()
        bed = int(data.get('bed', 0))
        bath = int(data.get('bath', 0))

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
        util.load_saved_artifacts()
        app.run(host='0.0.0.0', port=5000, debug=True)  # Accessible on local network
    except Exception as e:
        print(f"Failed to start the server. Details: {str(e)}")
