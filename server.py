from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Home Price Prediction API!"

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    # Extract input data from the form
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bed = int(request.form['bed'])  # Replacing 'bhk' with 'bed'
    bath = int(request.form['bath'])

    # Call the prediction function from util.py
    estimated_price = util.get_estimated_price(location, total_sqft, bed, bath)

    # Prepare the response
    response = jsonify({
        'estimated_price': estimated_price
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run(host="0.0.0.0", port=10000)
