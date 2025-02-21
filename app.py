from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model, scaler, and label encoder
model = joblib.load("random_forest_weather_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure the request contains JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400

        # Get the input data from the request
        input_data = request.json.get("features")
        if not input_data:
            return jsonify({"error": "Missing 'features' key in request JSON"}), 400

        # Convert input data to a NumPy array
        input_array = np.array([input_data])

        # Scale the input features
        scaled_features = scaler.transform(input_array)

        # Make a prediction
        prediction = model.predict(scaled_features)

        # Decode the prediction
        decoded_prediction = label_encoder.inverse_transform(prediction)[0]

        # Return the prediction as JSON
        return jsonify({"prediction": decoded_prediction}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    # Allow Flask to listen on all interfaces for testing with devices on the same network
    app.run(host="0.0.0.0", port=5000, debug=True)
