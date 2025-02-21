from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("random_forest_noise_model.pkl")  # Ensure your trained model is saved as 'random_forest_noise_model.pkl'
scaler = joblib.load("scaler.pkl")  # Load the scaler used for feature scaling

@app.route('/predict', methods=['POST'])
def predict_noise():
    try:
        # Get input data from request
        data = request.get_json()

        temperature = float(data['temperature'])
        pressure = float(data['pressure'])
        humidity = float(data['humidity'])
        wind_speed = float(data['wind_speed'])
        traffic_density = float(data['traffic_density'])

        # Prepare input features for prediction
        input_features = np.array([[temperature, pressure, humidity, wind_speed, traffic_density]])

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(input_features)

        # Make prediction using the trained model
        predicted_noise_level = model.predict(scaled_features)[0]

        # Return the predicted noise level as a response
        return jsonify({'predicted_noise_level': round(predicted_noise_level, 2)})
    
    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Running the Flask app on all available IP addresses at port 8080
    app.run(host='0.0.0.0', port=7000, debug=True)
