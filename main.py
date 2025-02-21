from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("aqi_model.pkl")  # Ensure your trained model is saved as 'aqi_model.pkl'

@app.route('/predict', methods=['POST'])
def predict_aqi():
    try:
        # Get input data from request
        data = request.get_json()

        temperature = float(data['temperature'])
        pressure = float(data['pressure'])
        humidity = float(data['humidity'])
        wind_speed = float(data['wind_speed'])
        wind_direction = float(data['wind_direction'])

        # Make prediction
        input_features = np.array([[temperature, pressure, humidity, wind_speed, wind_direction]])
        predicted_aqi = model.predict(input_features)[0]

        return jsonify({'aqi': round(predicted_aqi, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
