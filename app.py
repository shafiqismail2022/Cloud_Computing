from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import pickle

model_path = "./fraud_model.plk"

if not os.path.exists(model_path):
    print("Model file not found! Path tried:", model_path)
    exit()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Load the trained model (ensure fraud_model.pkl is in the same directory)
model = joblib.load(model_path)

@app.route('/', methods=['GET'])
def home():
    return "Credit Card Fraud Detection API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON request
    data = request.get_json(force=True)
    
    # 'features' should be a list of values the model expects (flattened row)
    try:
        features = np.array(data['features']).reshape(1, -1)
    except Exception as e:
        return jsonify({'error': 'Invalid input format. Expected "features": [..]', 'details': str(e)}), 400

    # Make prediction
    prediction = model.predict(features)[0]
    
    # Return prediction result
    return jsonify({'fraud': bool(prediction)})

if __name__ == '__main__':
    # Run on localhost for testing
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)