from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the ML Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input features from JSON request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        # Perform prediction
        prediction = model.predict(features)
        
        # Send response
        response = {
            'prediction': int(prediction[0])
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
