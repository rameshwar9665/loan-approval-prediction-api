from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("loan_approval_rf_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Convert input values into array
    input_features = np.array([list(data.values())])
    
    # Predict using the model
    prediction = model.predict(input_features)
    
    # Map prediction to message
    result = "approved" if prediction[0] == 1 else "rejected"
    
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
