import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS so the React app on 5173 can talk to 5000
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "message": "Sentinel API is running. Use /predict or /metrics endpoints."
    }), 200

# ============================================================================
# LOAD MODEL & PREPROCESSORS
# ============================================================================
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")
encoders_path = os.path.join(base_dir, "encoder.pkl")
metrics_path = os.path.join(base_dir, "metrics.pkl")

print("Loading ML Assets...")
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoders_path, "rb") as f:
        encoder = pickle.load(f)
    print("Assets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Required file not found: {e}")
    model = scaler = encoder = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler or not encoder:
        return jsonify({"error": "ML assets not loaded properly"}), 500
        
    data = request.json
    
    # Extract React inputs
    amount = data.get('amount', 100)
    time_val = data.get('time', 12)
    device = data.get('device', 'known')
    location_code = data.get('location', 'US')
    txn_type = data.get('type', 'online_purchase')

    # 1. Map Device
    device_map = {
        'android_phone': 'Mobile', 'ios_phone': 'Mobile',
        'windows_pc': 'Desktop', 'mac_desktop': 'Desktop', 'linux_pc': 'Desktop',
        'tablet': 'Tablet'
    }
    model_device = device_map.get(device, 'Unknown Device')
    
    # 2. Map Location
    model_location = 'India' if location_code == 'IN' else 'New York' 
    if location_code == 'GB': model_location = 'Boston'
    if location_code == 'CA': model_location = 'Seattle'
    
    # 3. Map Type
    type_map = {
        'atm_withdrawal': 'ATM Withdrawal',
        'bank_transfer': 'Bank Transfer',
        'bill_payment': 'Bill Payment',
        'online_purchase': 'Online Purchase',
        'pos_payment': 'POS Payment'
    }
    model_type = type_map.get(txn_type, 'Online Purchase')
    
    # 4. Prepare exact dict app.py uses
    features_dict = {
        'Transaction_Amount': float(amount),
        'Transaction_Type': model_type,
        'Time_of_Transaction': float(time_val),
        'Device_Used': model_device,
        'Location': model_location,
        'Previous_Fraudulent_Transactions': 0, # Assume 0 for demo if not passed
        'Account_Age': 365, # Standard
        'Number_of_Transactions_Last_24H': 2, # Standard
        'Payment_Method': 'Credit Card' # Fallback
    }
    
    # 5. ML Preprocessing
    features_df = pd.DataFrame([features_dict])

    try:
        # Categorical
        categorical_cols = ['Transaction_Type', 'Device_Used', 'Location', 'Payment_Method']
        cat_encoded = pd.DataFrame(
            encoder.transform(features_df[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        
        # Numerical
        numerical_cols = ['Transaction_Amount', 'Time_of_Transaction', 'Previous_Fraudulent_Transactions', 
                        'Account_Age', 'Number_of_Transactions_Last_24H']
        num_scaled = pd.DataFrame(scaler.transform(features_df[numerical_cols]), columns=numerical_cols)
        
        # Combine
        final_features = pd.concat([num_scaled, cat_encoded], axis=1)

        # 6. Predict
        prediction = int(model.predict(final_features)[0])
        probability = float(model.predict_proba(final_features)[0][1])
        
        # 7. Generate Explanations
        explanations = []
        if float(amount) > 2000:
            explanations.append("Extremely high transaction amount.")
        if float(time_val) > 0 and float(time_val) < 5:
            explanations.append("High-risk time window (late night).")
        if device in ['vpn', 'bot', 'new']:
            explanations.append("Untrusted or anonymizing device detected.")
        if probability > 0.5 and len(explanations) == 0:
            explanations.append("Unusual transaction pattern detected.")
            
        if len(explanations) == 0:
            explanations.append("Transaction characteristics appear normal.")

        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "explanation": explanations
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as f:
                model_metrics = pickle.load(f)
            return jsonify(model_metrics)
        else:
            return jsonify({
                "Accuracy": "N/A", "Precision": "N/A", "Recall": "N/A", 
                "F1 Score": "N/A", "ROC AUC": "N/A", "Training Size": "N/A"
            }), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the API server
    print("Starting Fraud API Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
