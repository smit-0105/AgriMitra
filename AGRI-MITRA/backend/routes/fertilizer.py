from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import os

# --- Blueprint ---
fertilizer_bp = Blueprint('fertilizer_bp', __name__)

# The 'fertilizer_model' (the pipeline) is attached to 'fertilizer_bp' in main.py
# This is called dependency injection

@fertilizer_bp.route('/predict', methods=['POST'])
def predict_fertilizer():
    # Access the model loaded in main.py
    model = getattr(fertilizer_bp, 'model', None)
    
    if model is None:
        return jsonify({'error': 'Fertilizer model not loaded'}), 500

    try:
        data = request.json

        print("--- Fertilizer API Debug ---")
        for key in ["cropType", "temperature", "humidity", "ph", "rainfall"]:
            value = data.get(key)
            print(f"{key}: {value} (type: {type(value)})")
        print("---------------------------")

        # Validate required fields
        required_fields = ["cropType", "temperature", "humidity", "ph", "rainfall"]
        missing = [f for f in required_fields if f not in data or data[f] in [None, "", []]]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        crop = data.get("cropType")
        # Validate numeric fields
        try:
            temperature = float(data.get("temperature"))
            humidity = float(data.get("humidity"))
            ph = float(data.get("ph"))
            rainfall = float(data.get("rainfall"))
        except (TypeError, ValueError):
            return jsonify({"error": "Please fill all fields with valid numbers (temperature, humidity, ph, rainfall)."}), 400

        # Prepare DataFrame as expected by the ML model
        input_data = {
            "Crop": [crop],
            "Temperature": [temperature],
            "Humidity": [humidity],
            "pH_Value": [ph],
            "Rainfall": [rainfall]
        }
        print(f"Creating DataFrame: {input_data}")
        input_df = pd.DataFrame(input_data)

        # Run prediction
        prediction = model.predict(input_df)
        n_val, p_val, k_val = prediction[0]
        return jsonify({
            'N_recommendation_kg_ha': round(n_val, 2),
            'P_recommendation_kg_ha': round(p_val, 2),
            'K_recommendation_kg_ha': round(k_val, 2)
        })

    except Exception as e:
        print(f"Error during fertilizer prediction: {e}")
        return jsonify({'error': f"Error during prediction: {e}"}), 400
