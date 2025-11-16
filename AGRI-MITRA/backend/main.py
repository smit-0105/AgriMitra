from flask import Flask
from flask_cors import CORS
import os
import joblib

# --- Import your route blueprints ---
from routes.disease import disease_bp
from routes.fertilizer import fertilizer_bp
from routes.dss import dss_bp      # For Decision Support System
from routes.gis import gis_bp      # For GIS/Asset Mapping data

# --- Startup File Checks ---
# This will help debug file path issues before they cause errors.
print("--- AGRI-MITRA Backend Startup Check ---")

# Define model paths
fertilizer_model_path = os.path.join('models', 'fertilizer_model.pkl')
disease_model_path = os.path.join('models', 'disease_model.pt')

# Check if model files exist
fertilizer_model_ok = os.path.exists(fertilizer_model_path)
disease_model_ok = os.path.exists(disease_model_path)

if fertilizer_model_ok:
    print(f"[OK] Found fertilizer model: {fertilizer_model_path}")
else:
    print(f"[!! MISSING !!] Fertilizer model not found at: {fertilizer_model_path}")
    print("    -> Please copy 'fertilizer_model.pkl' from 'ml/' to 'backend/models/'")

if disease_model_ok:
    print(f"[OK] Found disease model: {disease_model_path}")
else:
    print(f"[!! MISSING !!] Disease model not found at: {disease_model_path}")
    print("    -> Please copy 'disease_model.pt' from 'ml/' to 'backend/models/'")

print("------------------------------------------")


# --- Model Loading (Fertilizer) ---
fertilizer_model = None # Initialize as None
if fertilizer_model_ok:
    print("Loading fertilizer model...")
    try:
        fertilizer_model = joblib.load(fertilizer_model_path)
        print("Fertilizer model loaded successfully.")
    except Exception as e:
        print(f"Error loading fertilizer model: {e}")
        fertilizer_model = None # Ensure it's None if loading fails
else:
    print("Skipping fertilizer model loading (file not found).")


# --- App Setup ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for your frontend

# --- Pass the loaded model to the blueprint ---
# This makes 'fertilizer_model' available inside the 'fertilizer_bp' routes
if fertilizer_model:
    fertilizer_bp.model = fertilizer_model

# --- Register Blueprints ---
# This tells Flask to use the route files we imported.
app.register_blueprint(disease_bp, url_prefix='/api/disease')
app.register_blueprint(fertilizer_bp, url_prefix='/api/fertilizer')
app.register_blueprint(dss_bp, url_prefix='/api/dss')
app.register_blueprint(gis_bp, url_prefix='/api/gis')

@app.route('/')
def index():
    """A simple route to check if the API is running and models are loaded."""
    return jsonify({
        "status": "AGRI-MITRA Backend API is running.",
        "models_loaded": {
            "fertilizer_model": fertilizer_model is not None,
            "disease_model": disease_model_ok # We check 'ok' since it loads in its own route
        },
        "model_paths": {
            "fertilizer_model_path": fertilizer_model_path,
            "disease_model_path": disease_model_path
        }
    })

# --- Main execution ---
# This corrected check ensures the server only runs when
# you execute this file directly (e.g., `python main.py`)
if __name__ == '__main__':
    # 'debug=True' auto-reloads the server when you save changes.
    # In production, you would use a Gunicorn server.
    app.run(port=8000, debug=True)