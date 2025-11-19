from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
import os
import joblib

# --- Import your route blueprints ---
# These imports work because app.py is inside the 'backend' folder
from routes.disease import disease_bp
from routes.fertilizer import fertilizer_bp
from routes.dss import dss_bp
from routes.gis import gis_bp

# --- Configuration ---
# We set the static folder to '../frontend' so Flask can find your HTML/CSS/JS
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app) # Enable CORS to allow browser requests

# --- Startup File Checks ---
print("--- AGRI-MITRA Unified Server Startup ---")
fertilizer_model_path = os.path.join('models', 'fertilizer_model.pkl')
disease_model_path = os.path.join('models', 'disease_model.pt')

fertilizer_model_ok = os.path.exists(fertilizer_model_path)
disease_model_ok = os.path.exists(disease_model_path)

if fertilizer_model_ok:
    print(f"[OK] Found fertilizer model: {fertilizer_model_path}")
else:
    print(f"[!! MISSING !!] Fertilizer model not found at: {fertilizer_model_path}")

if disease_model_ok:
    print(f"[OK] Found disease model: {disease_model_path}")
else:
    print(f"[!! MISSING !!] Disease model not found at: {disease_model_path}")
print("------------------------------------------")

# --- Model Loading ---
fertilizer_model = None
if fertilizer_model_ok:
    try:
        print("Loading fertilizer model...")
        fertilizer_model = joblib.load(fertilizer_model_path)
        print("Fertilizer model loaded successfully.")
    except Exception as e:
        print(f"Error loading fertilizer model: {e}")

# --- Pass Model to Blueprint ---
if fertilizer_model:
    fertilizer_bp.model = fertilizer_model

# --- Register API Blueprints ---
app.register_blueprint(disease_bp, url_prefix='/api/disease')
app.register_blueprint(fertilizer_bp, url_prefix='/api/fertilizer')
app.register_blueprint(dss_bp, url_prefix='/api/dss')
app.register_blueprint(gis_bp, url_prefix='/api/gis')

# --- Frontend Routes ---

@app.route('/')
def serve_index():
    """Serves the main index.html page"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    """
    Serves any other file from the frontend folder 
    (e.g., style.css, script.js, gis_map.html)
    """
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return jsonify({'error': 'File not found'}), 404

# --- Main Execution ---
if __name__ == '__main__':
    print("🚀 AGRI-MITRA System Starting...")
    print("👉 Open your browser at: http://localhost:8000")
    app.run(port=8000, debug=True)
