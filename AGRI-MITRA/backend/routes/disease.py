from flask import Blueprint, request, jsonify
import torch
import os

# --- THIS IS THE FIX ---
# It now imports from your *local* models folder, not the 'ml' folder.
from models.cnn_definition import SimpleCNN
# --- END FIX ---

from utils.image_preprocess import preprocess_image
from utils.gradcam import generate_gradcam

# --- Config ---
# IMPORTANT: This must match the NUM_CLASSES in your training scripts
NUM_CLASSES = 10 
MODEL_PATH = os.path.join('models', 'disease_model.pt')

# TODO: You MUST update this list to match the 10 classes your
# model was trained on, in the exact same order.
CLASS_NAMES = [
    'Class 0 (e.g., Apple Scab)', 'Class 1 (e.g., Apple Healthy)', 
    'Class 2 (e.g., Corn Leaf Spot)', 'Class 3 (e.g., Corn Healthy)', 
    'Class 4 (e.g., Grape Black Rot)', 'Class 5 (e.g., Grape Healthy)', 
    'Class 6 (e.g., Tomato Blight)', 'Class 7 (e.g., Tomato Healthy)', 
    'Class 8 (e.g., Potato Blight)', 'Class 9 (e.g., Potato Healthy)'
] 


# --- Model Loading ---
model = None
try:
    if os.path.exists(MODEL_PATH):
        # Use the imported SimpleCNN class
        model = SimpleCNN(num_classes=NUM_CLASSES)
        # Load the saved weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        print("Disease model loaded successfully.")
    else:
        # This warning will be printed by main.py, but good to have a backup
        print(f"Warning: {MODEL_PATH} not found. Disease API will not work.")
except Exception as e:
    print(f"Error loading disease model: {e}")
    model = None

# --- Blueprint ---
disease_bp = Blueprint('disease_bp', __name__)

@disease_bp.route('/predict', methods=['POST'])
def predict_disease():
    if model is None:
        return jsonify({'error': 'Disease model is not loaded. Check server logs.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 1. Read and preprocess the image
        image_bytes = file.read()
        image_tensor = preprocess_image(image_bytes)
        
        # 2. Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        # 3. Get class name and confidence
        class_index = predicted_idx.item()
        if class_index >= len(CLASS_NAMES):
             return jsonify({'error': f'Model predicted an invalid class index: {class_index}'}), 500
        
        disease_name = CLASS_NAMES[class_index]
        confidence_score = confidence.item() * 100
        
        # 4. (Optional) Generate Grad-CAM
        # heatmap = generate_gradcam(model, image_tensor, model.conv2)
        
        return jsonify({
            'disease_detected': disease_name,
            'confidence': f"{confidence_score:.2f}%",
            'remedy': f"Mock remedy for {disease_name}...",
            # 'gradcam_heatmap': heatmap
        })

    except Exception as e:
        print(f"Error during disease prediction: {e}")
        return jsonify({'error': str(e)}), 500