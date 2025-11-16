import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os

# --- Configuration ---
DATA_FILE = 'data/raw/Fertilizer_Recommendation.csv'
MODEL_SAVE_PATH = 'fertilizer_model.pkl'

def train_model():
    print("--- Starting Fertilizer Regression Model Training ---")

    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please make sure the file exists.")
        return

    # --- Preprocessing ---

    # Correct feature columns based on your dataset
    X = df[['Crop', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]
    
    # Correct target columns
    y = df[['Nitrogen', 'Phosphorus', 'Potassium']]

    # Define which columns are categorical and which are numeric
    categorical_features = ['Crop']
    numeric_features = ['Temperature', 'Humidity', 'pH_Value', 'Rainfall']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])

    # Model Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
    ])

    print("Training model pipeline on 100% of data...")
    pipeline.fit(X, y)

    print("Model training complete.")

    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("IMPORTANT: Copy fertilizer_model.pkl to backend/models/")

if __name__ == "__main__":
    train_model()
