from fastapi import FastAPI, UploadFile, Form
import pandas as pd
from src.preprocessing import load_scaler, preprocess_data
from src.model import load_model, predict

# Paths to model and scaler files
MODEL_PATH = "models/loan_model.pkl"
SCALER_PATH = "models/scalers_real.pkl"

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Loan Prediction API"}

@app.post("/predict/")
async def predict_loan(features: dict):
    """
    Endpoint to predict loan approval.
    Expects a JSON input with feature values.
    """
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features])
        
        # Preprocess data
        preprocessed_data = preprocess_data(input_data, scaler)
        
        # Make prediction
        prediction = predict(model, preprocessed_data)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
