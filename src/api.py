from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.preprocessing import load_scaler, preprocess_data
from src.model import load_model, predict

# Paths to model and scaler files
MODEL_PATH = "models/loan_model.pkl"
SCALER_PATH = "models/scalers_real.pkl"

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

# FastAPI app initialization
app = FastAPI()

# Define the Pydantic model for loan application input
class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: int
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: int
    Credit_History: int
    Property_Area: str

# List of columns expected by the model (same as your model's training columns)
expected_columns = [
    "Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", 
    "Credit_History", "Gender_Female", "Gender_Male", "Married_No", "Married_Yes", 
    "Education_Graduate", "Education_Not Graduate", "Self_Employed_No", "Self_Employed_Yes", 
    "Property_Area_Rural", "Property_Area_Semiurban", "Property_Area_Urban"
]

# Function to apply one-hot encoding and ensure the input data matches the trained model's expected columns
def preprocess_input(features: dict) -> pd.DataFrame:
    # Convert input data to DataFrame
    input_data = pd.DataFrame([features])
    
    # Perform one-hot encoding on categorical features
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    
    # Ensure that the input data has the same columns as the model training data
    missing_cols = set(expected_columns) - set(input_data_encoded.columns)
    for c in missing_cols:
        input_data_encoded[c] = 0  # Add missing columns with 0s (if they were not in the input)

    # Reorder columns to match the model's expected order
    input_data_encoded = input_data_encoded[expected_columns]

    return input_data_encoded

@app.get("/")
def root():
    return {"message": "Welcome to the Loan Prediction API"}

@app.post("/predict/")
async def predict_loan(features: LoanApplication):
    """
    Endpoint to predict loan approval.
    Expects a JSON input with feature values based on the LoanApplication model.
    """
    try:
        # Preprocess the input to match model expectations (one-hot encode categorical columns)
        input_data = preprocess_input(features.dict())
        
        # Preprocess the data using the scaler
        preprocessed_data = preprocess_data(input_data, scaler)
        
        # Make a prediction using the trained model
        prediction = predict(model, preprocessed_data)
        
        # Map prediction to loan status
        result = "Approved" if prediction[0] == 1 else "Rejected"
        
        return {"prediction": result}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))  # Raise a proper HTTP exception with a status code
