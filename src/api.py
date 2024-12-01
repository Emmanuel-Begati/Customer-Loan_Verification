from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.preprocessing import load_scaler, preprocess_data
from src.model import load_model, predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import csv

# Paths to model, scaler, and data
MODEL_PATH = "models/loan_model.pkl"
SCALER_PATH = "models/scalers_real.pkl"
NEW_DATA_PATH = "data/prediction_log.csv"

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

# Columns expected by the model
expected_columns = [
    "Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Gender_Female", "Gender_Male", "Married_No", "Married_Yes",
    "Education_Graduate", "Education_Not Graduate", "Self_Employed_No", "Self_Employed_Yes",
    "Property_Area_Rural", "Property_Area_Semiurban", "Property_Area_Urban"
]

# Function to preprocess input
def preprocess_input(features: dict) -> pd.DataFrame:
    input_data = pd.DataFrame([features])
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    missing_cols = set(expected_columns) - set(input_data_encoded.columns)
    for c in missing_cols:
        input_data_encoded[c] = 0
    return input_data_encoded[expected_columns]

# Function to log prediction data
def log_prediction(data: dict):
    os.makedirs(os.path.dirname(NEW_DATA_PATH), exist_ok=True)
    file_exists = os.path.isfile(NEW_DATA_PATH)
    with open(NEW_DATA_PATH, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

@app.get("/")
def root():
    return {"message": "Welcome to the Loan Prediction API"}

@app.post("/predict/")
async def predict_loan(features: LoanApplication):
    try:
        input_data = preprocess_input(features.dict())
        preprocessed_data = preprocess_data(input_data, scaler)
        prediction = predict(model, preprocessed_data)
        print(preprocessed_data)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        
        # Log the prediction data
        log_data = {**features.dict(), "Loan_Status": prediction[0]}
        log_prediction(log_data)

        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain/")
async def retrain_model():
    try:
        if not os.path.exists(NEW_DATA_PATH):
            raise HTTPException(status_code=400, detail="No new data found for retraining.")
        
        data = pd.read_csv(NEW_DATA_PATH)
        X = data.drop("Loan_Status", axis=1)
        y = data["Loan_Status"]
        X = pd.get_dummies(X, drop_first=True)
        
        # Handle missing columns
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_columns]
        X_scaled = scaler.transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Retrain model
        new_model = RandomForestClassifier(random_state=42)
        new_model.fit(X_train, y_train)
        
        # Save updated model
        joblib.dump(new_model, MODEL_PATH)

        # Reload updated model into memory
        global model
        model = new_model

        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))