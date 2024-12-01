import joblib
import numpy as np
import pandas as pd

def load_scaler(scaler_path):
    """
    Load the saved scaler for preprocessing.
    
    Parameters:
    - scaler_path (str): Path to the scaler file.
    
    Returns:
    - scaler: Loaded scaler object.
    """
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}.")
    return scaler

def preprocess_data(data, scaler):
    """
    Preprocess the input data using the loaded scaler.
    
    Parameters:
    - data (pd.DataFrame): Input data to preprocess.
    - scaler: Loaded scaler object.
    
    Returns:
    - np.ndarray: Scaled data ready for prediction.
    """
    # Ensure data is in DataFrame format
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a Pandas DataFrame.")
    
    # Apply the scaler
    scaled_data = scaler.transform(data)
    return scaled_data
