import joblib

def load_model(model_path):
    """
    Load the trained model from a file.
    
    Parameters:
    - model_path (str): Path to the saved model file.
    
    Returns:
    - model: Loaded model.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}.")
    return model

def predict(model, preprocessed_data):
    """
    Make predictions using the loaded model.
    
    Parameters:
    - model: Loaded model.
    - preprocessed_data (np.ndarray): Preprocessed input data.
    
    Returns:
    - np.ndarray: Model predictions.
    """
    predictions = model.predict(preprocessed_data)
    return predictions
