---

# Loan Eligibility Prediction

A machine learning project that predicts the likelihood of loan approval based on user-provided details such as income, credit history, and loan amount. This project includes model training, testing, and deployment via a user-friendly web application.

---

## 📽 Video Demo  
[Watch the Demo](https://youtu.be/KQSfEF7-7LU)  

---

## 🌐 Live Demo  
[API endpoint](https://customer-loan-verification.onrender.com/docs)  

adding the other links soon...stay tuned...

---

## 📖 Project Description  

This project is designed to help users predict loan eligibility based on key financial and personal metrics.  
The application:
1. Accepts user inputs via a web form.
2. Uses a trained machine learning model to predict loan eligibility.
3. Displays results dynamically, with a friendly message indicating the user's likelihood of loan approval.  

The backend is implemented using FastAPI, while the frontend is an HTML form styled with CSS. The prediction model is trained on a dataset of loan applications using Scikit-learn.

---

## 🚀 Features  
- Machine Learning-based predictions.  
- Friendly and dynamic user interface.  
- Deployment-ready package with Docker support.  
- Load testing and stress testing to ensure robustness.  

---

## 🛠️ Setup Instructions  

### Prerequisites  
- Python 3.9 or higher  
- FastAPI  
- Scikit-learn  
- Docker (optional for containerized deployment)

### Steps to Set Up Locally  
1. **Clone the repository:**  
   ```bash
   https://github.com/Emmanuel-Begati/Customer-Loan_Verification/
   ```

2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the backend server:**  
   ```bash
   uvicorn main:app --reload
   ```

4. **Open the web application:**  
   Open `index.html` in your browser or integrate with the backend using the provided FastAPI endpoint.

---

## 📊 Flood Request Simulation Results  

The model was tested for robustness under load using simulation tools:  
- **Maximum Requests:** 5000  
- **Success Rate:** 98%  
- **Average Response Time:** ~150ms  

Results show the application is capable of handling high traffic efficiently.

---

## 📁 Notebook  

All preprocessing, model training and evaluation steps are included in the Jupyter Notebook provided in the repository.  

### Key Sections in the Notebook  
1. **Preprocessing Functions:** Functions for data cleaning and feature engineering.  
2. **Model Training:** Logistic regression and performance evaluation.  
3. **Model Testing and Prediction:** Functions to predict new data.

---

## 🧠 Model File  

- **Trained Model:** Available as a `.pkl` file in the `models/` directory.  

To use the model:  
```python
import pickle
with open('models/loan_model.pkl', 'rb') as file:
    model = pickle.load(file)
```

---

## 📦 Deployment Package  

### Option 1: Public URL + Docker Image  
The project can be deployed as a containerized application using Docker. Although I deployed the API using render 

**Build and run the Docker image:**  
```bash
docker build -t loan-prediction-app .
docker run -p 8000:8000 loan-prediction-app
```

### Option 2: Mobile/Desktop App  
This project can be extended to a mobile or desktop app using tools like Flutter or Electron.

---

## 🤝 Contributing  

Contributions are welcome! Please fork the repository and submit a pull request for review.

---

## 📄 License  

This project is licensed under the MIT License.

--- 

### 📧 Contact  

For inquiries or issues, contact:  
[Your Email](mailto:begati16@gmail.com)

---
