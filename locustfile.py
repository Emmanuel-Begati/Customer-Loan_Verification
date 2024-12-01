from locust import HttpUser, between, task

class LoanPredictionUser(HttpUser):
    # The time between the execution of each task (in seconds)
    wait_time = between(1, 3)

    @task
    def predict_loan(self):
        # Define the data you want to send to the FastAPI prediction endpoint
        payload = {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": 2,
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000.0,
            "CoapplicantIncome": 2000.0,
            "LoanAmount": 150.0,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Property_Area": "Urban"
        }

        # Send a POST request to the prediction endpoint
        self.client.post("/predict/", json=payload)

    # You can add more tasks to simulate different actions here
