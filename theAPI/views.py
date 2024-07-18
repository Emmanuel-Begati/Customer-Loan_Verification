from django.shortcuts import render
from .forms import ApprovalForm
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from .models import approval
from .serializers import approvalSerializer
import pickle
import joblib
from keras import backend as K
import pandas as pd
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ApprovalsView(viewsets.ModelViewSet):
    queryset = approval.objects.all()
    serializer_class = approvalSerializer

def ohevalue(df):
    ohe_col = joblib.load("C:/Users/begat/Desktop/Machine-Learning-and-AI/Credit Card Loan Verification/container/allcol.pkl")
    cat_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    df_processed = pd.get_dummies(df, columns=cat_columns, dummy_na=False, drop_first=True)
    df_processed = df_processed.replace({True: 1, False: 0})
    new_dict = {}
    for i in ohe_col:
        if i in df_processed.columns:
            new_dict[i] = df_processed[i].values
        else:
            new_dict[i] = np.zeros(len(df_processed))
    new_df = pd.DataFrame(new_dict)
    return new_df

def approvereject(unit):
    try:
        mdl = joblib.load("C:/Users/begat/Desktop/Machine-Learning-and-AI/Credit Card Loan Verification/container/loan_model.pkl")
        scalers = joblib.load("C:/Users/begat/Desktop/Machine-Learning-and-AI/Credit Card Loan Verification/container/scalers_real.pkl")
        
        # Debugging statements
        print(f"Type of scalers: {type(scalers)}")
        if isinstance(scalers, np.ndarray):
            print("Scalers object is a numpy array, which is incorrect.")
        else:
            print("Scalers object is correctly loaded.")

        X = scalers.transform(unit)
        y_pred = mdl.predict(X)
        y_pred = (y_pred > 0.58)
        newdf = pd.DataFrame(y_pred, columns=['Status'])
        newdf = newdf.replace({True: 'Approved', False: 'Rejected'})
        K.clear_session()
        return (newdf.values[0][0], X[0])
    except ValueError as e:
        return (str(e), None)

@csrf_exempt
def cxcontact(request):
    if request.method == 'POST':
        form = ApprovalForm(request.POST)
        if form.is_valid():
            First_Name = form.cleaned_data['First_Name']
            Last_Name = form.cleaned_data['Last_Name']
            Dependents = form.cleaned_data['Dependents']
            ApplicantIncome = form.cleaned_data['ApplicantIncome']
            CoapplicantIncome = form.cleaned_data['CoapplicantIncome']
            LoanAmount = form.cleaned_data['LoanAmount']
            Loan_Amount_Term = form.cleaned_data['Loan_Amount_Term']
            Credit_History = form.cleaned_data['Credit_History']
            Gender = form.cleaned_data['Gender']
            Married = form.cleaned_data['Married']
            Education = form.cleaned_data['Education']
            Self_Employed = form.cleaned_data['Self_Employed']
            Property_Area = form.cleaned_data['Property_Area']
            
            myDict = (request.POST).dict()
            df = pd.DataFrame(myDict, index=[0])
            processed_df = ohevalue(df)
            decision, scaled_data = approvereject(processed_df)
            print(decision, scaled_data)
            for column, value in processed_df.items():
                print(f"{column}: {value.iloc[0]}")  
            return JsonResponse({'status': 'success', 'decision': decision, 'scaled_data': scaled_data.tolist() if scaled_data is not None else None})
        
        else:
            return JsonResponse({'status': 'error', 'message': 'Form is not valid.'})
    else:
        form = ApprovalForm()
        return render(request, 'theAPI/cxform.html', {'form': form})
