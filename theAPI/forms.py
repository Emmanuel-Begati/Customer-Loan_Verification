from django import forms 

class ApprovalForm(forms.Form):
	First_Name = forms.CharField(max_length=15)
	Last_Name = forms.CharField(max_length=15)
	Dependents = forms.IntegerField()
	ApplicantIncome = forms.IntegerField()
	CoapplicantIncome = forms.IntegerField()
	LoanAmount = forms.IntegerField()
	Loan_Amount_Term = forms.IntegerField()
	Credit_History = forms.IntegerField()
	Gender = forms.ChoiceField(choices=[('Male', 'Male'), ('Female', 'Female')])
	Married = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
	Education = forms.ChoiceField(choices=[('Graduate', 'Graduate'), ('Not Graduate', 'Not Graduate')])
	Self_Employed = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
	Property_Area = forms.ChoiceField(choices=[('Urban', 'Urban'), ('Rural', 'Rural')])
	