from django.db import models

# Create your models here.
class approval(models.Model):
    GENDER_CHOICES = (
        ('Male','Male'),
        ('Female', 'Female')
    )
    MARRIED_CHOICES = (
        ('Yes','Yes'),
        ('No', 'No')
    )
    GRADUATED_CHOICES = (
        ('Graduate','Graduate'),
        ('Not Graduate', 'Not Graduate')
    )
    SELFEMPLOYED_CHOICES = (
        ('Yes','Yes'),
        ('No', 'No')
    )
    AREA_CHOICES = (
        ('Urban','Urban'),
        ('Rural', 'Rural'),
    )
    
        
    First_Name = models.CharField(max_length=15)
    Last_Name = models.CharField(max_length=15)
    Dependants = models.IntegerField()
    ApplicantIncome = models.FloatField()
    CoapplicantIncome = models.FloatField()
    LoanAmount = models.FloatField()
    Loan_Amount_Term = models.IntegerField()
    Credit_History = models.FloatField(default=0)
    Gender=models.CharField(max_length=10, choices=GENDER_CHOICES)
    Married=models.CharField(max_length=10, choices=MARRIED_CHOICES)
    Education=models.CharField(max_length=15, choices=GRADUATED_CHOICES)
    Self_Employed=models.CharField(max_length=10, choices=SELFEMPLOYED_CHOICES)
    Property_Area=models.CharField(max_length=10, choices=AREA_CHOICES)
    
    def __str__(self):
        return self.firstname