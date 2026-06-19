from django.db import models
from django.contrib.auth.models import User


class Patient(models.Model):
    user       = models.OneToOneField(User, on_delete=models.CASCADE)
    age        = models.IntegerField(default=0)
    gender     = models.CharField(max_length=10, choices=[('Male','Male'),('Female','Female'),('Other','Other')])
    phone      = models.CharField(max_length=15, blank=True)
    address    = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.get_full_name()} ({self.user.username})"


class Diagnosis(models.Model):
    patient          = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='diagnoses')
    symptoms_entered = models.TextField()             # comma-separated
    predicted_disease= models.CharField(max_length=200)
    confidence       = models.FloatField(default=0.0) # percentage
    recommendation   = models.TextField(blank=True)
    created_at       = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.patient} → {self.predicted_disease} ({self.created_at.date()})"

    def symptoms_list(self):
        return [s.strip() for s in self.symptoms_entered.split(',')]