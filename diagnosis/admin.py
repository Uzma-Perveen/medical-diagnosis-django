from django.contrib import admin
from .models import Patient, Diagnosis

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['user', 'age', 'gender', 'phone', 'created_at']
    search_fields = ['user__username', 'user__first_name']

@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    list_display = ['patient', 'predicted_disease', 'confidence', 'created_at']
    list_filter  = ['predicted_disease', 'created_at']
    search_fields = ['patient__user__username', 'predicted_disease']