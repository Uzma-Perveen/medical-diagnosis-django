from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Patient
from .ml_engine import load_model


class RegisterForm(UserCreationForm):
    email      = forms.EmailField(required=True)
    first_name = forms.CharField(max_length=50)
    last_name  = forms.CharField(max_length=50)
    age        = forms.IntegerField(min_value=1, max_value=120)
    gender     = forms.ChoiceField(choices=[('Male','Male'),('Female','Female'),('Other','Other')])
    phone      = forms.CharField(max_length=15, required=False)

    class Meta:
        model  = User
        fields = ['username','first_name','last_name','email','password1','password2']

    
    def __init__(self, *args, **kwargs):          # ← ADDED (makes fields look nice with Bootstrap)
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'


    def save(self, commit=True):
        user = super().save(commit=False)
        user.email      = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name  = self.cleaned_data['last_name']
        if commit:
            user.save()
            Patient.objects.create(
                user   = user,
                age    = self.cleaned_data['age'],
                gender = self.cleaned_data['gender'],
                phone  = self.cleaned_data.get('phone',''),
            )
        return user


def get_symptom_choices():
    try:
        _, symptoms = load_model()
        return [(s, s.replace('_', ' ').title()) for s in symptoms]
    except Exception:
        return []


class SymptomForm(forms.Form):
    symptoms = forms.MultipleChoiceField(
        choices   = get_symptom_choices,
        widget    = forms.CheckboxSelectMultiple,
        required  = True,
        label     = "Select your symptoms"
    )