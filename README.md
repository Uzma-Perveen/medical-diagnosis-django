# 🏥 AI-Powered Medical Diagnosis Assistant

An AI-powered web application built with **Django** and **Machine Learning** that allows patients to enter their symptoms and receive a preliminary disease prediction, along with a downloadable PDF diagnosis report.

> ⚠️ **Disclaimer:** This system does not replace professional medical advice. It only provides predictive assistance based on a machine learning model and should not be used for actual medical diagnosis.

---

## 📋 Problem Statement

Healthcare systems face a major challenge in handling large volumes of patient inquiries efficiently. Patients often experience symptoms but are unable to determine the severity of their condition or the appropriate medical specialty to consult. This project bridges that gap by combining web development and artificial intelligence to provide preliminary healthcare guidance.

---

## ✨ Features

- 🔐 **User Authentication** — Register, login, and logout securely
- 👤 **Patient Profiles** — Stores age, gender, and contact details
- 🩺 **Symptom Checker** — Searchable checklist of 130+ symptoms
- 🤖 **AI Disease Prediction** — Random Forest model predicts the most likely disease with a confidence score
- 💡 **Smart Recommendations** — Suggests next steps based on the predicted condition
- 📄 **PDF Report Generation** — Download a professional diagnosis report
- 🕘 **Diagnosis History** — View all past diagnoses per patient
- 🛠️ **Admin Dashboard** — Manage patients and diagnosis records via Django Admin

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django (Python) |
| Machine Learning | scikit-learn (Random Forest Classifier) |
| Database | SQLite |
| Frontend | HTML, Bootstrap 5 |
| PDF Generation | ReportLab |
| Data Handling | Pandas, NumPy, Joblib |

---

## 🧠 AI Component

The system uses a **Random Forest Classifier** trained on a symptom-disease dataset. Users select symptoms from a checklist, and the model predicts the most probable disease along with a confidence score, drawn from its top-5 internal probability ranking.

---

## 🗂️ Project Structure

```
medical_diagnosis_project/
├── manage.py
├── requirements.txt
├── ml_model/
│   ├── train_model.py
│   └── disease_model.pkl
├── medical_diagnosis/        # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── diagnosis/                # Main app
    ├── models.py
    ├── views.py
    ├── urls.py
    ├── forms.py
    ├── admin.py
    ├── ml_engine.py
    └── templates/diagnosis/
        ├── base.html
        ├── login.html
        ├── register.html
        ├── dashboard.html
        ├── symptom_input.html
        ├── result.html
        └── history.html
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Uzma-Perveen/medical-diagnosis-django.git
cd medical-diagnosis-django
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML model
```bash
python ml_model/train_model.py
```

### 5. Apply database migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Create an admin (superuser) account
```bash
python manage.py createsuperuser
```

### 7. Run the development server
```bash
python manage.py runserver
```

### 8. Open in browser
```
http://127.0.0.1:8000/
```

---

## 📸 Core Modules

1. **Authentication System** — Login, Registration, Logout
2. **Patient Management Module**
3. **Symptom Input & Processing Module**
4. **AI-Based Disease Prediction Engine**
5. **Diagnosis Report Generation (PDF Download)**
6. **Admin Dashboard**

---

## 🔮 Future Improvements

- Analytics & visualization dashboard (diagnosis trends, common symptoms)
- REST API endpoints for mobile app integration
- Deployment to a cloud platform (Render / Railway / PythonAnywhere)
- Expanded and validated medical dataset for higher prediction accuracy
- Multi-language support

---

## ⚖️ License

This project is for educational purposes. It is **not intended for real-world clinical use**.

---

## 🙋 Author

**Uzma Perveen**
GitHub: [@Uzma-Perveen](https://github.com/Uzma-Perveen)
