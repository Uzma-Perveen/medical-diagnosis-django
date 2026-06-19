import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_model', 'disease_model.pkl')

_cache = None

def load_model():
    global _cache
    if _cache is None:
        _cache = joblib.load(MODEL_PATH)
    return _cache['model'], _cache['symptoms']


def predict_disease(selected_symptoms: list[str]) -> dict:
    """
    selected_symptoms: list of symptom strings the user checked
    Returns: { disease, confidence, all_probabilities }
    """
    model, symptom_list = load_model()

    # Build feature vector
    vector = np.zeros(len(symptom_list))
    for s in selected_symptoms:
        if s in symptom_list:
            vector[symptom_list.index(s)] = 1

    probs   = model.predict_proba([vector])[0]
    classes = model.classes_
    top_idx = np.argsort(probs)[::-1][:5]

    return {
        'disease':    classes[top_idx[0]],
        'confidence': round(probs[top_idx[0]] * 100, 2),
        'top5': [
            {'disease': classes[i], 'confidence': round(probs[i] * 100, 2)}
            for i in top_idx
        ]
    }


RECOMMENDATIONS = {
    'Diabetes':          'Consult an Endocrinologist. Monitor blood sugar regularly.',
    'Hypertension':      'Consult a Cardiologist. Reduce salt intake and stress.',
    'Malaria':           'Visit a hospital immediately. Antimalarial medication required.',
    'Dengue':            'Seek emergency care. Stay hydrated and monitor platelet count.',
    'Typhoid':           'Consult a General Physician. Antibiotic treatment needed.',
    'Tuberculosis':      'Visit a Pulmonologist. DOTS therapy may be required.',
    'Common Cold':       'Rest, stay hydrated, and take OTC cold medicine.',
    'Pneumonia':         'Consult a Pulmonologist. Antibiotics may be required.',
    'Heart attack':      '🚨 EMERGENCY — Call 115 immediately!',
    'Hepatitis B':       'Consult a Gastroenterologist. Antiviral therapy may be needed.',
    'Fungal infection':  'Consult a Dermatologist. Antifungal cream required.',
    'Migraine':          'Consult a Neurologist. Avoid triggers (light, noise, stress).',
}

def get_recommendation(disease: str) -> str:
    return RECOMMENDATIONS.get(disease, 'Please consult a qualified physician for further evaluation.')