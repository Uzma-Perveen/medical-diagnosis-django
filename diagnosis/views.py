from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse
from django.db.models import Count
from django.utils import timezone

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

from .forms import RegisterForm, SymptomForm
from .models import Patient, Diagnosis
from .ml_engine import predict_disease, get_recommendation


# ─── Auth Views ──────────────────────────────────────────────────────────────

def register_view(request):
    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, "Account created successfully!")
        return redirect('dashboard')
    return render(request, 'diagnosis/register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        user = authenticate(request,
                            username=request.POST['username'],
                            password=request.POST['password'])
        if user:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, "Invalid credentials.")
    return render(request, 'diagnosis/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


# ─── Core Views ───────────────────────────────────────────────────────────────

@login_required
def dashboard(request):
    patient   = get_object_or_404(Patient, user=request.user)
    diagnoses = patient.diagnoses.all()[:5]
    total     = patient.diagnoses.count()
    return render(request, 'diagnosis/dashboard.html', {
        'patient': patient, 'diagnoses': diagnoses, 'total': total
    })


@login_required
def symptom_input(request):
    form = SymptomForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        selected = form.cleaned_data['symptoms']
        result   = predict_disease(selected)
        disease  = result['disease']
        rec      = get_recommendation(disease)

        patient  = get_object_or_404(Patient, user=request.user)
        diagnosis = Diagnosis.objects.create(
            patient           = patient,
            symptoms_entered  = ', '.join(selected),
            predicted_disease = disease,
            confidence        = result['confidence'],
            recommendation    = rec,
        )
        return redirect('result', pk=diagnosis.pk)
    return render(request, 'diagnosis/symptom_input.html', {'form': form})


@login_required
def result_view(request, pk):
    diagnosis = get_object_or_404(Diagnosis, pk=pk, patient__user=request.user)
    return render(request, 'diagnosis/result.html', {'diagnosis': diagnosis})


@login_required
def history_view(request):
    patient   = get_object_or_404(Patient, user=request.user)
    diagnoses = patient.diagnoses.all()
    return render(request, 'diagnosis/history.html', {'diagnoses': diagnoses})


# ─── PDF Report ───────────────────────────────────────────────────────────────

@login_required
def download_report(request, pk):
    diagnosis = get_object_or_404(Diagnosis, pk=pk, patient__user=request.user)
    patient   = diagnosis.patient

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    # Title
    title_style = ParagraphStyle('title', parent=styles['Title'], textColor=colors.darkblue)
    story.append(Paragraph("🏥 Medical Diagnosis Report", title_style))
    story.append(Spacer(1, 12))

    # Patient info table
    data = [
        ['Patient Name', patient.user.get_full_name()],
        ['Username',     patient.user.username],
        ['Age',          str(patient.age)],
        ['Gender',       patient.gender],
        ['Report Date',  diagnosis.created_at.strftime('%Y-%m-%d %H:%M')],
    ]
    t = Table(data, colWidths=[150, 300])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.lightblue),
        ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',   (0,0), (-1,-1), 11),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING',    (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # Diagnosis
    story.append(Paragraph(f"<b>Predicted Disease:</b> {diagnosis.predicted_disease}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {diagnosis.confidence}%", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Symptoms Reported:</b>", styles['Normal']))
    for s in diagnosis.symptoms_list():
        story.append(Paragraph(f"• {s.replace('_',' ').title()}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Recommendation:</b> {diagnosis.recommendation}", styles['Normal']))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<i>⚠️ Disclaimer: This report is AI-generated and does not replace professional medical advice.</i>",
                            ParagraphStyle('disclaimer', parent=styles['Normal'], textColor=colors.red)))

    doc.build(story)
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="diagnosis_{pk}.pdf"'
    return response
