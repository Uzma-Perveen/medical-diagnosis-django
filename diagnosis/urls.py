from django.urls import path
from . import views

urlpatterns = [
    path('',            views.dashboard,      name='dashboard'),
    path('register/',   views.register_view,  name='register'),
    path('login/',      views.login_view,     name='login'),
    path('logout/',     views.logout_view,    name='logout'),
    path('diagnose/',   views.symptom_input,  name='symptom_input'),
    path('result/<int:pk>/', views.result_view, name='result'),
    path('history/',    views.history_view,   name='history'),
    path('report/<int:pk>/', views.download_report, name='download_report'),
]