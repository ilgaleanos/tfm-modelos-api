from django.urls import path

from . import views

urlpatterns = [
    path('red', views.red, name='red'),
    path('regresion', views.regresion, name='regresion'),
]