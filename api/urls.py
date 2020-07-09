from django.urls import path
from api import views

urlpatterns = [
    path('dataset/upload', views.upload_dataset, name='upload_dataset'),
    path('analysis/optimum_clusters_number', views.optimum_clusters_number, name='optimum_clusters_number')
]
