from django.urls import path
from api import views

urlpatterns = [
    path('dataset/upload', views.upload_dataset, name='upload_dataset'),
    path('analysis/optimum_clusters_number', views.optimum_clusters_number, name='api/optimum_clusters_number'),
    path('analysis/clustering', views.clustering, name='api/clustering'),
    path('analysis/save-result', views.save_result, name='api/save_result'),
    path('analysis/donwload-clusters-zip', views.donwload_clusters_zip, name='api/donwload_clusters_zip')
]
