from django.urls import path, re_path
from app import views

urlpatterns = [
    path('data/index', views.data_index, name='data_index'),
    re_path(r'^.*\.html|^.', views.pages, name='pages'),
    path('', views.index, name='home'),
]
