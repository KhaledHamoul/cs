from django.urls import path, re_path
from app import views

urlpatterns = [
    # dataset urls
    path('data/index', views.data_index, name='data_index'),
    path('data/view/<int:id>', views.data_view, name='data_view'),
    path('data/update/<int:id>', views.data_update, name='data_update'),
    path('data/delete/<int:id>', views.data_delete, name='data_delete'),
    # analysis urls
    path('analysis/optimum_clusters_number', views.optimum_clusters_number, name='optimum_clusters_number'),

    # all others static pages
    re_path(r'^.*\.html|^.', views.pages, name='pages'),
    # home page
    path('', views.index, name='home'),
]