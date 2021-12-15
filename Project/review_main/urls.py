# cities/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path('search/', views.url_search, name='search_results'),
    path('', views.home, name='home'),
]