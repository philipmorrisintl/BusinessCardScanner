from django.urls import path

from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('api/scan-card', views.api.scan_card, name='scan_card'),
]
