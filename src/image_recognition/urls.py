from django.urls import path
from image_recognition.views import(
    upload_image_view,
    detail_image_view,
    edit_image_view,
    upload_history_view,
)

app_name = 'image_recognition'

urlpatterns = [
    path('upload/', upload_image_view, name='upload'),
    path('history/', upload_history_view, name='upload_history'),
    path('<slug>/', detail_image_view, name='detail'),
    path('<slug>/edit', edit_image_view, name='edit'),
]