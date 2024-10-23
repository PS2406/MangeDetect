from django import template
from account.models import Account
from image_recognition.models import UploadedImage 

register = template.Library()

@register.simple_tag
def get_user_count():
    return Account.objects.count()

@register.simple_tag
def get_image_count():
    return UploadedImage.objects.count()