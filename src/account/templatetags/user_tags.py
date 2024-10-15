from django import template
from account.models import Account  # Update this import to match your project structure

register = template.Library()

@register.simple_tag
def get_user_count():
    return Account.objects.count()