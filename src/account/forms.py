from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate

from account.models import Account

class RegistrationForm(UserCreationForm):
    email = forms.EmailField(max_length=60, help_text='Required. Please add a valid email address.')

    class Meta:
        model = Account
        fields = ("email", "username", "password1", "password2")
        
class AccountAuthenticationForm(forms.ModelForm):
    password = forms.CharField(label='Password', widget=forms.PasswordInput)

    class Meta:
        model = Account
        fields = ('email', 'password')

    def clean(self):
        if self.is_valid():
            email = self.cleaned_data['email']
            password = self.cleaned_data['password']
            if not authenticate(email=email, password=password):
                raise forms.ValidationError("Invalid Login.")
            
class AccountUpdateForm(forms.ModelForm):
    # Make email uneditable and not required when updating form.
    email = forms.EmailField(disabled=True, required=False)

    class Meta:
        model = Account
        fields = ('email', 'username')
    
    def clean_email(self):
        if self.is_valid():
            email = self.cleaned_data['email']

            # Check if the account exists.
            try:
                account = Account.objects.exclude(pk=self.instance.pk).get(email=email)
            except Account.DoesNotExist:
                return email
            raise forms.ValidationError('Email "%s" is already in use.' % account.email)
        
    def clean_username(self):
        if self.is_valid():
            username = self.cleaned_data['username']

            # Check if the account exists.
            try:
                account = Account.objects.exclude(pk=self.instance.pk).get(username=username)
            except Account.DoesNotExist:
                return username
            raise forms.ValidationError('Username "%s" is already in use.' % account.username)