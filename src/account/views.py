from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from account.forms import RegistrationForm, AccountAuthenticationForm, AccountUpdateForm
from image_recognition.models import UploadedImage
from django.http import JsonResponse
from account.models import Account
from django.contrib import messages

# Create your views here.

def registration_view(request):
    context = {}

    if request.POST:
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            email = form.cleaned_data.get('email')
            raw_password = form.cleaned_data.get('password1')
            account = authenticate(email = email, password = raw_password)
            login(request, account)
            return redirect('home')
        else:
            context['registration_form'] = form
    else: # GET request
        form = RegistrationForm()
        context['registration_form'] = form
    return render(request, 'account/register.html', context)

def logout_view(request):
    logout(request)
    messages.success(request, 'Successfully logged out!')
    return redirect('home')

def login_view(request):
    context = {}

    user = request.user
    if user.is_authenticated:
        return redirect("home")
    
    if request.POST:
        form = AccountAuthenticationForm(request.POST)
        if form.is_valid():
            email = request.POST['email']
            password = request.POST['password']
            user = authenticate(email=email, password=password)

            if user:
                login(request, user)
                messages.success(request, 'Successfully logged in!')
                return redirect("home")
    else:
        form = AccountAuthenticationForm()
    
    context['login_form'] = form
    return render(request, 'account/login.html', context)

def account_view(request):
    if not request.user.is_authenticated:
        return redirect("login")
    
    context = {}

    # Handle POST request for account updates
    if request.POST:
        form = AccountUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            user = form.save(commit=False)
            user.username = form.cleaned_data.get('username')
            user.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('account')
    else:
        form = AccountUpdateForm(instance=request.user)
        
    context['account_form'] = form

    # Get sort parameter from URL
    sort_param = request.GET.get('sort', '')
    
    # Base queryset
    uploaded_images = UploadedImage.objects.filter(author=request.user).select_related('author')
    
    # Apply sorting based on parameter
    if sort_param:
        if sort_param == 'default':
            # Default sorting (you can specify your default order)
            uploaded_images = uploaded_images.order_by('-date_uploaded')
        else:
            # Map sort parameters to model fields
            sort_mapping = {
                'result': 'result',
                '-result': '-result',
                'date_uploaded': 'date_uploaded',
                '-date_uploaded': '-date_uploaded',
            }
            
            if sort_param in sort_mapping:
                uploaded_images = uploaded_images.order_by(sort_mapping[sort_param])
    else:
        # Default sorting if no parameter is provided
        uploaded_images = uploaded_images.order_by('-date_uploaded')
    
    # Add sort parameter to context for template
    context['sort'] = sort_param
    context['uploaded_images'] = uploaded_images

    return render(request, 'account/account.html', context)

def must_authenticate_view(request):
    return render(request, 'account/must_authenticate.html', {})

def user_count(request):
    count = Account.objects.count()
    return JsonResponse({'count': count})
        