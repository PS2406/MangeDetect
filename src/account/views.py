from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from account.forms import RegistrationForm, AccountAuthenticationForm, AccountUpdateForm
from image_recognition.models import UploadedImage
from django.http import JsonResponse
from account.models import Account
from django.contrib import messages
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from datetime import datetime, timedelta
from django.utils import timezone

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
            messages.success(request, 'Successfully created account!')
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
    # Check if the user is authenticated
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
    filter_param = request.GET.get('filter')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    page_number = request.GET.get('page', 1)  # Get the page number from request
    
    # Base queryset
    queryset = UploadedImage.objects.filter(author=request.user).select_related('author')
    
    # Apply date range filtering if provided
    if start_date and end_date:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            # Add one day to end_date to include the entire end date
            end_date = end_date + timedelta(days=1)
            queryset = queryset.filter(date_updated__range=(start_date, end_date))
        except ValueError:
            # Handle invalid date format
            pass

    # Apply filtering based on result parameter
    if filter_param:
        if filter_param == 'mange':
            queryset = queryset.filter(result__icontains='mange')
        elif filter_param == 'normal':
            queryset = queryset.filter(result__icontains='normal')

    # Apply sorting based on parameter
    if sort_param:
        if sort_param == 'default':
            # Default sorting
            queryset = queryset.order_by('-date_updated')
        elif sort_param == 'result':
            queryset = queryset.order_by('result')
        elif sort_param == '-result':
            queryset = queryset.order_by('-result')
        elif sort_param == 'author':
            queryset = queryset.order_by('author__username')  
        elif sort_param == '-author':
            queryset = queryset.order_by('-author__username')
        elif sort_param == 'date_uploaded':
            queryset = queryset.order_by('date_updated')
        elif sort_param == '-date_uploaded':
            queryset = queryset.order_by('-date_updated')
    else:
        # Default sorting if no sort parameter
        queryset = queryset.order_by('-date_updated')

    # Create paginator instance
    paginator = Paginator(queryset, 10)  # 10 items per page
    try:
        uploaded_images = paginator.page(page_number)
    except PageNotAnInteger:
        # If page is not an integer, deliver first page.
        uploaded_images = paginator.page(1)
    except EmptyPage:
        # If page is out of range, deliver last page of results.
        uploaded_images = paginator.page(paginator.num_pages)
    
    # Add sort parameter and uploaded images to context for template
    context.update ({
        'uploaded_images': uploaded_images,
        'sort': sort_param,  # Pass current sort to template
        'filter': filter_param,  # Pass current filter to template
        'current_sort': sort_param if sort_param else 'default',  # For highlighting current sort option
        'current_filter': filter_param if filter_param else 'all',  # For highlighting current filter option
        'start_date': start_date if start_date else None,
        'end_date': end_date if end_date else None,
        'today': timezone.now().date(),  # For max date in date inputs
    })
    return render(request, 'account/account.html', context)

def must_authenticate_view(request):
    return render(request, 'account/must_authenticate.html', {})
