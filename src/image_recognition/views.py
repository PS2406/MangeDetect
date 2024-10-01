from operator import attrgetter
from django.shortcuts import redirect, render, get_object_or_404
from image_recognition.models import UploadedImage
from image_recognition.forms import UploadImageForm, UpdateUploadImageForm
from account.models import Account

# Create your views here.

def upload_image_view(request):
    context = {}

    user = request.user
    if not user.is_authenticated:
        return redirect('must_authenticate')
    
    form = UploadImageForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        obj = form.save(commit=False)
        author = Account.objects.filter(email=user.email).first()
        obj.author = author
        obj.save()
        form = UploadImageForm()

    context['form'] = form

    return render(request, "image_recognition/upload.html", context)

def detail_image_view(request, slug):
    context = {}

    uploaded_image = get_object_or_404(UploadedImage, slug=slug)
    context['uploaded_image'] = uploaded_image

    return render(request, 'image_recognition/detail_image.html', context)

def edit_image_view(request, slug):
    context = {}

    user = request.user
    if not user.is_authenticated:
        return redirect("must_authenticate")
    
    uploaded_image = get_object_or_404(UploadedImage, slug=slug)
    if request.POST:
        form = UpdateUploadImageForm(request.POST or None, request.FILES or None, instance=uploaded_image)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.save()
            context['success_message'] = "Updated"
            uploaded_image = obj
            
    form = UpdateUploadImageForm(
        initial = {
            "title": uploaded_image.title,
            "result": uploaded_image.result,
            "image": uploaded_image.image,
        }
    )

    context['form'] = form
    return render(request, 'image_recognition/edit_image.html', context)

def upload_history_view(request):
    uploaded_images = UploadedImage.objects.all().order_by('-date_updated')[:30] # Limits recent image posts to 30.
    
    # Pass the images to the template
    return render(request, 'image_recognition/upload_history.html', {'uploaded_image': uploaded_images})