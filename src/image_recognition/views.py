from operator import attrgetter
from django.shortcuts import redirect, render, get_object_or_404
from django.http import JsonResponse
from image_recognition.models import UploadedImage
from image_recognition.forms import UploadImageForm, UpdateUploadImageForm
from account.models import Account
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torchvision.models import ResNet18_Weights
from django.core.exceptions import PermissionDenied
from django.core.cache import cache
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods
from django.views import View

def rate_limit(key_prefix, limit, period):
    def decorator(fn):
        def wrapper(request, *args, **kwargs):
            key = f"{key_prefix}:{request.user.id if request.user.is_authenticated else request.META['REMOTE_ADDR']}"
            count = cache.get(key, 0)
            
            if count >= limit:
                raise PermissionDenied("Rate limit exceeded. Please try again later.")
            
            cache.set(key, count + 1, period)
            return fn(request, *args, **kwargs)
        return wrapper
    return decorator

# Define a simple class mapping
class_mapping = {
    0: "Normal",
    1: "Mange"
}

# Load the ResNet model (do this at startup)
def load_model():
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify the final fully connected layer for your binary classification
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_mapping))  # 2 classes: Normal, Mange
    
    # Load your custom weights
    try:
        state_dict = torch.load('wombat_mange_classification_model_v1.4.pth', map_location=torch.device('cpu'))
        
        # Remove the fc layer weights to avoid size mismatch
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc")}
        
        # Load remaining weights into the model
        model.load_state_dict(state_dict, strict=False)
        print("Custom model weights loaded successfully (fc layer excluded).")
    except FileNotFoundError:
        print("Custom model weights not found. Using pre-trained weights.")
    except RuntimeError as e:
        print(f"Error loading custom weights: {e}")
        print("Using pre-trained weights instead.")
    
    # Freeze all layers except the final fully connected layer
    for name, param in model.named_parameters():
        if "fc" in name:  # Unfreeze the final classification layer
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    model.eval()  # Set the model to evaluation mode
    return model



# Global variable to store the model
image_model = load_model()

def predict_image(image):
    if image_model is None:
        return "Model not loaded", 0.0
    
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference and make prediction
    with torch.no_grad():
        output = image_model(input_batch)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top prediction and its probability
    top_prob, top_class = torch.max(probabilities, 0)
    predicted_class = class_mapping[top_class.item()]
    probability = top_prob.item()
    
    return predicted_class, probability

# Function to train the model.
   

@rate_limit('image_upload', 100, 3600)  # Rate limited to: 100 uploads per hour
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
        
        # Open the uploaded image
        image = Image.open(request.FILES['image'])
        
        # Get prediction and probability
        prediction, probability = predict_image(image)
        
        # Save prediction result
        obj.result = f"Prediction: {prediction} (Confidence: {probability:.2f})"
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
            
            # If a new image is uploaded, update the prediction
            if 'image' in request.FILES:
                image = Image.open(request.FILES['image'])
                predicted_class = predict_image(image)
                obj.result = f"Predicted class: {predicted_class}"
            
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