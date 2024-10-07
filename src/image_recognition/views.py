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
import io
from torch.optim.lr_scheduler import StepLR

# Define a simple class mapping
class_mapping = {
    0: "Normal",
    1: "Mange"
}

# Load the ResNet model (do this at startup)
def load_model():
    # Use weights parameter instead of pretrained
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),  # Add dropout layer as in training
        torch.nn.Linear(num_ftrs, len(class_mapping))  # 2 classes: Normal and Mange
    )
    
    # Load your custom weights if you have them
    # If you're using a custom model, you might need to adjust this part
    try:
        state_dict = torch.load('wombat_mange_model_v4.pth', weights_only=True)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("Custom model weights not found. Using pre-trained weights.")
    except RuntimeError as e:
        print(f"Error loading custom weights: {e}")
        print("Using pre-trained weights instead.")
    
    model.eval()
    return model

# Global variable to store the model
image_model = load_model()

def predict_image(image):
    if image_model is None:
        return "Model not loaded", 0.0
    # Preprocess the image
    preprocess = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(),
       transforms.RandomVerticalFlip(),
       transforms.RandomRotation(20),
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
       transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
       transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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