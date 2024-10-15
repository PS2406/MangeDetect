from django import forms
from image_recognition.models import UploadedImage

class UploadImageForm(forms.ModelForm):

    class Meta:
        model = UploadedImage
        fields = ['result', 'image']


class UpdateUploadImageForm(forms.ModelForm):

    class Meta:
        model = UploadedImage
        fields = ['result', 'image']

    def save(self, commit=True):
        uploaded_image = self.instance
        #uploaded_image.title = self.cleaned_data['title']
        uploaded_image.result = self.cleaned_data['result']

        # If new image is set, change it. If not, leave it.
        if self.cleaned_data['image']:
            uploaded_image.image = self.cleaned_data['image']

        if commit:
            uploaded_image.save()
        return uploaded_image
