from django.db import models
from django.db.models.signals import pre_save
from django.utils.text import slugify
from django.conf import settings
from django.db.models.signals import post_delete
from django.dispatch import receiver
import uuid

# Create your models here.

def upload_location(instance, filename, **kwargs):
    file_path = 'image_recognition/{author_id}/{image_id}-{filename}'.format(
        author_id = str(instance.author.id), 
        image_id=str(instance.image_id), 
        filename=filename
    )
    return file_path

class UploadedImage(models.Model):
    title                           = models.CharField(max_length= 50, null=False, blank=False)
    image                           = models.ImageField(upload_to=upload_location)
    image_id                        = models.UUIDField(default=uuid.uuid4, editable=True, unique=False)
    result                          = models.TextField(null=True, blank=True)
    date_uploaded                   = models.DateTimeField(auto_now_add=True, verbose_name="date uploaded")
    date_updated                    = models.DateTimeField(auto_now=True, verbose_name="date updated")
    author                          = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    slug                            = models.SlugField(blank=True, unique=True)

    def __str__(self):
        return self.title

# If uploaded image is deleted, delete image associated with it (when set to False).
@receiver(post_delete, sender=UploadedImage)
def submission_delete(sender, instance, **kwargs):
    instance.image.delete(False)

def pre_save_uploaded_image_receiver(sender, instance, *args, **kwargs):
    if not instance.slug:
        instance.slug = slugify(f"{instance.author.username}-{instance.image_id}")

pre_save.connect(pre_save_uploaded_image_receiver, sender=UploadedImage)