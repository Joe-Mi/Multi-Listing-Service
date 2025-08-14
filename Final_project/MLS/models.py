from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db.models.signals import post_save
from django.dispatch import receiver
from datetime import datetime
import random

# Create your models here.
class User(AbstractUser):
    class Role(models.TextChoices):
        ADMIN = 'admin', 'Admin' 
        BUYER = 'buyer', 'Buyer'
        SELLER = 'seller', 'Seller'


    role = models.CharField(max_length=50, choices=Role.choices)
    age = models.IntegerField(validators=[MinValueValidator(18)], default=18, blank=True)

    def save(self, *arg, **kwargs):
        if not self.pk and not self.role:
            self.role = self.base_role
        return super().save(*arg, **kwargs)
        

# Seller class
class sellerManager(BaseUserManager):
    def get_queryset(self, *args, **kwargs):
        results = super().get_queryset(*args, **kwargs)
        return results.filter(role=User.Role.SELLER)
    
class Seller(User):
    base_role = User.Role.SELLER
    seller = sellerManager()

    class Meta:
        proxy = True

@receiver(post_save, sender=Seller)
def create_S_Profile(sender, instance, created, **kwargs):
    if created and instance.role == User.Role.SELLER:
        sellerProfile.objects.create(user=instance)

class sellerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    company = models.CharField(max_length=200)


# Listing class and details
def default_zpid():
    i = random.randint(1, 255)
    return f'ls{datetime.now().strftime("%Y%m%d%H%M%S")}n{i}'

class Listing(models.Model):
    listing_name = models.CharField(max_length=200)  # title of the listing
    zpid = models.CharField(max_length=200, default=default_zpid)
    price = models.FloatField(validators=[MinValueValidator(0.0)], db_index=True) # Price of the property
    type = models.CharField(max_length=50, db_index=True)  # Property type
    space = models.FloatField(validators=[MinValueValidator(0.0)])  # living Area in square meters
    bedrooms = models.IntegerField(validators=[MinValueValidator(0)])
    bathrooms = models.IntegerField(validators=[MinValueValidator(0)])
    county = models.CharField(max_length=100)
    city = models.CharField(max_length=100, db_index=True)
    streetAddress = models.CharField(max_length=200)
    seller_id = models.ForeignKey(sellerProfile, on_delete=models.CASCADE, null=True)   # the listing broker
    status = models.CharField(max_length=200, default='available')
    views = models.IntegerField(default=0)  # Number of views
    rating = models.FloatField(default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(5.0)])  # Rating
    images = models.ImageField(upload_to='listing_images/', blank=True, null=True)  # Path for storing images

    def __str__(self):
        return self.listing_name

# Buyer class
class buyerManager(BaseUserManager):
    def get_queryset(self, *args, **kwargs):
        results = super().get_queryset(*args, **kwargs)
        return results.filter(role=User.Role.BUYER)
    
class Buyer(User):
    base_role = User.Role.BUYER
    buyer = buyerManager()

    class Meta:
        proxy = True
 
    
@receiver(post_save, sender=Buyer)
def create_B_Profile(sender, instance, created, **kwargs):
    if created and instance.role == User.Role.BUYER:
        buyerProfile.objects.create(user=instance)

class buyerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    saved_listings = models.ManyToManyField(Listing, blank=True)
    budget = models.FloatField(validators=[MinValueValidator(0.0)], null=True, default=0.0 )
    address = models.CharField(max_length=200, blank=True, default='')
    P_price = models.FloatField(validators=[MinValueValidator(0.1)], default=0.4)
    P_square_feet = models.FloatField(validators=[MinValueValidator(0.1)],  default=0.4)
    P_bedrooms = models.FloatField(validators=[MinValueValidator(0.0)], default=0.4)
    P_bathrooms = models.FloatField(validators=[MinValueValidator(0.0)], default=0.4)
    P_year = models.FloatField(validators=[MinValueValidator(0.0)], default=0.4)

class Chats(models.Model):
    users = models.ManyToManyField(User)  # Multiple participants in the chat
    last_updated = models.DateTimeField(auto_now=True)  # Automatically update on any change

    def latest_msg(self):
        return self.chat_logs.order_by('-sent').first()


class ChatLogs(models.Model):
    chat = models.ForeignKey(Chats, on_delete=models.CASCADE, related_name='chat_logs')  # Chat reference
    message = models.TextField(blank=True)  # Flexible field for storing message content
    sent = models.DateTimeField(auto_now_add=True)  # Timestamp when the message is sent
    from_user = models.ForeignKey(User, related_name='sent_messages', on_delete=models.CASCADE)
    
    class Meta:
        ordering = ['sent']

