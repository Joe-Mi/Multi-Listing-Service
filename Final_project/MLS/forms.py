from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import *

class loginForm(forms.Form):
    username = forms.CharField(label="Username")
    password = forms.CharField(label="Password", widget=forms.PasswordInput)

class UserForm(UserCreationForm):
    age = forms.IntegerField(label="Age", min_value=18, initial=18)

    class Meta:
        model = User
        fields = ('username', 'age')

class BuyerSignUpForm(UserForm):
    address = forms.CharField(max_length=255, required=True)
    budget = forms.FloatField(label="Budget", min_value=0.0)
    
    class Meta(UserForm.Meta):
        fields = UserForm.Meta.fields + ("address", "budget")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = User.Role.BUYER
        if commit:
            user.save()
            buyerProfile.objects.create(user=user, address=self.cleaned_data["address"])
            return user

class SellerSignUpForm(UserForm):
    company = forms.CharField(max_length=255, required=True)
    
    class Meta(UserForm.Meta):
        fields = UserForm.Meta.fields + ("company",)

    def save(self, commit=True):
        user = super().save(commit=False)
        user.role = User.Role.SELLER
        if commit:
            user.save()
            sellerProfile.objects.create(user=user, company=self.cleaned_data["company"])
            return user

class CreateNewListing(forms.ModelForm):
    class Meta:  
        model = Listing
        fields = ('listing_name', 'price', 'type', 
                  'space','bedrooms', 'bathrooms', 
                  'county', 'city', 'streetAddress')

    def __init__(self, *args, **kwargs):
        self.seller = kwargs.pop('seller', None)
        super().__init__(*args, **kwargs)

    def save(self, commit=True):
        listing = super().save(commit=False)
        if self.seller:
            seller_profile = sellerProfile.objects.get(user=self.seller)
            listing.seller_id = seller_profile
        if commit:
            listing.save()
            self.save_m2m 
        return listing

class CreateChatForm(forms.ModelForm):
    class Meta:
        model = Chats
        fields = []


class PreferenceForm(forms.ModelForm):
    P_CHOICES = [
        (0.4, 'Not Important'),
        (0.6, 'Important'),
        (0.8, 'Very Important'),
        (1.0, 'Critical')
    ]

    # Override fields to customize their form behavior
    P_price = forms.ChoiceField(choices=P_CHOICES, initial=0.4)
    P_square_feet = forms.ChoiceField(choices=P_CHOICES, initial=0.4)
    P_bedrooms = forms.ChoiceField(choices=P_CHOICES, initial=0.4)
    P_bathrooms = forms.ChoiceField(choices=P_CHOICES, initial=0.4)
    P_year = forms.ChoiceField(choices=P_CHOICES, initial=0.4)

    class Meta:
        model = buyerProfile
        fields = ['address', 'budget', 'P_price', 'P_square_feet', 'P_bedrooms', 'P_bathrooms', 'P_year']

    def clean_P_price(self):
        return float(self.cleaned_data['P_price'])

    def clean_P_square_feet(self):
        return float(self.cleaned_data['P_square_feet'])

    def clean_P_bedrooms(self):
        return float(self.cleaned_data['P_bedrooms'])

    def clean_P_bathrooms(self):
        return float(self.cleaned_data['P_bathrooms'])

    def clean_P_year(self):
        return float(self.cleaned_data['P_year'])


class updateListingForm(forms.Form):
    L_TYPE = [
        ('Single Family', 'single Family'),
        ('Multi Family', 'multi Family'),
        ('Apartment', 'apartment'),
        ('Townhouse', 'townhouse',),
        ('Condo', 'condo'),
        ('Lot', 'lot'),
        ('Manufactured', 'manufactured')
    ]

    L_STATUS = [
        ('Available', 'available'),
        ('Unavailable', 'unavailable')
    ]

    listing_name = forms.CharField(max_length=200)
    price = forms.FloatField(validators=[MinValueValidator(1.0)], initial=100.0)
    type = forms.ChoiceField(choices=L_TYPE, initial='Single Family')
    space = forms.FloatField(validators=[MinValueValidator(1.0)], initial=100.0)
    bedrooms = forms.IntegerField(validators=[MinValueValidator(1)], initial=1)
    bathrooms = forms.IntegerField(validators=[MinValueValidator(1)], initial=1)
    county = forms.CharField(max_length=100)
    city = forms.CharField(max_length=100)
    streetAddress = forms.CharField(max_length=200)
    status = forms.ChoiceField(choices=L_STATUS, initial='available')