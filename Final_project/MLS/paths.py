from django.urls import path
from . import views

urlpatterns = [
    path('login', views.Login, name='Login'),
    path('logout', views.Logout, name='Logout'),
    path('signIn/buyer', views.buyer_signIn, name='buyer_signIn'),
    path('signIn/seller', views.seller_signIn, name='seller_signIn'),
    path('home', views.home, name='home'),
    path('search', views.search, name='search'),
    path('profile', views.profile, name='profile'),
    path('profile/<int:id>', views.profile, name='chat'),
    path('listing/<int:listing_id>', views.listing, name='Listing'),
    path('update/<int:listing_id>', views.updateListing, name='updateListing'),
    path('user/<str:name>/saved', views.buyerListings, name='buyerListings'),
    path('user/update', views.updatePrefrences, name='update'),
    path('user/listings', views.sellerListings, name='sellerListings'),
    path('save', views.save, name='save'),
    path('remove', views.remove, name='remove'),
    path('unsave', views.unsave, name='unsave'),
    path('Send', views.send, name='send'),
    path('user/chat/<int:id>', views.newchat, name='newchat'),
]
