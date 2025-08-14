from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from .models import *
import json


# Create your tests here.
User = get_user_model()

class AuthTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.Bsignup_url = reverse('buyer_signIn')
        self.Ssignup_url = reverse('seller_signIn')
        self.login_url = reverse('Login')
        self.home_url = reverse('home')

    def test_buyer_signup_and_login(self):
        # Step 1: Sign up
        signup_data = {
            'username': 'buyer01',
            'age': 40,
            'budget': 200000.0,
            'address': 'New york',
            'password1': 'StrongPassword123',
            'password2': 'StrongPassword123'
        }
        response = self.client.post(self.Bsignup_url, signup_data, follow=True)
        self.assertEqual(response.status_code, 200)

        # Check if user was actually created
        self.assertTrue(User.objects.filter(username='buyer01').exists())

        # Login using POST to login view
        login_data = {
            'username': 'buyer01',
            'password': 'StrongPassword123',
        }
        response = self.client.post(self.login_url, login_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'For You:')

        # Confirm session is active
        response = self.client.get(self.home_url)
        self.assertEqual(response.status_code, 200)

    def test_seller_signup_and_login(self):
        # Step 1: Sign up
        signup_data = {
            'username': 'seller01',
            'age': 20,
            'company': 'test.ltd',
            'password1': 'StrongPassword123',
            'password2': 'StrongPassword123'
        }
        response = self.client.post(self.Ssignup_url, signup_data, follow=True)
        self.assertEqual(response.status_code, 200)

        # Check if user was actually created
        self.assertTrue(User.objects.filter(username='seller01').exists())

        # Login using POST to login view
        login_data = {
            'username': 'seller01',
            'password': 'StrongPassword123',
        }
        response = self.client.post(self.login_url, login_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Your listings:')

        # Confirm user session is active
        response = self.client.get(self.home_url)
        self.assertEqual(response.status_code, 200)


class BuyerProfileTestCase(TestCase):
    def setUp(self):
        self.client = Client()

        # Create buyer user
        self.buyer_user = Buyer.objects.create_user(
            username='buyer01',
            password='StrongPassword123',
            role='buyer'
        )

        # Auto-created via signal
        self.buyer_profile = self.buyer_user.buyerprofile

        # Log in buyer for the test session
        self.client.force_login(self.buyer_user)

    def test_buyer_updates_profile_and_logs_out(self):
        # Update profile
        response = self.client.post(reverse('update'), {
            'address': 'Newer Address',
            'budget': 300000.0,
            'P_price': 0.3,
            'P_square_feet': 0.3,
            'P_bedrooms': 0.3,
            'P_bathrooms': 0.3,
            'P_year': 0.3,
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.buyer_profile.refresh_from_db()
        self.assertEqual(self.buyer_profile.address, 'Newer Address')
        self.assertEqual(self.buyer_profile.budget, 300000.0)

        # Confirm user is still authenticated
        self.assertTrue(response.wsgi_request.user.is_authenticated)

        # Logout
        response = self.client.get(reverse('Logout'), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Register')  # Assuming your login/register page contains this
        self.assertFalse(response.wsgi_request.user.is_authenticated)


class ListingTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.seller_user = Seller.objects.create(
            username='seller01',
            password='StrongPassword123',
            role='seller'
        )
        self.seller_user.set_password('StrongPassword123')
        self.seller_user.save()

        self.seller_profile = self.seller_user.sellerprofile  # already created by signal
        self.seller_profile.company = 'Real Co'
        self.seller_profile.save()


    def test_create_listing(self):
        self.client.login(username='seller01', password='StrongPassword123')  # ← Login step

        response = self.client.post(reverse('sellerListings'), {
            'listing_name': 'Test Property',
            'price': 500000.0,
            'type': 'Single Family Home',
            'space': 1500,
            'bedrooms': 3,
            'bathrooms': 2,
            'county': 'Maricopa',
            'city': 'Phoenix',
            'streetAddress': '123 Main St'
        })

        self.assertEqual(response.status_code, 302)
        self.assertEqual(Listing.objects.count(), 1)

        listing = Listing.objects.first()
        self.assertEqual(listing.listing_name, 'Test Property')
        self.assertEqual(listing.city, 'Phoenix')
        self.assertEqual(listing.seller_id, self.seller_profile)



    def test_search_and_filter_listings(self):
        # Create listing to search
        Listing.objects.create(
            listing_name="Test Property",
            price=500000.0,
            type="Single Family Home",
            space=1500,
            bedrooms=3,
            bathrooms=2,
            county="Maricopa",
            city="Phoenix",
            streetAddress="123 Main St",
            seller_id=self.seller_profile
        )

        # Step 1: Search by keyword
        response = self.client.post(reverse('search'), {'searchQuery': 'Test Property'}, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Property')

        # Step 2: Apply price filter
        response = self.client.post(reverse('search'), {
            'searchQuery': 'Phoenix',
            'min_price': 400000
        }, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Maricopa')


class chatTest(TestCase):
    def setUp(self):
        # Create buyer and seller users
        self.buyer_user = Buyer.objects.create_user(username='buyer1', password='pass123', role='buyer')
        self.seller_user = Seller.objects.create_user(username='seller1', password='pass123', role='seller')

        # Create their profiles
        self.buyer_profile = buyerProfile.objects.get(user=self.buyer_user)
        self.seller_profile = sellerProfile.objects.get(user=self.seller_user)

    def test_buyer_saves_listing_and_contacts_seller(self):
        # Create listing
        listing = Listing.objects.create(
            listing_name="Hilltop Retreat",
            city="Curepipe",
            county="Plaines Wilhems",
            price=400000,
            type="House",
            space=170,
            bedrooms=3,
            bathrooms=2,
            streetAddress="123 Hill St",
            seller_id=self.seller_profile
        )

        # Log in buyer
        self.client.login(username='buyer1', password='pass123')

        # Save the listing
        response = self.client.post(reverse('save'), {'listing_id': listing.pk}, follow=True)
        self.assertContains(response, 'Saved')

        # Contact seller
        response = self.client.post(reverse('newchat', args=[listing.pk]), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'profile.html')
        self.assertContains(response, 'chats')

        # Send message
        chat = Chats.objects \
            .filter(users=self.buyer_user) \
            .filter(users=self.seller_user) \
            .first()

        response = self.client.post(
            reverse('send'),
            data=json.dumps({'message': 'Yes, it’s available.', 'chat_id': chat.pk}),
            content_type='application/json'
        )
        json_response = json.loads(response.content)
        self.assertEqual(json_response['message'], 'Yes, it’s available.')
        self.assertEqual(json_response['chat_id'], chat.pk)

    def test_seller_responds_in_chat_and_updates_listing(self):
        # Create chat
        chat = Chats.objects.create()
        chat.users.add(self.buyer_user, self.seller_user)

        # Log in seller
        self.client.login(username='seller1', password='pass123')

        # Respond in chat
        response = self.client.post(
            reverse('send'),
            data=json.dumps({'message': 'Yes, it’s available.', 'chat_id': chat.pk}),
            content_type='application/json'
        )
        self.assertTrue(ChatLogs.objects.filter(chat=chat, from_user=self.seller_user).exists())
