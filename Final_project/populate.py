import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Final_project.settings')
django.setup()

import pandas as pd
import random
from MLS.models import *
from django.db import transaction

# Load the CSV file, update to current location if neccesary:
csv_file = 'final/Final_project/data.csv'
data = pd.read_csv(csv_file)

# Drop rows missing critical fields
data_subset = data.dropna()

# Get all existing Seller objects
sellers = list(sellerProfile.objects.all())

def create_listing(row):
    return Listing(
        listing_name= row['Title'],
        zpid = row['Uniq Id'],
        price=row.get('Price', 0.0) or 0,
        type=row.get('Property Type', 'Single Family'),
        space=row.get('Sqr Ft', 0) or 0,
        bedrooms=row.get('Beds', 0) or 0,
        bathrooms=row.get('Bath', 0) or 0,
        county=row.get('State', 'Unknown'),
        city=row.get('City', 'Unknown'),
        streetAddress=row.get('Address Full', 'Unknown'),
        seller_id=random.choice(sellers),
        status='Avaliable',
        views=random.randint(23, 5035)
    )

def clearDb():
    a = Listing.objects.filter(city='Unknown')

    while a.exists():
        with transaction.atomic():
            a.delete()


# Bulk insert only the first 100 rows
listings = [create_listing(row) for index, row in data_subset.iterrows()]

with transaction.atomic():
    Listing.objects.bulk_create(listings)