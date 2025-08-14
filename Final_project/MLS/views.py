from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.db.models import Q
from django.utils.timezone import now

from .models import *
from .forms import *

import os
import pandas as pd
import numpy as np
import pickle
import faiss
import json

from Final_project.settings import BASE_DIR

#The recommedation functions: 
def get_by_preferences(user):
    """Filter listings based on user preferences."""
    w_price = user.P_price
    w_bedrooms = user.P_bedrooms
    w_bathrooms = user.P_bathrooms
    w_space = user.P_square_feet
    data = [w_price, w_bedrooms, w_bathrooms, w_space]
    return data

def recommend(id_, quantity_similar_items=5, user_weights=None, alpha=0.7):
    """Return personalized recommendations using FAISS and optional user preferences."""
    index_path = os.path.join(BASE_DIR, "faiss_index_2.index")
    df_path = os.path.join(BASE_DIR, "data.csv")
    emb_path = os.path.join(BASE_DIR, "all_embeddings_2.pkl")

    # Load data, embeddings, and FAISS index
    df_raw = pd.read_csv(df_path)
    df_raw.reset_index(drop=True, inplace=True)

    with open(emb_path, "rb") as f:
        all_embeddings = pickle.load(f)

    index = faiss.read_index(index_path)
    
    # Get index of the query item
    try:
        idx = df_raw.loc[df_raw['Uniq Id'] == id_].index[0]
        query_embedding = np.array(all_embeddings[idx]).astype('float32').reshape(1, -1)
        base_city = df_raw.iloc[idx]['State']
    except IndexError:
        print(f"Warning: ID {id_} not found. Using attribute-based fallback.")

        # Fetch the listing from your DB to extract attributes
        try:
            listing = Listing.objects.get(zpid=id_)  # assumes ID is always valid in DB
        except Listing.DoesNotExist:
            return None

        # Extract core attributes
        target_city = listing.city
        target_beds = listing.bedrooms
        target_baths = listing.bathrooms
        target_price = listing.price

        # Define tolerance ranges
        price_margin = 0.25  # 15% price range
        price_min = target_price * (1 - price_margin)
        price_max = target_price * (1 + price_margin)

        # Filter in the dataset for similar items
        fallback_items = df_raw[
            (df_raw['City'] == target_city) &
            (abs(df_raw['Beds'] - target_beds) <= 2) &
            (abs(df_raw['Bath'] - target_baths) <= 2) &
            (df_raw['Price'] >= price_min) &
            (df_raw['Price'] <= price_max)
        ]

        # If nothing matches exactly, relax filters incrementally (optional)
        if fallback_items.empty:
            fallback_items = df_raw[df_raw['City'] == target_city].head(10)

        fallback_indices = fallback_items.index
        query_embeddings = np.array([all_embeddings[i] for i in fallback_indices]).astype('float32')
        query_embedding = np.mean(query_embeddings, axis=0, keepdims=True)
        faiss.normalize_L2(query_embedding)

        # Pick any fallback index to get the base city (e.g. the first one)
        fallback_idx = fallback_indices[0]
        base_city = df_raw.iloc[fallback_idx]['State']

    # Retrieve and normalize query embedding
    faiss.normalize_L2(query_embedding)

    # Search top candidates via FAISS
    D, I = index.search(query_embedding, k=quantity_similar_items + 50)
    top_indices = I[0]
    top_scores = D[0]

    # Build result DataFrame
    results_df = df_raw.iloc[top_indices].copy()
    results_df['Similarity'] = top_scores

    # Filter out the query item itself
    results_df = results_df[results_df['Uniq Id'] != id_]
    results_df = results_df[
        (results_df['State'] == base_city)
    ]

    # Apply user preference scoring
    if user_weights is not None:
        preference_columns = ['Price', 'Sqr Ft', 'Beds', 'Bath']
        preference_features = results_df[preference_columns]
        preference_scores = preference_features.dot(user_weights)
        results_df['PrefScore'] = preference_scores
        results_df['FinalScore'] = alpha * results_df['Similarity'] + (1 - alpha) * results_df['PrefScore']
    else:
        results_df['FinalScore'] = results_df['Similarity']

    # Sort and return top results
    top_sorted = results_df.sort_values('FinalScore', ascending=False)
    top_results = top_sorted.head(quantity_similar_items)

    # Match to Django Listing objects using 'Uniq Id'
    # Normalize Uniq Ids
    top_results['Uniq Id'] = top_results['Uniq Id'].astype(str).str.strip()
    uniq_ids = top_results['Uniq Id'].tolist()

    # Ensure zpid is stored and compared as string
    similar_listings = Listing.objects.filter(zpid__in=uniq_ids)

    # Debug: log missing
    found_ids = set(similar_listings.values_list('zpid', flat=True))
    missing_ids = set(uniq_ids) - found_ids
    if missing_ids:
        print(f"Missing listings for Uniq Ids: {missing_ids}")

    return similar_listings


# Create your views here.
def buyer_signIn(request):
    if request.method == 'POST':
        form = BuyerSignUpForm(request.POST)
        print(form)
        if form.is_valid():
            newUser = form.save()
            return redirect('Login')
        else:
            print(form.errors)
            return redirect('buyer_signIn') 
    else:
        form = BuyerSignUpForm()
        return render(request, "b_signIn.html", {"buyerF": form })
    
def seller_signIn(request):
    if request.method == 'POST':
        form = SellerSignUpForm(request.POST)
        print(form)
        if form.is_valid():
            newUser = form.save()
            return redirect('Login')
        else:
            print(form.errors)
            return redirect('seller_signIn') 
    else:
        form = SellerSignUpForm()
        return render(request, "s_signIn.html", {"sellerF": form })

def Login(request):
    if request.method == 'POST':
        UForm = loginForm(request.POST)

        if UForm.is_valid():
            username = UForm.cleaned_data['username']
            password = UForm.cleaned_data['password']
            print(username, ", ", password)
            
            user = authenticate(request, username=username, password=password)
            print(user)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                return render(request, 'login.html', {'error':'invalid details', 'LForm': loginForm})
        else:
            raise ValueError("Invalid value: %(value)s")
    else:
        return render(request, 'login.html', {'LForm': loginForm})

def Logout(request):
    logout(request)
    return redirect('Login')

@login_required
def home(request):
    user = request.user
    # Popular listings sorted by views
    popular = Listing.objects.order_by('-views')[:5]

    if user.role == 'buyer':
        profile = user.buyerprofile
        location = profile.address
        budget = profile.budget

        forYou = Listing.objects.filter(    
            Q(county__icontains=location) |
            Q(city__icontains=location) |
            Q(streetAddress__icontains=location),
            price__lte = budget
        )

        listings = Listing.objects.filter(    
            Q(county__icontains=location) |
            Q(city__icontains=location) |
            Q(streetAddress__icontains=location),
            price__lte = budget
        )

        # Prevent indexing errors
        if listings.count() < 1:
            forYou = []
        else:
            weights = get_by_preferences(profile)
            forYou = []
            seed_index = 0

            while len(forYou) < 5 and seed_index < listings.count():
                seed = listings[seed_index]
                seed_index += 1

                similar = recommend(seed.zpid, 5, weights)
                if similar:
                    forYou.extend(similar)

            # De-duplicate, limit to 5, and check forYou is not None/empty
            if forYou:
                # Use a dictionary to remove duplicates by zpid
                cleaned_forYou = {item.zpid: item for item in forYou if hasattr(item, 'zpid')}
                print(cleaned_forYou)
                
                # Get top 5 unique items
                forYou = Listing.objects.filter(zpid__in = list(cleaned_forYou.keys()))[:5]
            else:
                # If no recommendations returned, fall back gracefully
                forYou = []

        nearYou = listings[:5]
        saved = profile.saved_listings

        return render(request, 'home.html', {'forYou':forYou, 'nearYou':nearYou,'saved': saved})

    elif user.role == 'seller':
        # Initia
        profile = user.sellerprofile
        listings = Listing.objects.filter(seller_id = profile)[:5]

        return render(request, 'home.html', {'saved': listings, 'popular': popular})
        

def listing(request, listing_id):
    # Fetch the listing by ID or return a error message
    try:
        listing_obj = get_object_or_404(Listing, pk=listing_id)
    except Exception as e:
        messages.error(request, f"Unable to fetch listing: {e}")
        return redirect('home')
    
    # Increment the views count
    listing_obj.views += 1
    listing_obj.save(update_fields=['views'])
    is_seller = False
    is_saved = False

    reccs = recommend(f'{ listing_obj.zpid }', 5)
    print(listing_obj.zpid)
    print(reccs)

    if request.user.role == 'buyer':
        buyerAcc = request.user.buyerprofile
        is_saved = buyerAcc.saved_listings.filter(pk=listing_id).exists()
        form = None

    elif listing_obj.seller_id == request.user.sellerprofile:
        is_seller = True
        form = updateListingForm(initial={
            'listing_name': listing_obj.listing_name,
            'price': listing_obj.price,
            'space': listing_obj.space,
            'bedrooms': listing_obj.bedrooms,
            'bathrooms': listing_obj.bathrooms,
            'county': listing_obj.county,
            'city': listing_obj.city,                        
            'address': listing_obj.streetAddress,
            'status': listing_obj.status
        })
    else:
        is_saved = False
        is_seller = False
        form = None

    return render(request, 'Listing.html', {'listing': listing_obj, 'is_saved': is_saved,
                                             'form': form, 'is_seller': is_seller, 'reccs': reccs})

def updateListing(request, listing_id):
    try:
        listing = get_object_or_404(Listing, pk=listing_id)
    except Exception as e:
        messages.error(request, f"Unable to fetch listing: {e}")
        return redirect('home')

    if request.method == 'POST':
        form = updateListingForm(request.POST)
        if form.is_valid():
            listing.listing_name = form.cleaned_data['budget']
            listing.price = float(form.cleaned_data['price'])
            listing.space = float(form.cleaned_data['space'])            
            listing.bedrooms = int(form.cleaned_data['bedrooms'])
            listing.bathrooms = int(form.cleaned_data['bathrooms'])
            listing.county = form.cleaned_data['county']
            listing.city = form.cleaned_data['city']           
            listing.streetAddress = form.cleaned_data['streetAddress']
            listing.status = form.cleaned_data['status']
            listing.save()
            return redirect('Listing', listing.pk)
        else:
            messages.error(request, "Invalid form.")
            return redirect('Listing', listing.pk) 
    else:
        messages.error(request, "Invalid request method.")
        return redirect('Listing', listing.pk) 

def query_results(query):
    results = Listing.objects.filter(
        Q(listing_name__icontains=query) |
        Q(county__icontains=query) |
        Q(city__icontains=query) |
        Q(streetAddress__icontains=query) |
        Q(bedrooms__icontains=query) |
        Q(bathrooms__icontains=query) |
        Q(space__icontains=query)
    )
    return results

def search(request):
    if request.method == 'POST':
        # Retrieve search query from POST data
        search_query = request.POST.get('searchQuery')
        if not search_query:
            messages.error(request, "Please enter a valid search query.")
            return render(request, 'search.html', {})

        # Perform a keyword search based on the query
        search_results = query_results(search_query)

        # Save query in session for later use
        request.session['Query'] = search_query

        # Render results or show an empty state
        if not search_results.exists():
            messages.error(request, "No listings match your search query.")
            return render(request, 'search.html', {'Results': []})

        return render(request, 'search.html', {'Results': search_results[:15]})

    elif request.method == 'GET':
        # Retrieve filters from GET data
        max_price = request.GET.get('price', None)
        preferred_location = request.GET.get('location', None)
        property_type = request.GET.get('type', None)

        # Retrieve previous query from session
        search_query = request.session.get('Query', '')

        # Base query for filtering results
        search_results = query_results(search_query)

        # Apply additional filters if provided
        if max_price:
            search_results = search_results.filter(price__lte=max_price)

        if preferred_location:
            search_results = search_results.filter(
                Q(county__icontains=preferred_location) |
                Q(city__icontains=preferred_location) |
                Q(streetAddress__icontains=preferred_location)
            )
        
        if property_type != "None":
            search_results = search_results.filter(type__iexact=property_type)

        # Render results or show an empty state
        if not search_results.exists():
            messages.error(request, "No listings match your query and filters.")
            return render(request, 'search.html', {'searchResults': []})

        return render(request, 'search.html', {'Results': search_results[:15]})
    else:
        # Default fallback for unsupported HTTP methods
        messages.error(request, "Invalid request method.")
        return render(request, 'search.html', {})

def buyerListings(request, name):
    # Ensure the logged-in user matches the username in the URL
    if request.user.username != name:
        return redirect('home')  # or return 403

    try:
        profile = request.user.buyerprofile
    except AttributeError:
        messages.error(request, "Buyer profile not found.")
        return redirect('home')

    saved_listings = profile.saved_listings.all()

    form = PreferenceForm(initial={
        'budget': profile.budget,
        'address': profile.address,
        'P_price': profile.P_price,
        'P_square_feet': profile.P_square_feet,
        'P_bedrooms': profile.P_bedrooms,
        'P_bathrooms': profile.P_bathrooms,
        'P_year': profile.P_year,
    })

    return render(request, 'buyerListing.html', {
        'saved_listings': saved_listings,
        'PForm': form,
        'profile': profile
    })


def updatePrefrences(request):
    user = request.user
    try:
        profile = user.buyerprofile
    except AttributeError:
        messages.error(request, "Buyer profile not found.")
        return redirect('home')

    if request.method == 'POST':
        form = PreferenceForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Preferences updated successfully.")
            return redirect('buyerListings', user.username)
        else:
            messages.error(request, "Form is invalid. Please correct the errors.")
            return redirect('buyerListings', user.username)
    else:
        messages.error(request, "Invalid request method.")
        return redirect('buyerListings', user.username)

def sellerListings(request):
    seller = request.user
    sellerListings = Listing.objects.filter(seller_id=seller.sellerprofile)[:10]
    print(sellerListings)

    if request.method == 'POST':
        listing_form = CreateNewListing(request.POST, seller=request.user)

        if listing_form.is_valid():
            new_listing = listing_form.save()
            print("LISTING SAVED")
            return redirect('sellerListings')
        else:
            print(listing_form.errors)
    else:
        listing_form = CreateNewListing(seller=request.user)

    return render(request, 'sellerListing.html', {
        'seller_Listings': sellerListings,
        'ListingForm': listing_form
    })

def save(request):
    if request.method == "POST":
        Listing_id = request.POST.get('listing_id')
        listing = Listing.objects.filter(pk=Listing_id).first()
        
        if not listing:
            print("Listing not found.")
            return redirect("home")
        
        buyerAcc = request.user.buyerprofile
        buyerAcc.saved_listings.add(listing)
        return redirect("buyerListings", request.user.username)
    else:
        messages.error(request,"Form data is not valid.") 
        return redirect("home")
    
def remove(request):
    if request.method == 'POST':
        listing_id = request.POST.get('listing_id')
        listing = Listing.objects.filter(pk=listing_id).first()
        
        if not listing:
            return redirect("sellerListings")
        
        acc = request.user.sellerprofile
        
        if listing.seller_id == acc:
            listing.delete()
            return redirect("sellerListings")
        else:
            return redirect("sellerListings") 
    else:
        messages.error(request, "Form data is not valid.")
        return redirect('sellerListings')

def unsave(request):
    if request.method == "POST":
        listing_id = request.POST.get('listing_id')
        listing = Listing.objects.filter(pk=listing_id).first()
        
        if not listing:
            print("Listing not found.")
            return redirect("home")
        
        buyerAcc = request.user.buyerprofile
        is_saved = buyerAcc.saved_listings.filter(pk=listing_id).exists()
        
        if not buyerAcc:
            return redirect('Login')
        
        if is_saved:
            buyerAcc.saved_listings.remove(listing)
            return redirect("buyerListings", request.user.username)
        else:
            print("no listing in saved listing.") 
            return redirect("buyerListings", request.user.username)
    else:
        messages.error(request, "Form data is not valid.")
        return redirect("home")
    

def newchat(request, id):
    acc = request.user # Logged-in user
    chats = Chats.objects.filter(users=acc)  # Fetch current user's chats.
    
    seller_p = get_object_or_404(sellerProfile, pk=id)  # Get seller profile
    sellerAcc = seller_p.user
    if not seller_p:
        print(f"No Seller entry found for Account ID {id}")
    
    print(f"Logged-in Account: {acc}")
    print(f"Seller's Account: {sellerAcc}")

    existing_chat = Chats.objects.filter(users=acc).filter(users=sellerAcc).first()
    if existing_chat:
        return redirect('profile')
        # If chat exists, redirect to profile with chat and logs
    
    if request.method == 'POST':
        chat_form = CreateChatForm(request.POST)
        if chat_form.is_valid():
            chat = chat_form.save(commit=False)
            chat.last_updated = now()
            chat.save()
            chat.users.add(sellerAcc, acc)

            init_message = ChatLogs.objects.create(
                chat_id=chat.pk,
                message="Hey!",
                sent=now(),
                from_user=acc
            )
            init_message.save()
            newchatlogs = ChatLogs.objects.filter(chat_id=chat)
            
            for log in newchatlogs:
                print('saved, chatlog id: ' + str(log.chat_id))
            return render(request, 'profile.html',  {'chats': chats, 'chatlogs': newchatlogs,'id':acc.pk, 'most_recent_chat': chat})
        else:
            messages.error(request, "Form data is not valid.")
            return redirect('profile') 
    else:
        messages.error(request, "Incorrect method.")
        return redirect('profile') 

def send(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)  # Parse JSON data
            print(data)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)

        message = data.get('message')
        chatID = data.get('chat_id')
        print(f"chat:{chatID} and message:{ message }")

        acc = request.user 

        if not message or not chatID:
            return JsonResponse({'error': 'Missing message or chat_id'}, status=400)

        Chat = get_object_or_404(Chats, id=chatID)

        new_log = ChatLogs.objects.create(
            chat_id= Chat.id,
            message=message,
            from_user=acc,
            sent=now()
        )

        return JsonResponse({
            'chat_id': Chat.id,
            'id': new_log.id,
            'from_user': new_log.from_user.username,
            'message': new_log.message,
            'sent': new_log.sent.isoformat(),
        })
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

def profile(request, id=None):
    # Fetch the logged-in user's account
    user = request.user

    if not user:
        print(f"No Account object found for username: {user}")
        return redirect('login')
 
    # Fetch all chats involving the user
    chats = Chats.objects.filter(users=user).order_by('-last_updated')

    # If a specific chat ID is provided
    if request.method == 'GET' and id is not None:
        chat = Chats.objects.filter(pk=id).first()
        
        if not chat:
            print(f"No chat found with ID {id}.")
            return redirect('home')  # Redirect if the chat is not found

        chatlogs = ChatLogs.objects.filter(chat_id=chat).order_by('sent')

        return render(request, 'profile.html', {
            'chatlogs': chatlogs,
            'chats': chats,
            'most_recent_chat': chat,
            'chat_id': chat.pk,
        })
    else:
        # Fetch the most recent chat
        most_recent_chat = chats.first()

        # Fetch chat logs for the most recent chat
        logs = []
        if most_recent_chat:
            logs = ChatLogs.objects.filter(chat_id=most_recent_chat).order_by('sent')
            if logs.exists():
                print('saved, chatlog id', logs.first().chat_id)
            else:
                print("No chat logs found.")
        else:
            print("No chats found for the user.")

        chat_id = most_recent_chat.pk if most_recent_chat else None

        return render(request, 'profile.html', {
            'chatlogs': logs,
            'chats': chats,
            'most_recent_chat': most_recent_chat,
            'chat_id': chat_id,
        })
