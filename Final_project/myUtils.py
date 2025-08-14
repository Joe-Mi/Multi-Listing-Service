import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Final_project.settings')
django.setup()
from Final_project.settings import BASE_DIR

import os
import joblib
import pickle
import faiss
import numpy as np
import pandas as pd
import pygeohash as pgh
from scipy.sparse import hstack, csr_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences


ohe = joblib.load("ohe.pkl")             # OneHotEncoder
sc = joblib.load("scaler.pkl")           # StandardScaler
tfidf = joblib.load("tfidf.pkl")         # TfidfVectorizer
tokenizer = joblib.load("tokenizer.pkl") # Tokenizer for descriptions
embedding_net = keras.models.load_model("embedding_model.keras")  # Your embedding model

def process_single_listing(new_row, ohe, sc, tfidf, tokenizer):
    # Ensure required fields are filled
    for col in ['Property Type', 'State', 'City', 'Normalized Features', 'Normalized Description']:
        new_row[col] = str(new_row.get(col, '') or 'Unknown')

    # Geohash ID
    new_row['geohash'] = pgh.encode(new_row['Latitude'], new_row['Longitude'], precision=5)
    geohash_category = pd.Categorical([new_row['geohash']], categories=ohe.categories_[2])  # ensure same category space
    new_row['geohash_id'] = geohash_category.codes[0]

    # Encode categorical
    cat_input = ohe.transform([[new_row['Property Type'], new_row['State'], new_row['City']]])

    # Scale numeric
    num_input = sc.transform([[new_row.get(col, 0) for col in ['Price', 'Sqr Ft', 'Lot Size', 'Beds', 'Bath', 'Year Built','Price Sqr Ft']]])

    # TF-IDF for features
    feat_input = tfidf.transform([new_row['Normalized Features']])

    # Tokenize and pad description
    seq = tokenizer.texts_to_sequences([new_row['Normalized Description']])
    padded_seq = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

    # Description embedding
    desc_embed = embedding_net.predict(padded_seq)

    # Combine all features
    combined = hstack([
        csr_matrix(desc_embed),
        csr_matrix(num_input),
        cat_input,
        csr_matrix([[new_row['geohash_id']]])
    ])
    
    return combined.astype('float32')

def add_listing_to_recommender(new_listing):
    # 1. Load the current embeddings and index
    with open("all_embeddings_2.pkl", "rb") as f:
        all_embeddings = pickle.load(f)

    index = faiss.read_index("faiss_index_2.index")

    # 2. Format the listing dict for embedding
    listing_dict = {
        'Uniq Id': new_listing.zpid,
        'Price': new_listing.price,
        'Sqr Ft': new_listing.space,
        'Beds': new_listing.bedrooms,
        'Bath': new_listing.bathrooms,
        'Lot Size': new_listing.space + (new_listing.space * 0.25),
        'Year Built': 2025,
        'Price Sqr Ft': (new_listing.space / new_listing.price),
        'Latitude': new_listing.latitude,
        'Longitude': new_listing.longitude,
        'City': new_listing.city,
        'State': new_listing.county,
        'Property Type': new_listing.type,
        'Normalized Features': '',
        'Normalized Description': new_listing.listing_name
    }

    # 3. Process into final input vector
    processed_vector = process_single_listing(pd.Series(listing_dict), ohe, sc, tfidf, tokenizer).toarray()
    faiss.normalize_L2(processed_vector)

    # 4. Update FAISS index and embedding array
    index.add(processed_vector)
    all_embeddings = np.vstack([all_embeddings, processed_vector])

    # 5. Save updated index and embedding file
    faiss.write_index(index, "faiss_index_2.index")
    with open("all_embeddings_2.pkl", "wb") as f:
        pickle.dump(all_embeddings, f)

    # 6. Append to data.csv (as backup cache)
    df_path = os.path.join(BASE_DIR, "data.csv")
    df = pd.read_csv(df_path)
    df = pd.concat([df, pd.DataFrame([listing_dict])], ignore_index=True)
    df.to_csv("data.csv", index=False)
