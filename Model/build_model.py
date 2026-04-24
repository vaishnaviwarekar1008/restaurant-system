import pandas as pd
import numpy as np
import pickle
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")
df = pd.read_csv('../Dataset/archive/zomato.csv', encoding='utf-8')
print(f"Original Dataset Shape: {df.shape}")

# Pre-clean restaurant names to fix any mojibake or encoding artifacts 
def clean_string(text):
    text = str(text)
    # Common latin-1 to utf-8 artifacts
    text = text.replace('Ã©', 'e').replace('Ã¨', 'e').replace('Ã\xad', 'i')
    text = text.replace('Ã¢', 'a').replace('Ã´', 'o').replace('Ã»', 'u').replace('Ã±', 'n')
    text = text.replace('Â', '').replace('©', '')
    # Remove any crazy symbols keeping ascii only
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

df['name'] = df['name'].apply(clean_string)

# --------------------------
# 1. DATA PREPROCESSING
# --------------------------
print("Applying data preprocessing...")
# Dropping columns that are not needed for Content-Based Filtering
df.drop(['url', 'address', 'phone', 'dish_liked', 'reviews_list', 'menu_item'], axis=1, inplace=True)

# Cleaning the 'rate' column
df['rate'] = df['rate'].astype(str).apply(lambda x: x.replace('/5', '').strip())
df['rate'] = df['rate'].replace(['NEW', '-'], np.nan)
df['rate'] = df['rate'].astype(float)
df['rate'].fillna(df['rate'].mean(), inplace=True)

# Drop missing values in important categorical fields
df.dropna(subset=['cuisines', 'rest_type', 'location', 'listed_in(city)'], inplace=True)

# Drop duplicate restaurants based on 'name' to keep the similarity matrix manageable 
df.drop_duplicates(subset='name', keep='first', inplace=True)

# Optional: Sample the dataset to 4000 restaurants so Cosine Similarity fits easily in standard memory
df = df.sample(n=min(4000, len(df)), random_state=42).reset_index(drop=True)
print(f"Preprocessed Dataset Shape: {df.shape}")

# --------------------------
# 2. FEATURE ENGINEERING (Content-Based)
# --------------------------
print("Feature engineering for Content-Based Filtering...")
def create_soup(x):
    # Combine relevant text columns into a single 'soup'
    cuisines = str(x['cuisines']).replace(', ', ' ')
    rest_type = str(x['rest_type']).replace(', ', ' ')
    city = str(x['listed_in(city)']).replace(' ', '')
    return f"{cuisines} {rest_type} {city}"

df['soup'] = df.apply(create_soup, axis=1)
# Simplify the text
df['soup'] = df['soup'].apply(lambda x: re.sub('[^a-zA-Z ]', ' ', x).lower())

# Calculate similarity
print("Calculating cosine similarity matrix...")
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index, index=df['name']).drop_duplicates()

# --------------------------
# 3. EXPORT REQUIRED FILES
# --------------------------
print("Exporting data and models to Flask folder...")
# Save cleaned dataset
df.to_csv('../Flask/restaurant1.csv', index=False)

# Save the similarity matrix, indices mappings, and relevant reference columns
model_data = {
    'similarity_matrix': cosine_sim,
    'indices': indices,
    'restaurant_data': df[['name', 'cuisines', 'location', 'rate', 'approx_cost(for two people)']]
}

with open('../Flask/restaurant.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Process completed! Files saved as `restaurant1.csv` and `restaurant.pkl` inside the Flask folder.")
