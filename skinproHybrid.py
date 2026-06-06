import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF

# Load the dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Latent Feature Filtering (NMF-based Item-to-Item Similarity)
def collaborative_filtering(data, category, skin_type, db_concerns_list):
    """
    Stage 1: Train NMF globally on the product category to maximize matrix density.
    Stage 2: Post-processing safety filter to strictly enforce skin type constraints.
    """
    # Filter globally for category and quality products
    filtered_data = data[
        (data['category'].str.lower() == category.lower()) & 
        (data['user_rating'].isin([4, 5]))
    ].copy()
    
    if filtered_data.empty:
        return pd.DataFrame(columns=['product_name', 'category', 'skintype', 'concern', 'user_rating'])
    
    # Text vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    data_text = filtered_data[['product_name', 'skintype', 'concern']].apply(
        lambda x: ' '.join(x.map(str)), axis=1
    )
    tfidf_matrix = vectorizer.fit_transform(data_text)
    
    try:
        n_samples, n_features = tfidf_matrix.shape
        n_components = min(10, n_samples, n_features)
        
        if n_components < 1:
            return pd.DataFrame(columns=['product_name', 'category', 'skintype', 'concern', 'user_rating'])
            
        # Run NMF on global matrix
        nmf_model = NMF(n_components=n_components, init='random', random_state=42, max_iter=400)
        item_profiles = nmf_model.fit_transform(tfidf_matrix)
        
        item_similarity = cosine_similarity(item_profiles)
        
        concern_regex = "|".join(db_concerns_list) if db_concerns_list else ".*"
        concern_mask = filtered_data['concern'].str.contains(concern_regex, case=False).values
        matched_indices = [i for i, matched in enumerate(concern_mask) if matched]
        
        if not matched_indices:
            return pd.DataFrame(columns=['product_name', 'category', 'skintype', 'concern', 'user_rating'])
            
        similarity_scores = item_similarity[matched_indices].mean(axis=0)
        filtered_data['SimilarityScore'] = similarity_scores
        
        # --- STAGE 2: Post-Processing Deterministic Safety Filter ---
        safe_data = filtered_data[filtered_data['skintype'].str.contains(skin_type, case=False)]
        
        final_recs = safe_data.sort_values(by='SimilarityScore', ascending=False)
        return final_recs[['product_name', 'category', 'skintype', 'concern', 'user_rating']].drop_duplicates().head(5)
        
    except Exception as e:
        print(f"Latent Filtering Error: {e}")
        return pd.DataFrame(columns=['product_name', 'category', 'skintype', 'concern', 'user_rating'])

# Fallback: Top-Rated Products with Safety Constraints
def top_rated_products(data, category, skin_type, db_concerns_list):
    filtered_data = data[data['category'].str.lower() == category.lower()].copy()
    high_rated = filtered_data[filtered_data['user_rating'].isin([4, 5])]
    
    # Enforce strict skin type safety first
    safe_high_rated = high_rated[high_rated['skintype'].str.contains(skin_type, case=False)]
    
    concern_regex = "|".join(db_concerns_list) if db_concerns_list else ".*"
    matched_products = safe_high_rated[safe_high_rated['concern'].str.contains(concern_regex, case=False)]
    
    if matched_products.empty:
        return safe_high_rated[['product_name', 'category', 'skintype', 'concern', 'user_rating']].head(5)
        
    return matched_products[['product_name', 'category', 'skintype', 'concern', 'user_rating']].head(5)

# Content-Based Filtering
def content_based_filtering(data, category, skin_type, db_concerns_list):
    vectorizer = TfidfVectorizer(stop_words="english")
    filtered_data = data[data['category'].str.lower() == category.lower()].copy()

    data_text = filtered_data[['product_name', 'category', 'skintype', 'concern']].apply(
        lambda x: ' '.join(x.map(str)), axis=1
    )
    tfidf_matrix = vectorizer.fit_transform(data_text)

    concerns_str = " ".join(db_concerns_list) if db_concerns_list else ""
    user_features = f"{concerns_str} {skin_type.lower()} {category.lower()}"
    user_vector = vectorizer.transform([user_features])
    
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    filtered_data['SimilarityScore'] = similarity_scores
  
    concern_regex = "|".join(db_concerns_list) if db_concerns_list else ".*"
    relevant_products = filtered_data[filtered_data['concern'].str.contains(concern_regex, case=False)]
    relevant_products = relevant_products[relevant_products['skintype'].str.contains(skin_type, case=False)]
    
    return relevant_products.sort_values(by='SimilarityScore', ascending=False)[
        ['product_name', 'category', 'skintype', 'concern', 'user_rating']
    ]

def main():    
    st.title("Skincare Recommendation System")

    data = load_data("dataset.csv")
    
    category = st.sidebar.selectbox("Select Product Category", data["category"].unique())
    skin_type_options = ["Oily", "Dry", "Combination", "Sensitive", "Normal"]
    skin_type = st.sidebar.selectbox("Select Skin Type", skin_type_options)
    
    concern_options = {
        "Acne": "acne",
        "Blackheads": "blackhead",
        "Whiteheads": "whitehead",
        "Broken Barrier": "brokenbarrier",
        "Dark Spots": "darkspots",
        "Exfoliation": "exfoliation",
        "Hydration": "hydration",
        "Irritation": "irritation",
        "Pigmentation": "pigmentation",
        "Pores": "pores",
        "Skin Soothing": "skinsoothing",
        "Sun Protection": "sunprotection"
    }
    
    selected_concerns = st.sidebar.multiselect(
        "Select Skin Concerns (Choose multiple)", 
        options=list(concern_options.keys()),
        default=["Acne", "Pigmentation"]
    )

    db_concerns_list = [concern_options[c] for c in selected_concerns]
    
    if st.button("Get Recommendations"):
        st.subheader("Direct Match Content-Based Recommendations")
        content_recs = content_based_filtering(data, category, skin_type, db_concerns_list)
        st.table(content_recs)
        
        st.subheader("Latent Feature Recommendations (NMF Matrix Filter)")
        collab_recs = collaborative_filtering(data, category, skin_type, db_concerns_list)
        
        if collab_recs.empty or len(collab_recs) < 1:
            st.write("Latent semantic data is insufficient; showing top-rated products instead.")
            collab_recs = top_rated_products(data, category, skin_type, db_concerns_list)
        
        st.table(collab_recs)

if __name__ == "__main__":
    main()
