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

# Collaborative Filtering (NMF-based) - Only top-rated products
def collaborative_filtering(data, category, skin_concerns):
    user_enc = LabelEncoder()
    product_enc = LabelEncoder()
    
    # Filter data for the selected category
    filtered_data = data[data['category'].str.lower() == category.lower()]
    
    if filtered_data.empty:
        return pd.DataFrame(columns=['product_name', 'category', 'skintype', 'concern', 'user_rating'])
    
    # Filter out products with lower ratings (e.g., 5 or 4 star ratings only)
    filtered_data = filtered_data[filtered_data['user_rating'].isin([4, 5])]
    
    filtered_data['user_id'] = user_enc.fit_transform(filtered_data['product_id'])
    filtered_data['product_id'] = product_enc.fit_transform(filtered_data['product_id'])
    
    num_users = filtered_data['user_id'].nunique()
    num_products = filtered_data['product_id'].nunique()
    
    if num_users == 0 or num_products == 0:
        return pd.DataFrame(columns=['product_name', 'category', 'skintype', 'concern', 'user_rating'])
    
    # Create user-product interaction matrix
    user_product_matrix = coo_matrix(
        (filtered_data['user_rating'], 
         (filtered_data['user_id'], filtered_data['product_id'])),
        shape=(num_users, num_products)
    )
    
    try:
        # Apply NMF
        model = NMF(n_components=20, init='random', random_state=42)
        user_embedding = model.fit_transform(user_product_matrix)
        product_embedding = model.components_
        
        # Calculate product similarity
        item_similarity = cosine_similarity(product_embedding)
        
        # Find top product indices and get product IDs
        top_product_indices = item_similarity.argsort(axis=1)[-5:].flatten()
        top_product_ids = product_enc.inverse_transform(top_product_indices)
        
        # Filter products by skin concerns
        relevant_products = filtered_data[filtered_data['product_id'].isin(top_product_ids)]
        relevant_products = relevant_products[relevant_products['concern'].str.contains(skin_concerns, case=False)]
        
        return relevant_products[['product_name', 'category', 'skintype', 'concern', 'user_rating']]
    
    except:
        # In case of errors with sparse data
        return pd.DataFrame(columns=['product_name', 'category', 'skintype', 'concern', 'user_rating'])

# Fallback: Top-Rated Products
def top_rated_products(data, category, skin_concerns):
    filtered_data = data[data['category'].str.lower() == category.lower()]
    # Filter out products with lower ratings (e.g., 5 or 4 star ratings only)
    filtered_data = filtered_data[filtered_data['user_rating'].isin([4, 5])]
    
    # Filter products by skin concerns
    relevant_products = filtered_data[filtered_data['concern'].str.contains(skin_concerns, case=False)]
    
    return relevant_products[['product_name', 'category', 'skintype', 'concern', 'user_rating']]

# Content-Based Filtering
def content_based_filtering(data, category, skin_type, skin_concerns):
    vectorizer = TfidfVectorizer(stop_words="english")
    
    # Filter data by category
    filtered_data = data[data['category'].str.lower() == category.lower()]
    
    # Generate combined text for TF-IDF
    data_text = filtered_data[['product_name', 'category', 'skintype', 'concern']].apply(
        lambda x: ' '.join(x.map(str)), axis=1
    )
    tfidf_matrix = vectorizer.fit_transform(data_text)
    
    # Create user query vector
    user_features = f"{skin_concerns} {skin_type.lower()} {category.lower()}"
    user_vector = vectorizer.transform([user_features])
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    filtered_data['SimilarityScore'] = similarity_scores
    
    # Filter products by skin concerns and skin type
    relevant_products = filtered_data[filtered_data['concern'].str.contains(skin_concerns, case=False)]
    relevant_products = relevant_products[relevant_products['skintype'].str.contains(skin_type, case=False)]
    
    # Sort by similarity score and return
    return relevant_products.sort_values(by='SimilarityScore', ascending=False)[
        ['product_name', 'category', 'skintype', 'concern', 'user_rating']
    ]

# Main Streamlit app
def main():    
    st.title("Skincare Recommendation System")
    
    # Load data
    data = load_data("dataset.csv")
    
    # Sidebar Inputs
    category = st.sidebar.selectbox("Select Product Category", data["category"].unique())
    skin_type_options = ["Oily", "Dry", "Combination", "Sensitive", "Normal"]
    skin_type = st.sidebar.selectbox("Select Skin Type", skin_type_options)
    skin_concerns = st.sidebar.text_input("Enter Skin Concerns (comma-separated)", "Acne, Pigmentation")
    
    # Recommendations button
    if st.button("Get Recommendations"):
        # Content-based recommendations
        st.subheader("Content-Based Recommendations")
        content_recs = content_based_filtering(data, category, skin_type, skin_concerns)
        st.table(content_recs)
        
        # Collaborative filtering (with fallback to top-rated)
        st.subheader("Hybrid Recommendations (Collaborative + Top-Rated Fallback)")
        collab_recs = collaborative_filtering(data, category, skin_concerns)
        
        if collab_recs.empty or len(collab_recs) < 1:
            st.write("Collaborative filtering data is insufficient; showing top-rated products instead.")
            collab_recs = top_rated_products(data, category, skin_concerns)
        
        st.table(collab_recs)


if __name__ == "__main__":
    main()
