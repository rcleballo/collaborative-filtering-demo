import pandas as pd
import random
import pygwalker as pyg
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np

# Sample list of positive and negative reviews
positive_reviews = [
    "Great product, fast delivery!",
    "Exceeded my expectations, very happy.",
    "Amazing quality, highly recommend!",
    "Solid product, will purchase again.",
    "Good value for money."
]
negative_reviews = [
    "Satisfactory but could be better.",
    "Quality is good, but delivery was slow.",
    "Not worth the price, quite disappointed.",
    "Works as expected, nothing special.",
    "Terrible experience, would not buy again."
]

# Generate random ratings and reviews for 21 products and 10 customers
num_products = 21
num_customers = 10
product_names = [f'Product {i}' for i in range(1, num_products + 1)]
customer_ids = [f'Customer {i}' for i in range(1, num_customers + 1)]

# List to store the product review data
reviews_data = []

# Loop through each product and customer to generate ratings and reviews
for product in product_names:
    for customer in customer_ids:
        # Randomly choose whether the review is positive or negative
        is_positive = random.choice([True, False])
        
        if is_positive:
            rating = random.randint(4, 5)  # Higher rating for positive reviews
            review = random.choice(positive_reviews)
        else:
            rating = random.randint(1, 3)  # Lower rating for negative reviews
            review = random.choice(negative_reviews)
        
        reviews_data.append([product, customer, rating, review])

# Convert the data into a DataFrame for easy manipulation and analysis
df2 = pd.DataFrame(reviews_data, columns=['Product', 'Customer', 'Rating', 'Review'])

# User-Item Matrix Creation
user_item_matrix = df2.pivot_table(index='Customer', columns='Product', values='Rating').fillna(0)

# Compute User Similarity Matrix
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Predict Ratings Function
def predict_ratings(user_index, user_similarity, user_ratings):
    sim_scores = user_similarity[user_index]
    weighted_ratings = user_ratings.T.dot(sim_scores) / np.abs(sim_scores).sum()
    return weighted_ratings

# Recommend Products Based on a Selected Product
def recommend_products_based_on_product(selected_product, user_index, user_item_matrix, top_n=3):
    similar_users = user_similarity[user_index]
    predicted_ratings = np.dot(similar_users, user_item_matrix) / np.sum(np.abs(similar_users))
    
    # Convert predictions into a Series for easier manipulation
    predicted_ratings_df = pd.Series(predicted_ratings, index=user_item_matrix.columns)
    
    # Filter out products that have already been rated by the current user
    rated_products = user_item_matrix.loc[user_item_matrix.index[user_index]] > 0
    
    # Recommend only unrated products
    unrated_products = ~rated_products
    
    # Recommend top N products based on predicted ratings for unrated products
    recommendations = predicted_ratings_df[unrated_products].sort_values(ascending=False).head(top_n)
    
    # If no recommendations, display a message
    if recommendations.empty:
        return "No recommendations available, all products are rated."

    return recommendations


# Streamlit Interface
st.set_page_config(page_title="Product Recommendation System with PyGWalker", layout="wide")
st.title("Simulated Shop with Product Recommendations")

# Step 1: Select a customer
customer = st.selectbox("Select Customer", user_item_matrix.index)

# Step 2: Customer selects a product
selected_product = st.selectbox("Pick a product", product_names)

# Step 3: Predict Ratings and Recommend Products
customer_index = user_item_matrix.index.get_loc(customer)

# Recommend based on the selected product
st.write(f"**You selected: {selected_product}**")

# Recommend similar products based on the selected one
recommendations = recommend_products_based_on_product(selected_product, customer_index, user_item_matrix)
st.write(f"**Recommended Products for {customer} based on {selected_product}:**")
st.write(recommendations)

# PyGWalker Visualization
st.write("## Explore the Data using PyGWalker")
pyg.walk(df2)
