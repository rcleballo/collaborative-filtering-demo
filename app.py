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

# Recommend Products Function
def recommend_products(user, actual_ratings, predicted_ratings, top_n=3):
    rated_products = actual_ratings[actual_ratings > 0]
    recommendations = predicted_ratings[rated_products == 0].sort_values(ascending=False).head(top_n)
    return recommendations

# Streamlit Interface
st.title("Product Recommendation System with PyGWalker")

# Select Customer
customer = st.selectbox("Select Customer", user_item_matrix.index)

# Predict Ratings for Selected Customer
customer_index = user_item_matrix.index.get_loc(customer)
predicted_ratings = predict_ratings(customer_index, user_similarity, user_item_matrix.values)

# Convert predictions to a Series for easier manipulation
predicted_ratings_series = pd.Series(predicted_ratings, index=user_item_matrix.columns)

# Display Actual and Predicted Ratings
st.write(f"**Actual Ratings by {customer}:**")
actual_ratings = user_item_matrix.loc[customer]
st.write(actual_ratings)

st.write(f"**Predicted Ratings for {customer}:**")
st.write(predicted_ratings_series)

# Recommend Top Products
top_recommendations = recommend_products(customer, actual_ratings, predicted_ratings_series)
st.write(f"**Top {len(top_recommendations)} Product Recommendations for {customer}:**")
st.write(top_recommendations)

# **PyGWalker Visualization**
st.write("## Explore the Data using PyGWalker")

# Use PyGWalker to create an interactive visualization of your dataframe
pyg.walk(df2)
