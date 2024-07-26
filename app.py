from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

app = Flask(__name__)

# Load the movies data 
movies_data = pd.read_csv('content/movies.csv')

# Preprocessing steps 
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Vectorize the combined features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

# Save the model using pickle
model_file = 'movie_recommender_model.pkl'
with open(model_file, 'wb') as file:
    pickle.dump((vectorizer, similarity), file)

# Function to recommend movies
def recommend_movies(movie_name):
    # Load the model from the pickle file
    with open(model_file, 'rb') as file:
        vectorizer_loaded, similarity_loaded = pickle.load(file)

    # finding the close match for the movie name given by the user
    find_close_match = difflib.get_close_matches(movie_name, movies_data['title'], n=1)

    if not find_close_match:
        return []  # Return empty list if no close match found

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data['title'] == close_match].index[0]

    # getting a list of similar movies
    similarity_score = list(enumerate(similarity_loaded[index_of_the_movie]))

    # sorting the movies based on their similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # get top 10 similar movies (excluding the input movie itself)
    recommended_movies = [movies_data.iloc[movie[0]]['title'] for movie in sorted_similar_movies[1:11]]

    return recommended_movies

# Route for the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_movies = []
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommended_movies = recommend_movies(movie_name)
    return render_template('index.html', recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
