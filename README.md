# Movie_Recommender

This project is a movie recommender system built using Flask. It recommends movies based on user input by leveraging TF-IDF vectorization and cosine similarity.

Features
Recommends movies based on a given movie title
Uses TF-IDF vectorization for text feature extraction
Calculates cosine similarity to find similar movies
Web interface for user interaction
Technologies Used
Python
Flask
Pandas
Scikit-learn
NumPy
Difflib

Installation
Clone the repository:

bash
git clone https://github.com/your-repository/movie-recommender.git
cd movie-recommender

Usage
Run the Flask application:

bash
python app.py
Open your browser and navigate to http://127.0.0.1:5000/.

Enter a movie title to get recommendations.

Files
app.py: Main application file
content/movies.csv: Dataset of movies
templates/index.html: HTML template for the web interface
movie_recommender_model.pkl: Pickle file for the saved model

How It Works
Data Loading and Preprocessing: Loads movie data and preprocesses selected features by filling missing values and combining them.
Feature Vectorization: Uses TF-IDF vectorizer to convert combined features into vectors.
Cosine Similarity Calculation: Calculates similarity between movie vectors.
Model Saving: Saves the vectorizer and similarity matrix using pickle.
Movie Recommendation: Finds the closest match to the input movie title and recommends similar movies based on cosine similarity scores.
Web Interface: Accepts user input and displays recommended movies.

Acknowledgments
Scikit-learn for providing tools for machine learning and data processing.
Flask for the web framework.
Pandas for data manipulation.
