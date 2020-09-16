# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:56:49 2020
@author: William Ashton
"""
# Import packages and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate
from sklearn.metrics.pairwise import cosine_similarity

# Read in dataset
movies = pd.read_csv('ml-latest-small/movies.csv',)
ratings = pd.read_csv('ml-latest-small/ratings.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')

# User input
print()
print("Hybrid Film Recommender Script, for Project module CMP3753M")
print()
print("Which operation would you like to perform?")
print("1) Content-based filtering (Item-based)")
print("2) Collaborative filtering (Item-based)")
print("3) Weighted hybrid filtering")
print("4) Collaborative prediction model")
user_option = int(input()) # Store user's choice of method
print()

# Data pre-processing

movies['genres'] = movies['genres'].str.replace('|',' ') # Replace | with blank for formatting
filtered_ratings = ratings.groupby('userId').filter(lambda x: len(x) >=20) # Group ratings by user ID, only keeping users who have rated 15 or more films
filtered_movies = filtered_ratings.movieId.unique().tolist() # Removes films which haven't been rated by users in filtered list

remaining_movies = len(filtered_ratings.movieId.unique())/len(movies.movieId.unique()) * 100 # Percentage of remaining films
#print(remaining_movies)
remaining_users = len(filtered_ratings.userId.unique())/len(ratings.userId.unique()) * 100 # Percentage of remaining users
#print(remaining_users)

movies = movies[movies.movieId.isin(filtered_movies)] # Apply filter to movies dataframe
# Remove timestamps, as they're unnecessary
tags.drop(['timestamp'], 1, inplace=True)
filtered_ratings.drop(['timestamp'], 1, inplace=True)

# Get tags and genres as attributes
movie_metadata=pd.merge(movies, tags, on='movieId', how='left') # Merge tags and movies
movie_metadata.fillna("", inplace=True) # Replace NaN with blankspace
movie_metadata=pd.DataFrame(movie_metadata.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x))) # Compile all of a films tags into one row
movie_metadata = pd.merge(movies, movie_metadata, on='movieId', how='left') # Re-merge movies and new metadata df
movie_metadata['metadata'] = movie_metadata[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1) # Join genres and tags

# Item-based

# Content-based filtering method
def content_based_filtering():

    # Convert metadata into matrix of TF-IDF features using LSA (SVD)
    vectorizer = TfidfVectorizer(stop_words='english') # Create tfidf method, using english stop words
    tfidf_matrix = vectorizer.fit_transform(movie_metadata['metadata']) # Apply tfidf method to genres and tags
    tfidf_data = pd.DataFrame(tfidf_matrix.toarray(), index=movie_metadata.index.tolist()) # Convert into dataframe
    svd = TruncatedSVD(n_components=200) # Create truncated SVD model, with 200 dimensions
    features_matrix = svd.fit_transform(tfidf_data) # Fit truncated SVD model to produce approximated matrices of features
    features_df = pd.DataFrame(features_matrix, index = movie_metadata.title.tolist()) # Convert into dataframe

    # Add vectors of user inputted films and tf-idf features
    feature_vector = np.array(features_df.loc[title]).reshape(1, -1) # For one film
    if input_no > 1:
        feature_vector += np.array(features_df.loc[title2]).reshape(1, -1) # For 2 films
        if input_no > 2:
            feature_vector += np.array(features_df.loc[title3]).reshape(1, -1) # For 3 films

    # Find similar films
    content_score = cosine_similarity(features_df, feature_vector).reshape(-1) # Calculate cosine similarity between inputted films and other films in dataset
    dictionary = {'Content':content_score} # Simple dictionary for the header of the new DF
    content_recommendations = pd.DataFrame(dictionary, index = features_df.index) # Create df of recommendations
    content_recommendations.sort_values(['Content'], ascending=False, inplace=True) # Sort values by similarity score, descending

    return content_recommendations, content_score, features_df # Return recommendations

# Collaborative filtering method
def collaborative_filtering():

    # Merge movies and ratings
    filtered_ratings_1 = pd.merge(movies[['movieId']], filtered_ratings, on="movieId", how="right")
    filtered_ratings_1 = filtered_ratings_1.pivot(index='movieId', columns='userId', values='rating').fillna(0) # Reshape by movieId

    # Convert ratings into matrix of TF-IDF features using SVD
    svd = TruncatedSVD(n_components=150) # Create truncated SVD model, with less dimensions
    latent_ratings_matrix = svd.fit_transform(filtered_ratings_1) # Fit truncated SVd model to produce approximated matrices of features
    latent_ratings_df = pd.DataFrame(latent_ratings_matrix, index=movie_metadata.title.tolist()) # Convert to df

    # Add vectors of user inputted films and rating data
    rating_vector = np.array(latent_ratings_df.loc[title]).reshape(1, -1) # One film
    if input_no > 1:
        rating_vector += np.array(latent_ratings_df.loc[title2]).reshape(1, -1) # 2
        if input_no > 2:
            rating_vector += np.array(latent_ratings_df.loc[title3]).reshape(1, -1) # 3

    # Find similar films
    collab_score = cosine_similarity(latent_ratings_df, rating_vector).reshape(-1) # Calculate cosine similarity between inputted films and other films in dataset
    dictionary = {'Collaborative':collab_score} # Dictionary for header
    collab_recommendations = pd.DataFrame(dictionary, index = latent_ratings_df.index) # Create df of recommendations
    collab_recommendations.sort_values('Collaborative', ascending=False, inplace=True) # Sort values by similarity score, descending
    
    return collab_recommendations, collab_score # Return recommendations

# Weighted hybrid filtering method
def hybrid_filtering():

    # Get scores from previous methods
    content_score = content_based_filtering()[1]
    collab_score = collaborative_filtering()[1]
    features_df = content_based_filtering()[2] # Get features for indexing

    # Compile recommendations
    hybrid_score = ((content_score + collab_score)/2) # Calculate average score for each film
    dictionary = {'Content': content_score, 'Collaborative':collab_score, 'Hybrid':hybrid_score} # Dictionary for header
    hybrid_recommendations = pd.DataFrame(dictionary, index=features_df.index) # Create df of recommendations
    hybrid_recommendations.sort_values('Hybrid', ascending=False, inplace=True) # Sort values by hybrid, descending

    return hybrid_recommendations # Return recommendations

# Collaborative prediction model
def predict_user_ratings():

    # Format ratings data
    filtered_ratings_1 = filtered_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0) # Reshape by userId
    # Perform SVD with NumPy
    users, sigma, vector = svds(filtered_ratings_1, k=18) # Create and apply SVD model
    model = SVD(n_factors=18) # Declare model to be evaluated with surprise
    sigma = np.diag(sigma) # Construct diagonal matrix
    latent_ratings_matrix = np.dot(np.dot(users, sigma), vector) # Get matrix of rating predictions through dot product
    predicted_ratings = pd.DataFrame(latent_ratings_matrix, columns = filtered_ratings_1.columns) # Convert to dataframe

    # Evaluate
    reader = Reader(rating_scale=(1, 5)) # Define rating scale
    ratings_data = Dataset.load_from_df(filtered_ratings[['userId', 'movieId', 'rating']], reader) # Read data
    print()
    cross_validate(model, ratings_data, measures=['RMSE', 'MAE'], cv=5, verbose=True) # Cross validate with 5 splits to calculate RMSE
    print()

    print("What is your user ID?")
    print()
    user = int(input()) # Get user ID
    user_row = user-1 # Rows begin at 0 so subtract 1 for the correct row
    user_predictions = predicted_ratings.iloc[user_row].sort_values(ascending=False) # Get predictions of specific user
    user_predictions.rename(columns={"Predictions":user_row}) # Rename column
    user_ratings = filtered_ratings[filtered_ratings.userId == (user)] # Get user's existing ratings
    all_user_ratings = pd.merge(user_ratings, movies, how = 'left',left_on = 'movieId', right_on = 'movieId').sort_values(['rating'], ascending=False) # Merge with movies
    all_user_ratings_2 = pd.merge(user_ratings, movies, how = 'left',left_on = 'movieId', right_on = 'movieId').sort_values(['rating'], ascending=False)

    # Print user rated films
    print('User {0}: You have rated {1} films'.format(user, all_user_ratings.shape[0]))
    print('------------------------------------')
    all_user_ratings.set_index('title', inplace=True) # Set index to title
    all_user_ratings.rename_axis(None, inplace=True)
    all_user_ratings.drop(columns=['userId', 'movieId', 'genres'], inplace = True) # Remove IDs
    print(all_user_ratings.head(15)) # Print rated films
    print()

    # Ask user for number of recommendations
    print("Enter the number of recommendations you would like to receive")
    prediction_input = input() # Store input
    print()
    if prediction_input.isdigit():
        recommendation_no = int(prediction_input)

        # Compile predictions
        print('{0} recommendations with highest predicted ratings for user {1}'.format(recommendation_no, user))
        print()
        top_user_recommendations = (movies[~movies['movieId'].isin(all_user_ratings_2['movieId'])].merge(pd.DataFrame(user_predictions), how = 'left', left_on = 'movieId', right_on = 'movieId')) # Remove films already rated
        top_user_recommendations.set_index('title', inplace=True) # Set index to title
        top_user_recommendations.rename_axis(None, inplace=True)
        top_user_recommendations.drop(columns=['movieId', 'genres'], inplace = True) # Drop ID
        top_user_recommendations.sort_values(by=[user_row], ascending = False, inplace= True) # Sort by highest predictions
        top_user_recommendations.rename(columns={user_row:'Predictions'}, inplace=True) # Rename columns

        print(top_user_recommendations.head(recommendation_no)) # Print predictions
        print()
    

# User input required for content, collab and hybrid model
if user_option < 4:
    print("How many films would you like to enter? Between 1 and 3 are recommended.")
    input_no = int(input()) # Number of films
    print()
    print("Enter the film(s) you would like to find similar recommendations for: ")
    if input_no == 1:
        # Take 1 film as input
        title = input(" 1) ")
        print()
        print('Recommendations based on {0}'.format(title))
    elif input_no == 2:
        # Take 2 films as input
        title = input("1) ")
        title2 = input("2) ")
        print()
        print('Recommendations based on {0} and {1}'.format(title, title2))
    elif input_no == 3:
        # Take 3 films as input
        title = input(" 1) ")
        title2 = input(" 2) ")
        title3 = input(" 3) ")
        print()
        print('Recommendations based on {0}, {1} and {2}'.format(title, title2, title3))
    print()

    # Decide which method to call
    if user_option == 1:
        # Content-based recommendations
        content_recommendations = content_based_filtering()[0]
        # Drop inputted films from list of recommendations
        content_recommendations.drop(title, inplace=True)
        if input_no > 1:
            content_recommendations.drop(title2, inplace=True)
            if input_no > 2:
                content_recommendations.drop(title3, inplace=True)
        print(content_recommendations.head(10)) # Print 10 Recs

    elif user_option == 2:
        # Collaborative recommendations
        collab_recommendations = collaborative_filtering()[0]
        # Drop inputted films from list of recommendations
        collab_recommendations.drop(title, inplace=True)
        if input_no > 1:
            collab_recommendations.drop(title2, inplace=True)
            if input_no > 2:
                collab_recommendations.drop(title3, inplace=True)
        print(collab_recommendations.head(10)) # Print 10 Recs

    elif user_option == 3:
        # Hybrid recommendations
        hybrid_recommendations = hybrid_filtering()
        # Drop inputted films from list of recommendations
        hybrid_recommendations.drop(title, inplace=True)
        if input_no > 1:
            hybrid_recommendations.drop(title2, inplace=True)
            if input_no > 2:
                hybrid_recommendations.drop(title3, inplace=True)
        print(hybrid_recommendations.head(10)) # Print 10 Recs

if user_option == 4 :
    # Call collaborative prediction model
    predict_user_ratings()