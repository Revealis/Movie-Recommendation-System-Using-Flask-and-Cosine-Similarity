import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def download_and_extract_data():
    import requests, zipfile, io
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall()

download_and_extract_data()

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

average_rating_per_movie = ratings.groupby('movieId')['rating'].mean()

movies['avg_rating'] = movies['movieId'].map(average_rating_per_movie)

user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def get_similar_users(user_id, num_users=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[:num_users].index
    return similar_users

def recommend_movies(user_id, num_recommendations=5):
    similar_users = get_similar_users(user_id)
    similar_users_ratings = user_movie_matrix.loc[similar_users]
    similar_users_mean_ratings = similar_users_ratings.mean(axis=0)
    user_ratings = user_movie_matrix.loc[user_id]
    recommendations = similar_users_mean_ratings[user_ratings == 0]
    recommendations = recommendations.sort_values(ascending=False)[:num_recommendations]
    return recommendations.index, recommendations.values

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    recommended_movies = pd.DataFrame() 
    user_id = None
    if request.method == 'POST':
        try:
            user_id = int(request.form['user_id'])
            if user_id not in user_movie_matrix.index:
                error_message = "No such user was found. Please enter a valid user ID."
            else:
                movie_ids, ratings = recommend_movies(user_id)
                recommended_movies = movies[movies['movieId'].isin(movie_ids)].copy()
                recommended_movies['avg_rating'] = recommended_movies['avg_rating'].round(2)
                recommended_movies = recommended_movies[['title', 'avg_rating']].sort_values(by='avg_rating', ascending=False)
        except ValueError:
            error_message = "Enter a valid user ID."
    return render_template('index.html', movies=recommended_movies.to_dict(orient='records'), user_id=user_id, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
