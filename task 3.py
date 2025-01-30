import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Sample Movie Dataset
data = {
    'MovieID': [1, 2, 3, 4, 5],
    'Title': ['Inception', 'Interstellar', 'The Dark Knight', 'Pulp Fiction', 'Fight Club'],
    'Genre': ['Sci-Fi Thriller', 'Sci-Fi Drama', 'Action Crime', 'Crime Drama', 'Drama Thriller']
}

df = pd.DataFrame(data)

# Step 2: Vectorize Genres using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Genre'])

# Step 3: Compute cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 4: Recommendation Function
def recommend_movies(title, cosine_sim=cosine_sim, df=df):
    # Get the index of the movie that matches the title
    idx = df[df['Title'] == title].index[0]
    
    # Get the pairwise similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top 3 most similar movies (excluding itself)
    sim_scores = sim_scores[1:4]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 3 similar movies
    return df['Title'].iloc[movie_indices]

# Step 5: Example: Recommend movies similar to 'Inception'
recommended_movies = recommend_movies('Inception')
print("Recommended movies for 'Inception':")
for movie in recommended_movies:
    print(movie)