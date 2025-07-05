# movie_recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create custom movie dataset
movies = pd.DataFrame({
    'title': [
        'The Dark Knight', 'Batman Begins', 'Inception',
        'Interstellar', 'The Prestige', 'Tenet', 'Avengers: Endgame',
        'Iron Man', 'Captain America: Civil War', 'Guardians of the Galaxy'
    ],
    'description': [
        'Batman raises the stakes in his war on crime.',
        'Bruce Wayne begins his fight to free crime-ridden Gotham City.',
        'A thief who steals corporate secrets through dream-sharing.',
        'Explorers travel through a wormhole in space.',
        'Two magicians engage in a battle to create the ultimate illusion.',
        'A secret agent embarks on a time-bending mission.',
        'After the devastating events of Infinity War, heroes assemble.',
        'A billionaire industrialist builds a high-tech suit.',
        'Political interference fractures the Avengers.',
        'A group of intergalactic criminals must pull together.'
    ]
})

# Step 2: Convert text descriptions into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['description'])

# Step 3: Compute cosine similarity between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 4: Recommendation function
def recommend(title, cosine_sim=cosine_sim):
    index = movies[movies['title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_movies = similarity_scores[1:6]  # top 5 excluding the movie itself
    
    print(f"\nRecommendations for: {title}")
    for i, score in top_movies:
        print(f"  - {movies.iloc[i]['title']}")

# Step 5: Test the recommender system
recommend("Inception")
