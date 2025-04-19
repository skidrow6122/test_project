import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.file_loader import (load_csv)

def test_cos_sim():
    print('📌 Cosine similarity test')

    doc1 = np.array([0, 1, 1, 1])
    doc2 = np.array([1, 0, 1, 1])
    doc3 = np.array([2, 0, 2, 2])

    print("Cosine similarity example between doc1 and doc2: ", dot(doc1, doc2) / (norm(doc1) * norm(doc2)))
    print("Cosine similarity example between doc1 and doc3: ", dot(doc1, doc3) / (norm(doc1) * norm(doc3)))
    print("Cosine similarity example between doc2 and doc3: ", dot(doc2, doc3) / (norm(doc2) * norm(doc3)))


def test_sklearn_cos_sim():
    # data load
    data = load_csv('movies_metadata.csv')
    print(data.head(2))

    # extract 20000 row to dataframe
    data = data.head(20000)

    # null check
    print("null checking in 'overview' field :", data['overview'].isnull().sum())

    # replace nukll to na
    data['overview'] = data['overview'].fillna('')

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['overview'])
    print("Shape of tfidf matrix: ", tfidf_matrix.shape)

    # calculating cosine similarity between overview and overview
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Result of Cosine similarity: ", cosine_sim.shape)

    # making dictionary contains title and index
    title_to_index = dict(zip(data['title'], data.index))

    idx = title_to_index['Toy Story']
    # get all similar movies based on the result of consine similarity
    sim_scores = list(enumerate(cosine_sim[idx]))
    # sorting movies
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # get top 10
    sim_scores = sim_scores[1:11]
    # get to 10`s index list
    movie_index_list = [idx[0] for idx in sim_scores]
    # return titles
    print("Movie Title list: ", data['title'].iloc[movie_index_list])














