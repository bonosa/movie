import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import scipy.spatial

# Load the BERT model. Various models trained on Natural Language Inference (NLI) https://www.sbert.net/docs/pretrained_models.html#nli-models
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('https://storage.googleapis.com/movves123/movies.csv')
    return df

df = load_data()

# We get all unique sentences
sentences = df['title'].unique()

# Compute embedding for both lists
embeddings = model.encode(sentences, convert_to_tensor=True)

# Compute cosine-similarities for each sentence with each other sentence
cosine_scores = scipy.spatial.distance.cdist(embeddings, embeddings, "cosine")

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(sentences))

st.title("Movie Recommender")
user_input = st.text_input("Enter a genre", "Action")

if st.button('Submit'):
    # Get the query embeddings
    queries = [user_input]
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    
    # For each query, we compute the cosine-similarity scores with every sentence in the corpus
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        st.write("\n\n======================\n\n")
        st.write("Query:", query)
        st.write("\nTop 5 most similar sentences in corpus:")

        for idx, distance in results[0:top_k]:
            st.write(sentences[idx].strip(), "(Score: %.4f)" % (1-distance))

        st.write("\n\n======================\n\n")

