import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

import numpy as np
# Load your data (e.g. movie titles and descriptions)

import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

df = pd.read_csv('movies.csv')

# Load a pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

import pandas as pd

chunksize = 100000  # adjust this value depending on your dataset and memory capacity
chunks = []

for chunk in pd.read_csv('movies.csv', chunksize=chunksize):
    # process each chunk here
    print(f'Processing {chunk.shape[0]} rows')
    embeddings = model.encode(chunk['primaryTitle'].tolist())
    chunks.append(embeddings)

# concatenate all chunks into a single array
embeddings = np.concatenate(chunks)


# Build a FAISS index for efficient similarity search
import nmslib

# Create a new index
index = nmslib.init(method='hnsw', space='cosinesimil')

# Add the embeddings to the index
index.addDataPointBatch(embeddings)

# Create the index
index.createIndex(print_progress=True)

def get_recommendations(query):
    query_embedding = model.encode([query])[0]
    indices, _ = index.knnQuery(query_embedding, k=5)  # find the 5 nearest neighbors
    recommendations = df['primaryTitle'].iloc[indices]
    return recommendations

# Streamlit interface
st.title('Movie Recommendation Engine')
user_input = st.text_input("Enter your movie preferences:")
if user_input:
    recommendations = recommend(user_input)
    for i in range(len(recommendations)):
        st.write(f"Recommendation {i+1}: {recommendations.iloc[i]['title']}")
