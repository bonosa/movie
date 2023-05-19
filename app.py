import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(file)
    return df

@st.cache(allow_output_mutation=True)
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache(allow_output_mutation=True)
def create_index(df, model):
    embeddings = model.encode(df['title'].tolist())
    f = len(embeddings[0])
    t = AnnoyIndex(f, 'angular')
    for i, v in enumerate(embeddings):
        t.add_item(i, v)
    t.build(10)
    return t

def get_recommendations(query, df, model, index):
    query_embedding = model.encode([query])[0]
    indices = index.get_nns_by_vector(query_embedding, 5)
    recommendations = df['title'].iloc[indices]
    return recommendations

st.title('Movie Recommendation App')

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    model = load_model()
    index = create_index(df, model)
    query = st.text_input('Enter a movie title:')
    if query:
        recommendations = get_recommendations(query, df, model, index)
        st.write(recommendations)
