import streamlit as st
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Knowledge Base",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Knowledge Base Semantic Search")

st.write("Ask any AI, Machine Learning, Python or Data Science question.")

# -------------------------------
# FAQ Dataset
# -------------------------------

faq = {

"Question":[

"What is Artificial Intelligence?",

"What is Machine Learning?",

"What is Deep Learning?",

"What is Data Science?",

"What is Python?",

"What is SQL?",

"What is a Large Language Model?",

"What is Prompt Engineering?",

"What is an AI Agent?",

"What is Semantic Search?"

],

"Answer":[

"Artificial Intelligence enables machines to perform tasks that normally require human intelligence.",

"Machine Learning enables computers to learn patterns from data without explicit programming.",

"Deep Learning is a subset of Machine Learning based on neural networks.",

"Data Science extracts useful insights from structured and unstructured data.",

"Python is a programming language widely used in AI and Machine Learning.",

"SQL is used to store, retrieve, and manage data in relational databases.",

"A Large Language Model is trained on huge amounts of text to understand and generate language.",

"Prompt Engineering is writing effective prompts to get better responses from AI.",

"An AI Agent can reason, plan, use tools, and complete tasks autonomously.",

"Semantic Search finds information based on meaning instead of exact keyword matching."

]

}

df = pd.DataFrame(faq)

# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------
# Generate Embeddings
# -------------------------------

embeddings = model.encode(df["Question"].tolist())

# -------------------------------
# User Input
# -------------------------------

user_query = st.text_input("Enter your question")

# -------------------------------
# Search
# -------------------------------

if st.button("Search"):

    if user_query.strip() == "":
        st.warning("Please enter a question.")

    else:

        query_embedding = model.encode([user_query])

        similarity = cosine_similarity(
            query_embedding,
            embeddings
        )[0]

        best_index = similarity.argmax()

        st.success("Best Match Found")

        st.subheader("Question")

        st.write(df.iloc[best_index]["Question"])

        st.subheader("Answer")

        st.write(df.iloc[best_index]["Answer"])

        st.metric(
            "Similarity Score",
            f"{similarity[best_index]*100:.2f}%"
        )
