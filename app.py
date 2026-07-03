import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Re-initialize the model and data if not already globally available
# (In a real app.py, these would typically be defined directly or loaded)
# For this example, assuming 'model', 'df', 'embeddings' are accessible if run in Colab context

# In a standalone app.py, you would define these. For instance:
# faq_data = {
#     "Question": [...],
#     "Answer": [...]
# }
# df = pd.DataFrame(faq_data)
# model = SentenceTransformer("all-MiniLM-L6-v2")
# questions = df["Question"].tolist()
# embeddings = model.encode(questions)


def semantic_search_app():
    st.title("FAQ Semantic Search")

    user_query = st.text_input("Ask your question:")

    if user_query:
        # Ensure 'model', 'embeddings', 'df' are defined before this point in a real app.py
        query_embedding = model.encode([user_query])
        similarity = cosine_similarity(query_embedding, embeddings)[0]
        best_index = similarity.argmax()

        st.subheader("Closest Question:")
        st.write(df.iloc[best_index]["Question"])

        st.subheader("Answer:")
        st.write(df.iloc[best_index]["Answer"])

        st.subheader("Similarity Score:")
        st.write(f"{round(similarity[best_index], 3)}")

# Make sure to call the app function at the end
semantic_search_app()
