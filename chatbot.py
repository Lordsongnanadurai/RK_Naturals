from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import pinecone
import google.generativeai as genai

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

gemini_api_key="your-api-key"
pinecone_key="your-api-key"
pinecone_index="your-index-name"

pinecone_key = pinecone_key
gemini_key = gemini_api_key
pinecone_index = pinecone_index

pinecone = Pinecone(api_key=pinecone_key)
index_name = pinecone_index
index = pinecone.Index(index_name)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

genai.configure(api_key=gemini_key)

def get_relevant_context(query, top_k=3, min_similarity=0.7, max_context_length=500):
    query_vector = embedding_model.encode([query])[0].tolist()
    results = index.query(vector=[query_vector], top_k=top_k, include_metadata=True, include_values=True)
    relevant_texts = []
    for result in results['matches']:
        text = result['metadata']['text']
        if 'values' in result:
            text_vector = result['values']
        else:
            continue
        similarity_score = cosine_similarity([query_vector], [text_vector])[0][0]
        if similarity_score >= min_similarity:
            relevant_texts.append((similarity_score, text))
    relevant_texts = sorted(relevant_texts, key=lambda x: x[0], reverse=True)
    context = ""
    for _, text in relevant_texts:
        if len(context) + len(text) > max_context_length:
            break
        context += " " + text
    return context.strip()

def generate_response_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    words = response.text.split()
    response_text = " ".join(words)
    return response_text