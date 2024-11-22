from flask import Flask, request, jsonify, render_template
from chatbot import get_relevant_context
from chatbot import generate_response_with_gemini
from textblob import TextBlob
import re

SYSTEM_PROMPT="You are a helpful assistant for a Holistic Health and Wellness center website. Respond to the user's message based on the following rules:\n\n1. **If the user's message only contains a greeting** (e.g., 'hi,' 'hello,' 'good morning'), respond with a simple greeting in return.\n2. **If the message contains both a greeting and a question**, start with a greeting and then provide a response to the question using the context provided.\n3. **If the message contains a farewell or expresses gratitude** (e.g., 'thanks,' 'goodbye,' 'see you'), respond with a polite closing message.\n4. **If the message is a question that relates to the context**, respond directly to the question, using information from the context below, without adding any greeting.\n5. **If the question is unrelated to the context**, respond with: 'I am sorry, I don't have the answer for that question.'\n\nPlease follow these rules carefully and ensure each response is under 60 words."


system_prompt = SYSTEM_PROMPT

app = Flask(__name__)

def preprocess_query(query):
    query = re.sub(r'^(what|how|why|who)\s+is\s+', '', query.strip(), flags=re.IGNORECASE)
    return query

def correct_spelling(message):
    blob = TextBlob(message)
    corrected = blob.correct()
    return str(corrected)

def chatbot_response(query):
    message = correct_spelling(query)
    message = preprocess_query(query)
    context = get_relevant_context(message)
    user_prompt = (
        "Context:\n"
        f"{context}\n\n"
    
        "User Message:\n"
        f"{message}\n\n"
    )
    prompt = f"{system_prompt}\n\n{user_prompt}"
    response = generate_response_with_gemini(prompt=prompt)
    return response

@app.route("/")
def index():
    return render_template("predict.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": "No message received"}), 400

    # Get the response from the model
    response = chatbot_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True) 
