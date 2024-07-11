import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import random
import numpy as np
import pandas as pd
import json
import joblib

from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, jsonify

# Load chatbot data
with open("lstm/chatbot.json", 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])

# Create a dictionary to store the data
dic = {"tag": [], "patterns": [], "responses": []}
for i in range(len(df)):
    ptrns = df.at[i, 'patterns']
    rspns = df.at[i, 'responses']
    tag = df.at[i, 'tag']

    for pattern in ptrns:
        dic['tag'].append(tag)
        dic['patterns'].append(pattern)
        dic['responses'].append(rspns)

# Create a new DataFrame from the dictionary
df = pd.DataFrame.from_dict(dic)

# Load the tokenizer, label encoder, and the trained model
with open('lstm/tokenizer_lstm.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('lstm/label_encoder_lstm.pkl', 'rb') as handle:
    lbl_enc = pickle.load(handle)

# Load the LSTM model
model = load_model('lstm/my_lstm_model.keras')

def preprocess_input(pattern, tokenizer, maxlen=18):
    if pattern is None:
        return None
    text = re.sub(r"[^a-zA-Z\']", ' ', str(pattern)).lower()
    text = text.split()
    text = " ".join(text)
    x_test = tokenizer.texts_to_sequences([text])
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
    return x_test

def predict_response(pattern, model, tokenizer, lbl_enc, df):
    x_test = preprocess_input(pattern, tokenizer)
    if x_test is None:
        return "I'm sorry, I couldn't understand that. Could you please rephrase?"
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax(axis=1)
    tag = lbl_enc.inverse_transform(y_pred)[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return random.choice(responses)

# Configure Flask
app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def get_response():
    try:
        userText = request.json.get('message')

        if not userText:
            return jsonify({"error": "No user text provided"}), 400

        response = predict_response(userText, model, tokenizer, lbl_enc, df)
        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error in get_response: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)