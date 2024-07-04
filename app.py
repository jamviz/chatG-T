import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import random
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import LabelEncoder
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
sess = tf.compat.v1.Session(config=config)

# Importando archivos del modelo
import pickle
uploaded = "lstm/chatbot.json"

with open("lstm/chatbot.json", 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])

# Almacenar datos que serán convertidos a DataFrame
dic = {"tag": [], "patterns": [], "responses": []}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]

    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)

# Crear un nuevo DataFrame a partir del diccionario
df = pd.DataFrame.from_dict(dic)

# Mostrar el DataFrame para verificar
df.head()

# Obtener las etiquetas únicas
df['tag'].unique()
# Load the tokenizer, label encoder, and trained model
with open('lstm/tokenizer_lstm.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('lstm/label_encoder_lstm.pkl', 'rb') as handle:
    lbl_enc = pickle.load(handle)

# Replace 'my_model.keras' with the actual filename of your saved model
model = load_model('lstm/my_lstm_model.keras')  # (LSTM, BiLSTM, GRU, or BiGRU)


def input_user(pattern, tokenizer, maxlen=18):
    if pattern is None:
        return None
    text = re.sub(r"[^a-zA-Z\']", ' ', str(pattern)).lower()
    text = text.split()
    text = " ".join(text)
    x_test = tokenizer.texts_to_sequences([text])
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
    return x_test

def predict_response(pattern, model, tokenizer, lbl_enc, df):
    x_test = input_user(pattern, tokenizer)
    if x_test is None:
        return "I'm sorry, I couldn't understand that. Could you please rephrase?"
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax(axis=1)
    tag = lbl_enc.inverse_transform(y_pred)[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return random.choice(responses)


######################################FLASK########################################
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
app.static_folder = 'static'
@app.route('/')
def home():

    return render_template('index.html')

model = None  # Variable global para almacenar el modelo actual

@app.route("/get", methods=["GET", "POST"])
def saludar():
    global model  # Usar la variable global

    if request.method == "POST":
        userText = request.form.get('msg')
        model_name = request.form.get('model')
    else:  # GET method
        userText = request.args.get('msg')
        model_name = request.args.get('model')

    if not userText:
        return jsonify({"error": "No user text provided"}), 400

    if model_name:
        # Cargar el modelo correspondiente solo si se seleccionó uno nuevo
        model_path = f'lstm/my_{model_name}_model.keras'
        try:
            model = load_model(model_path)
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    if model is None:
        # Si no hay un modelo cargado, cargar el modelo LSTM por defecto
        model_path = 'lstm/my_lstm_model.keras'
        try:
            model = load_model(model_path)
        except Exception as e:
            return jsonify({"error": f"Failed to load default model: {str(e)}"}), 500

    try:
        response = predict_response(userText, model, tokenizer, lbl_enc, df)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500

@app.route("/clear_history", methods=["POST"])
def clear_history():
    # Aquí puedes implementar la lógica para borrar el historial de mensajes
    return "Historial borrado"

@app.route("/faq", methods=["GET"])
def get_faq():
    # Implement your FAQ logic here
    return jsonify({"faq": "Your FAQ content here"})

    
if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True)
