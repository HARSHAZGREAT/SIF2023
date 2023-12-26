from flask import Flask, render_template, request, jsonify
from gtts import gTTS
import os
import pyttsx3

app = Flask(__name__)

import numpy as np
import json
import re
import tensorflow as tf
import random
import spacy
nlp = spacy.load('en_core_web_sm')

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'youre', 'youve', 'youll', 'youd', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'shes', 'her', 'hers', 'herself', 'it', 'its', 'itself','they', 'them', 'their', 'theirs', 'themselves','what', 'which', 'who', 'whom', 'this', 'that', 'thatll', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about','against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'dont', 'should', 'shouldve', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'arent', 'couldn', 'couldnt', 'didn', 'didnt', 'doesn', 'doesnt', 'hadn', 'hadnt', 'hasn', 'hasnt', 'haven', 'havent', 'isn', 'isnt', 'ma', 'might', 'mightnt', 'must', 'mustnt', 'need', 'neednt', 'shan', 'shant', 'shouldn', 'shouldnt', 'was', 'wasnt', 'weren', 'werent', 'won', 'wont', 'wouldn', 'wouldnt', 'like', 'use']

with open('static/intent.json') as f:
    intents = json.load(f)

def preprocessing(line):
    line = re.sub(r'[^a-zA-z.?!\']', ' ', line)
    line = re.sub(r'[ ]+', ' ', line)
    return line

inputs, targets = [], []
classes = []
intent_doc = {}

for intent in intents['intents']:
    if intent['intent'] not in classes:
        classes.append(intent['intent'])
    if intent['intent'] not in intent_doc:
        intent_doc[intent['intent']] = []
        
    for text in intent['text']:
        inputs.append(preprocessing(text))
        targets.append(intent['intent'])
        
    for response in intent['responses']:
        intent_doc[intent['intent']].append(response)

def tokenize_data(input_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
    
    tokenizer.fit_on_texts(input_list)
    
    input_seq = tokenizer.texts_to_sequences(input_list)

    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='pre')
    
    return tokenizer, input_seq

tokenizer, input_tensor = tokenize_data(inputs)

def create_categorical_target(targets):
    word={}
    categorical_target=[]
    counter=0
    for trg in targets:
        if trg not in word:
            word[trg]=counter
            counter+=1
        categorical_target.append(word[trg])
    
    categorical_tensor = tf.keras.utils.to_categorical(categorical_target, num_classes=len(word), dtype='int32')
    return categorical_tensor, dict((v,k) for k, v in word.items())

target_tensor, trg_index_word = create_categorical_target(targets)

print('input shape: {} and output shape: {}'.format(input_tensor.shape, target_tensor.shape))

epochs=50
vocab_size=len(tokenizer.word_index) + 1
embed_dim=512
units=128
target_length=target_tensor.shape[1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, dropout=0.2)),
    tf.keras.layers.Dense(units, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(target_length, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(lr=1e-2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

# train the model
model.fit(input_tensor, target_tensor, epochs=epochs, callbacks=[early_stop])


def response(sentence):
    sent_seq = []
    doc = nlp(repr(sentence))

    for token in doc:
        if token.text in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token.text])

        else:
            sent_seq.append(tokenizer.word_index["<unk>"])

    sent_seq = tf.expand_dims(sent_seq, 0)
    pred = model(sent_seq)

    pred_class = np.argmax(pred.numpy(), axis=1)
    
    return random.choice(intent_doc[trg_index_word[pred_class[0]]]), trg_index_word[pred_class[0]]

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        # Get the input message from the POST request
        user_message = request.form["msg"]
        
        # Get the chatbot's response and its type
        bot_response, response_type = response(user_message)
        
        speak(bot_response)

        return jsonify({"response": bot_response, "type": response_type})
    except Exception as e:
        # Handle errors gracefully
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)