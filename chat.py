from flask import Flask, render_template, request, jsonify
import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

# Load intents data and model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()


@app.route('/')
def home():
    return render_template('home.html')  

@app.route('/agriculture')
def agriculture():
    return render_template('agriculture.html')

@app.route('/Education')
def Education():
    return render_template('Education.html')

@app.route('/SocialWelfare')
def SocialWelfare():
    return render_template('SocialWelfare.html')

@app.route('/Entrepreneur')
def Entrepreneur():
    return render_template('Entrepreneur.html')

@app.route('/Health')
def Health():
    return render_template('Health.html')

@app.route('/Ask Assistant')
def AskAssistant():
    return render_template('index.html')




@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
    else:
        response = "I do not understand..."

    return jsonify({'bot_response': response})

if __name__ == '__main__':
    app.run(debug=True)
