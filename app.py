import json
import pickle

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

## Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pk1', 'rb'))

# Define the debug variable
debug = True

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = np.array(list(data.values())).reshape(1, -1)
    new_data_scaled = scalar.transform(new_data)
    output = regmodel.predict(new_data_scaled)
    return jsonify(output.tolist())

if __name__ == "__main__":
    app.run(debug=debug)
