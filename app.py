# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:04:07 2024

@author: ikonz
"""

''' Outline the technical design for deploying the prediction model through an online endpoint.'''

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug = True, port=5000)
