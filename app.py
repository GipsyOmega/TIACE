import json
import pickle

from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

label_dict = {0:"Normal", 1:"S", 2:"V", 3:"F", 4:"Q"}
## Load the model
model = pickle.load(open("models/tiace_pickle.pkl",'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        csv_file = request.files['csvFile']
        if csv_file:
            df = pd.read_csv(csv_file, header=None)
            feature = df.iloc[:100,:186].values
            label = df[187]
            X = feature.reshape(len(feature), feature.shape[1],1)

            # Prediction
            pred = model.predict(X)
            pred = pred.argmax(axis = -1)
            labels = [label_dict[x] for x in pred]
            pred_str = "\n".join(labels)
            return pred_str

    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
   