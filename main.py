from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route("/")
def index():
    locations = sorted(data['Location'].unique())
    return render_template('index.html', locations=locations)

@app.route("/predict", methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    area = float(request.form.get('area'))

    input = pd.DataFrame([[location, bhk, area]],columns=['Location', 'No. of Bedrooms', 'Area'])
    prediction = pipe.predict(input)[0]

    return str(np.absolute(np.round(prediction, 2)))

if __name__ == "__main__":
    app.run(debug=True, port=5000)