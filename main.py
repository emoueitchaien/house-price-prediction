# pip install flask, scikit-learn, pandas, pickle-mixin

from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

data = pd.read_csv("final_cleaned_data.csv")
pipe = pickle.load(open('linear_model.pkl', 'rb'))


@app.route('/')
def index():  # put application's code here
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = request.form.get('total_sqft')

    # print(location, bhk, bath, sqft)

    inputs = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = str(round(pipe.predict(inputs)[0] * 1e5, 2))


    return prediction


if __name__ == '__main__':
    app.run()
