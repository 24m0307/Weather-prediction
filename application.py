from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

application = Flask(__name__)
app = application

# import model and encoder 
model = pickle.load(open('model/weather_model.pkl', 'rb'))
encoder = pickle.load(open('model/weather_encoder.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # later you will add prediction code here
        precipitation = float(request.form.get('precipitation'))
        temp_max = float(request.form.get('temp_max'))
        temp_min = float(request.form.get('temp_min'))
        wind = float(request.form.get('wind'))

        # 2. Prepare data for prediction
        data = np.array([[precipitation, temp_max, temp_min, wind]])

        # 3. Make prediction
        pred = model.predict(data)

        # 4. Decode if needed (only if you used encoder)
        prediction = encoder.inverse_transform(pred)[0]

        # 5. Show result on same page
        return render_template('home.html', result=prediction)
        pass
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)

