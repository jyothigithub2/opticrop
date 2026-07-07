from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
app = Flask(__name__)
print("Current Folder:", os.getcwd())
print("Template Folder:", app.template_folder)
# Load Model
model = joblib.load("model/crop_model.pkl")
# Home Page
@app.route('/')
def home():
    return render_template("home.html")
# About Page
@app.route('/about')
def about():
    return render_template("about.html")
# Find Your Crop Page
@app.route('/findyourcrop')
def findyourcrop():
    return render_template("findyourcrop.html")
# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        features = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=[
                'N',
                'P',
                'K',
                'temperature',
                'humidity',
                'ph',
                'rainfall'
            ]
        )
        prediction = model.predict(features)
        return render_template(
            "findyourcrop.html",
            prediction_text=f"🌱 Recommended Crop: {prediction[0]}"
        )
    except Exception as e:
        return render_template(
            "findyourcrop.html",
            prediction_text=f"Error: {str(e)}"
        )
if __name__ == "__main__":
    app.run(debug=True)