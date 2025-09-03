from flask import Flask,render_template, request, redirect, url_for, flash

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load # dump is used to save the model and load is used to load the model

app = Flask(__name__)

plant_1_gen = pd.read_csv('Plant_1_Generation_Data.csv')
plant_2_gen = pd.read_csv('Plant_2_Generation_Data.csv')
plant_1_weather = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
plant_2_weather = pd.read_csv('Plant_2_Weather_Sensor_Data.csv')

# Merge generation and weather data based on time
plant_1 = pd.merge(plant_1_gen, plant_1_weather, on='DATE_TIME', how='inner')
plant_2 = pd.merge(plant_2_gen, plant_2_weather, on='DATE_TIME', how='inner')

combined_data = pd.concat([plant_1, plant_2], ignore_index=True)

# load the saved model and preprocessing components
model = joblib.load('rb_cyber_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/login")
def login():
    return render_template('login.html')

if __name__ == "__main__":
    app.run(debug=True)