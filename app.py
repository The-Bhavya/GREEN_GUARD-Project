from flask import Flask, render_template, request, redirect, url_for,session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime


app = Flask(__name__)

# Load trained model (saved from the notebook)
MODEL_PATH = Path('analysis') / 'dc_power_model_daylight.joblib'
model = joblib.load(MODEL_PATH)

# Features expected by the model
FEATURE_COLS = [
    'AMBIENT_TEMPERATURE',
    'MODULE_TEMPERATURE',
    'IRRADIATION',
    'hour',
    'day_of_week',
    'month',
    'PLANT_ID'
]

def build_features(payload: dict) -> pd.DataFrame:
    # Parse datetime and compute features
    dt_str = payload.get('datetime', '')
    if dt_str:
        try:
            dt = datetime.fromisoformat(dt_str)
        except ValueError:
            # Fallback for browsers that send without seconds
            dt = pd.to_datetime(dt_str)
    else:
        dt = datetime.now()

    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month

    ambient = float(payload.get('ambient_temperature', 0))
    module = float(payload.get('module_temperature', 0))
    irradiation = float(payload.get('irradiation', 0))
    plant_id = int(payload.get('plant_id', 1))

    row = {
        'AMBIENT_TEMPERATURE': ambient,
        'MODULE_TEMPERATURE': module,
        'IRRADIATION': irradiation,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'PLANT_ID': plant_id,
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    error = None
    explanation = None
    actions = []
    form = {
        'datetime': '',
        'ambient_temperature': '',
        'module_temperature': '',
        'irradiation': '',
        'plant_id': '1'
    }
    if request.method == 'POST':
        form.update(request.form)
        try:
            features = build_features(request.form)
            prediction_value = model.predict(features)[0]
            prediction = round(float(prediction_value), 2)

            # Build a simple explanation and suggested actions
            ambient = float(form.get('ambient_temperature') or 0)
            module = float(form.get('module_temperature') or 0)
            irradiation = float(form.get('irradiation') or 0)
            plant_id = int(form.get('plant_id') or 1)
            dt_text = form.get('datetime') or 'the selected time'

            parts = []
            parts.append(f"For Plant {plant_id} at {dt_text}, predicted DC Power is about {prediction} W.")

            # Irradiation context
            if irradiation <= 0.05:
                parts.append("Irradiation is near zero, so generation is expected to be minimal (night or heavy clouds).")
                actions.append("No action if it's night; otherwise check sky conditions and schedule accordingly.")
            elif irradiation < 0.3:
                parts.append("Low irradiation suggests overcast conditions, leading to reduced output.")
                actions.append("Consider rescheduling maintenance/cleaning for clearer periods.")
            elif irradiation >= 0.7:
                parts.append("High irradiation indicates strong sunlight, supporting higher power output.")

            # Temperature delta effect
            temp_delta = module - ambient
            if temp_delta > 20:
                parts.append("Module temperature is much higher than ambient, which can decrease panel efficiency.")
                actions.append("Improve airflow/ventilation or check for soiling causing heat buildup.")
            elif temp_delta < 5 and irradiation >= 0.5:
                parts.append("Module temperature is close to ambient under good sun, indicating healthy thermal behavior.")

            # Sanity check vs irradiation
            if irradiation >= 0.6 and prediction < 200:
                parts.append("Despite good sun, predicted power is low. This may indicate soiling, shading, or inverter limitation.")
                actions.append("Inspect panels for dirt/shading; verify inverter status and strings.")

            # Finalize explanation
            explanation = " ".join(parts)
        except Exception as exc:
            error = str(exc)
    return render_template('predict.html', prediction=prediction, explanation=explanation, actions=actions, error=error, form=form)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/base')
def base():
    return render_template('base.html')

app.run(debug=True)