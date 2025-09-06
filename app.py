from flask import Flask, render_template, request, redirect, url_for,session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from flask_bcrypt import bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

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

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'login' in request.form:
        email_or_username = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email_or_username).first() or \
                User.query.filter_by(username=email_or_username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    
    return render_template('login.html')

# Signup Route (use same login.html)
@app.route('/signup', methods=['GET', 'POST'])
def register():
    if request.method == 'POST' and 'signup' in request.form:
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        if existing_user:
            flash('Username or email already exists')
        else:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/base')
def base():
    return render_template('base.html')

if __name__ == '__main__':
    if not os.path.exists('users.db'):
        with app.app_context():
            db.create_all()
app.run(debug=True)
