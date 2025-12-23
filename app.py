from flask import Flask, render_template, request, redirect, url_for,session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import joblib
import re
from pathlib import Path
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import json


# Load environment variables from .env if present
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS', 'False') == 'True'
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
# Use setattr to avoid static type-checker errors when assigning login_view
setattr(login_manager, 'login_view', 'login')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin, db.Model):
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
@login_required
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
    if request.method == 'POST':
        email_or_username = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        # Backend validation
        if not email_or_username:
            flash('Email or username is required', 'danger')
            return render_template('login.html')
        
        # Stricter email validation: only one dot in domain (e.g., example.com)
        if '@' in email_or_username:
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email_or_username):
                flash('Please enter a valid email address', 'danger')
                return render_template('login.html')
            elif len(email_or_username) > 150:
                flash('Email is too long', 'danger')
                return render_template('login.html')
        
        if not password:
            flash('Password is required', 'danger')
            return render_template('login.html')
        
        user = User.query.filter_by(email=email_or_username).first() or \
                User.query.filter_by(username=email_or_username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    
    return render_template('login.html')

# Signup Route (use same login.html)
@app.route('/signup', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        # Backend validation
        errors = []
        
        # Username validation
        if not username:
            errors.append('Username is required')
        elif len(username) < 3:
            errors.append('Username must be at least 3 characters long')
        elif len(username) > 20:
            errors.append('Username must be less than 20 characters')
        elif not username.replace('_', '').isalnum():
            errors.append('Username can only contain letters, numbers, and underscores')
        
        # Email validation
        if not email:
            errors.append('Email is required')
        else:
            # Stricter email validation: only one dot in domain (e.g., example.com)
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                errors.append('Please enter a valid email address')
            elif len(email) > 150:
                errors.append('Email is too long')
        
        # Password validation
        if not password:
            errors.append('Password is required')
        elif len(password) < 6:
            errors.append('Password must be at least 6 characters long')
        elif len(password) > 50:
            errors.append('Password must be less than 50 characters')
        elif not any(c.isupper() for c in password):
            errors.append('Password must contain at least one uppercase letter')
        elif not any(c.islower() for c in password):
            errors.append('Password must contain at least one lowercase letter')
        elif not any(c.isdigit() for c in password):
            errors.append('Password must contain at least one number')
        
        # Check for errors
        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('login.html')
        
        # Check if user already exists
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        if existing_user:
            if existing_user.username == username:
                flash('Username already exists', 'danger')
            else:
                flash('Email already exists', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            # Instantiate without kwargs and assign attributes to avoid constructor binding issues
            new_user = User()
            new_user.username = username
            new_user.email = email
            new_user.password = hashed_password
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('login.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/base')
def base():
    return render_template('base.html')

# Dashboard Route
@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Load and prepare data
        plant1_gen = pd.read_csv('Plant_1_Generation_Data.csv')
        plant1_weather = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
        plant2_gen = pd.read_csv('Plant_2_Generation_Data.csv')
        plant2_weather = pd.read_csv('Plant_2_Weather_Sensor_Data.csv')

        # Convert DATE_TIME columns
        plant1_gen['DATE_TIME'] = pd.to_datetime(plant1_gen['DATE_TIME'])
        plant1_weather['DATE_TIME'] = pd.to_datetime(plant1_weather['DATE_TIME'])
        plant2_gen['DATE_TIME'] = pd.to_datetime(plant2_gen['DATE_TIME'])
        plant2_weather['DATE_TIME'] = pd.to_datetime(plant2_weather['DATE_TIME'])

        # Merge data
        plant1_data = pd.merge(plant1_gen, plant1_weather, on='DATE_TIME', how='inner')
        plant2_data = pd.merge(plant2_gen, plant2_weather, on='DATE_TIME', how='inner')
        plant1_data['PLANT_ID'] = 1
        plant2_data['PLANT_ID'] = 2
        combined_data = pd.concat([plant1_data, plant2_data], ignore_index=True)

        # Extract time features
        combined_data['HOUR'] = combined_data['DATE_TIME'].dt.hour
        combined_data['DAY_OF_WEEK'] = combined_data['DATE_TIME'].dt.day_name()
        combined_data['MONTH'] = combined_data['DATE_TIME'].dt.month_name()
        combined_data['DATE'] = combined_data['DATE_TIME'].dt.date

        # Calculate KPIs
        total_generation = combined_data['DC_POWER'].sum() / 1000000
        avg_daily_generation = combined_data.groupby('DATE')['DC_POWER'].sum().mean() / 1000
        max_daily_generation = combined_data.groupby('DATE')['DC_POWER'].sum().max() / 1000
        avg_irradiation = combined_data['IRRADIATION'].mean()
        avg_ambient_temp = combined_data['AMBIENT_TEMPERATURE'].mean()
        avg_module_temp = combined_data['MODULE_TEMPERATURE'].mean()

        # Create graphs
        graphs = create_dashboard_graphs(combined_data)
        
        # Prepare insights
        insights = calculate_insights(combined_data)

        return render_template('dashboard.html', 
                            graphs=graphs,
                            insights=insights,
                            kpis={
                                'total_generation': total_generation,
                                'avg_daily_generation': avg_daily_generation,
                                'max_daily_generation': max_daily_generation,
                                'avg_irradiation': avg_irradiation,
                                'avg_ambient_temp': avg_ambient_temp,
                                'avg_module_temp': avg_module_temp
                            })
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'danger')
        return redirect(url_for('home'))

def create_dashboard_graphs(data):
    """Create all dashboard graphs and return as JSON"""
    graphs = {}
    
    # 1. Daily Generation Trend
    daily_generation = data.groupby(['DATE', 'PLANT_ID'])['DC_POWER'].sum().reset_index()
    daily_generation['DC_POWER_MWh'] = daily_generation['DC_POWER'] / 1000000
    
    fig_daily = px.line(
        daily_generation, 
        x='DATE', 
        y='DC_POWER_MWh', 
        color='PLANT_ID',
        title='Daily Power Generation Trend',
        color_discrete_sequence=['#2E8B57', '#32CD32']
    )
    fig_daily.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graphs['daily_trend'] = pio.to_html(fig_daily, include_plotlyjs=False, full_html=False)
    
    # 2. Hourly Generation Pattern
    hourly_avg = data.groupby(['HOUR', 'PLANT_ID'])['DC_POWER'].mean().reset_index()
    hourly_avg['DC_POWER_kW'] = hourly_avg['DC_POWER'] / 1000
    
    fig_hourly = px.bar(
        hourly_avg,
        x='HOUR',
        y='DC_POWER_kW',
        color='PLANT_ID',
        title='Average Hourly Power Generation Pattern',
        color_discrete_sequence=['#2E8B57', '#32CD32'],
        barmode='group'
    )
    fig_hourly.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graphs['hourly_pattern'] = pio.to_html(fig_hourly, include_plotlyjs=False, full_html=False)
    
    # 3. Weather Correlation Heatmap
    weather_corr = data[['DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].corr()
    fig_heatmap = px.imshow(
        weather_corr,
        text_auto=True,
        aspect="auto",
        title='Weather Factors Correlation with Power Generation',
        color_continuous_scale='RdYlGn'
    )
    fig_heatmap.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graphs['weather_correlation'] = pio.to_html(fig_heatmap, include_plotlyjs=False, full_html=False)
    
    # 4. Irradiation vs Power Scatter
    sample_data = data.sample(n=min(3000, len(data)))
    fig_scatter = px.scatter(
        sample_data,
        x='IRRADIATION',
        y='DC_POWER',
        color='PLANT_ID',
        title='Solar Irradiation vs Power Generation',
        color_discrete_sequence=['#2E8B57', '#32CD32'],
        opacity=0.6
    )
    fig_scatter.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graphs['irradiation_scatter'] = pio.to_html(fig_scatter, include_plotlyjs=False, full_html=False)
    
    # 5. Monthly Performance
    monthly_data = data.groupby(['MONTH', 'PLANT_ID'])['DC_POWER'].sum().reset_index()
    monthly_data['DC_POWER_MWh'] = monthly_data['DC_POWER'] / 1000000
    
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data['MONTH'] = pd.Categorical(monthly_data['MONTH'], categories=month_order, ordered=True)
    monthly_data = monthly_data.sort_values('MONTH')
    
    fig_monthly = px.bar(
        monthly_data,
        x='MONTH',
        y='DC_POWER_MWh',
        color='PLANT_ID',
        title='Monthly Power Generation Comparison',
        color_discrete_sequence=['#2E8B57', '#32CD32'],
        barmode='group'
    )
    fig_monthly.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graphs['monthly_performance'] = pio.to_html(fig_monthly, include_plotlyjs=False, full_html=False)
    
    # 6. Power Distribution Histogram
    fig_hist = px.histogram(
        data,
        x='DC_POWER',
        color='PLANT_ID',
        title='Power Generation Distribution',
        color_discrete_sequence=['#2E8B57', '#32CD32'],
        nbins=30,
        opacity=0.7
    )
    fig_hist.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graphs['power_distribution'] = pio.to_html(fig_hist, include_plotlyjs=False, full_html=False)
    
    return graphs

def calculate_insights(data):
    """Calculate key insights from the data"""
    insights = {}
    
    # Daily generation insights
    daily_gen = data.groupby('DATE')['DC_POWER'].sum()
    insights['avg_daily'] = daily_gen.mean() / 1000
    insights['max_daily'] = daily_gen.max() / 1000
    insights['min_daily'] = daily_gen.min() / 1000
    
    # Hourly insights
    hourly_avg = data.groupby('HOUR')['DC_POWER'].mean()
    peak_hour = hourly_avg.idxmax()
    insights['peak_hour'] = peak_hour
    insights['peak_power'] = hourly_avg.max() / 1000
    
    # Weather insights
    weather_corr = data[['DC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']].corr()
    insights['irradiation_corr'] = weather_corr.loc['IRRADIATION', 'DC_POWER']
    insights['temp_corr'] = weather_corr.loc['AMBIENT_TEMPERATURE', 'DC_POWER']
    
    # Plant comparison
    plant1_total = data[data['PLANT_ID'] == 1]['DC_POWER'].sum() / 1000000
    plant2_total = data[data['PLANT_ID'] == 2]['DC_POWER'].sum() / 1000000
    insights['plant1_total'] = plant1_total
    insights['plant2_total'] = plant2_total
    insights['better_plant'] = 'Plant 1' if plant1_total > plant2_total else 'Plant 2'
    
    # Monthly insights
    monthly_total = data.groupby('MONTH')['DC_POWER'].sum()
    insights['best_month'] = monthly_total.idxmax()
    insights['worst_month'] = monthly_total.idxmin()
    
    return insights

if __name__ == '__main__':
    if not os.path.exists('users.db'):
        with app.app_context():
            db.create_all()

    # Use PORT env variable for deployment (default 5000)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
