from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your_secret_key'
model = joblib.load('final_model.pkl')
scalar = joblib.load('scaler.pkl')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Define the User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Create the users table
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Page 1: Login
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('choose_option'))
        else:
            flash('Invalid login credentials.')
    return render_template('login.html')

# Register route for new users
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists.')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

# Page 2: Form or CSV Upload Option
@app.route('/choose_option', methods=['GET', 'POST'])
@login_required
def choose_option():
    if request.method == 'POST':
        option = request.form.get('option')
        if option == 'form':
            return redirect(url_for('form_input'))
        elif option == 'csv':
            return redirect(url_for('csv_upload'))
    return render_template('choose_option.html')

# Page 3: Form Input Page
@app.route('/form', methods=['GET', 'POST'])
@login_required
def form_input():
    if request.method == 'POST':
        form_data = {
            'Gender': request.form['Gender'],
            'Married': request.form['Married'],
            'Dependents': request.form['Dependents'],
            'Education': request.form['Education'],
            'Self_Employed': request.form['Self_Employed'],
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': float(request.form['Credit_History']),
            'Property_Area': request.form['Property_Area']
        }
        df = pd.DataFrame([form_data])
        df = preprocess(df)
        prediction = model.predict(df)[0]
        prediction_result = 'Eligible' if prediction == 'Y' else 'Not Eligible'
        df_original = pd.DataFrame([form_data])
        img = visualize_data(df_original)
        img_path = os.path.join('static', 'visualizations', 'form_visualization.png')
        with open(img_path, 'wb') as f:
            f.write(img.getbuffer())
        return render_template('results.html', prediction=prediction_result,
                               image_url=url_for('static', filename='visualizations/form_visualization.png'))
    return render_template('form.html')

# CSV Upload Page
@app.route('/csv_upload', methods=['GET', 'POST'])
@login_required
def csv_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        if file:
            filepath = os.path.join('uploaded_files', file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            df_processed = preprocess(df)
            df['Loan_Status'] = model.predict(df_processed)
            output_path = os.path.join('uploaded_files', 'predicted_' + file.filename)
            df.to_csv(output_path, index=False)
            img_url = save_visualizations(df)
            return render_template('csv_results.html', image_url=url_for('static', filename=img_url),
                                   file_name='predicted_' + file.filename)
    return render_template('csv_upload.html')

# Preprocessing function
def preprocess(df):
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df = df.dropna()
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['LoanAmountLog'] = np.log1p(df['LoanAmount'])
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'],
                        drop_first=True)
    missing_cols = set(scalar.feature_names_in_) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[scalar.feature_names_in_]
    df = df.fillna(0)
    df = scalar.transform(df)
    return df

# Visualization function
def visualize_data(df):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if 'Loan_Status' in df.columns:
        sns.countplot(data=df, x='Loan_Status', ax=axes[0])
        axes[0].set_title('Loan Status Count')
    if 'ApplicantIncome' in df.columns:
        sns.histplot(df['ApplicantIncome'], kde=True, ax=axes[1])
        axes[1].set_title('Applicant Income Distribution')
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return img

def save_visualizations(df):
    img = visualize_data(df)
    visualization_dir = 'static/visualizations'
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
    img_path = os.path.join(visualization_dir, 'visualization.png')
    with open(img_path, 'wb') as f:
        f.write(img.getbuffer())
    return 'visualizations/visualization.png'

# File download route
@app.route('/uploaded_files/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join('uploaded_files', filename))

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Run the app
if __name__ == '__main__':
    if not os.path.exists('uploaded_files'):
        os.makedirs('uploaded_files')
    app.run(debug=True)
