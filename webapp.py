# Importing the Required Libraries
from flask import Flask, render_template, request
import joblib
import numpy as np

# Creating an Instance of Flask Application
application = Flask(__name__)

# Loading Pre-Trained Models
lr_model = joblib.load('models/lr_model.pkl')  # Linear Regression Model
knn_model = joblib.load('models/knn_model.pkl')  # KNN Model
svr_model = joblib.load('models/svr_model.pkl')  # SVR Model

# Loading Metrics
mse_scores = joblib.load('models/mse.pkl')  # MSE Scores
rmse_scores = joblib.load('models/rmse.pkl')  # RMSE Scores
r2_scores = joblib.load('models/r2.pkl')  # R² Scores

# Route for Home Page
@application.route('/')
def home():
    return render_template('home.html')  # Renders the form for input

# Route for Predictions
@application.route('/predict', methods=['POST'])
def predict():
    # Getting Input Values from the Form
    cement = float(request.form['cement'])
    slag = float(request.form['slag'])
    flyash = float(request.form['flyash'])
    water = float(request.form['water'])
    superplasticizer = float(request.form['superplasticizer'])
    coarseaggregate = float(request.form['coarseaggregate'])
    fineaggregate = float(request.form['fineaggregate'])
    age = int(request.form['age'])

    # Combining All Features
    features = np.array([[cement, slag, flyash, water, superplasticizer, coarseaggregate, fineaggregate, age]])

    # Make Predictions Using All Models
    prediction_lr = lr_model.predict(features)[0]
    prediction_knn = knn_model.predict(features)[0]
    prediction_svr = svr_model.predict(features)[0]

    # Render Results Page
    return render_template(
        'results.html',
        prediction_lr=round(prediction_lr, 2),
        prediction_knn=round(prediction_knn, 2),
        prediction_svr=round(prediction_svr, 2),
        mse_lr=round(mse_scores["Linear Regression"], 2),
        rmse_lr=round(rmse_scores["Linear Regression"], 2),
        r2_lr=round(r2_scores["Linear Regression"], 2),
        mse_knn=round(mse_scores["KNN Regression"], 2),
        rmse_knn=round(rmse_scores["K-Nearest Neighbors Regression"], 2),
        r2_knn=round(r2_scores["K-Nearest Neighbors Regression"], 2),
        mse_svr=round(mse_scores["SVR"], 2),
        rmse_svr=round(rmse_scores["Support Vector Regression"], 2),
        r2_svr=round(r2_scores["Support Vector Regression"], 2)
    )

# Run Flask Application
if __name__ == '__main__':
    application.run(debug=True)