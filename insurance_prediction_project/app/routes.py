from flask import render_template, request, redirect, url_for
from app import app
import pickle
import numpy as np

# Load the trained model
with open('app/model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction_form', methods=['GET', 'POST'])
def prediction_form():
    if request.method == 'POST':
        # Get data from the form
        gender = request.form['gender']
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        marital_status = request.form['marital_status']
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Convert categorical variables to numerical values (if necessary)
        gender = 1 if gender == 'male' else 0
        marital_status = 1 if marital_status == 'married' else 0
        smoker = 1 if smoker == 'yes' else 0
        region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
        region = region_map[region]

        # Prepare the input for prediction
        input_data = np.array([[age, gender, bmi, marital_status, children, smoker, region]])

        # Make the prediction
        predicted_cost = model.predict(input_data)

        # Return the result
        return render_template('result.html', insurance_cost=predicted_cost[0])

    return render_template('prediction_form.html')

