from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load pipeline model
with open('pipeline.pkl','rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(final_input)[0]

    # Human-readable output
    if prediction == 1:
        output = "❌ The customer is likely to CHURN."
    else:
        output = "✅ The customer is NOT likely to churn."

    return render_template('result.html', prediction=output)

    # Create input data in same order as model expects
    input_data = pd.DataFrame([{
        'SeniorCitizen': int(data['SeniorCitizen']),
        'Partner': int(data['Partner']),
        'Dependents': int(data['Dependents']),
        'tenure': float(data['tenure']),
        'OnlineSecurity': int(data['OnlineSecurity']),
        'MonthlyCharges': float(data['MonthlyCharges']),
        'TotalCharges': float(data['TotalCharges']),
        'gender_Male': int(data['gender_Male']),
        'InternetService_Fiber optic': int(data['InternetService_Fiber optic']),
        'InternetService_No': int(data['InternetService_No']),
        'Contract_One year': int(data['Contract_One year']),
        'Contract_Two year': int(data['Contract_Two year']),
        'PaymentMethod_Credit card (automatic)': int(data['PaymentMethod_Credit card (automatic)']),
        'PaymentMethod_Electronic check': int(data['PaymentMethod_Electronic check']),
        'PaymentMethod_Mailed check': int(data['PaymentMethod_Mailed check']),
    }])

    # Predict
    pred = model.predict(input_data)[0]

    return render_template('result.html', prediction=pred)

if __name__ == '__main__':
    app.run(debug=True)