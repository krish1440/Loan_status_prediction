from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model pipeline
model = joblib.load('best_catboost_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        gender = request.form['Gender']
        married = request.form['Married']
        dependents = request.form['Dependents']
        education = request.form['Education']
        self_employed = request.form['Self_Employed']
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = request.form['Property_Area']

        # Create a DataFrame with the input data to make a prediction
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })

        # Predict using the loaded pipeline
        prediction = model.predict(input_data)

        # Determine result and CSS class
        if prediction[0] == 1:
            result = "Approved"
            prediction_class = "approved"
        else:
            result = "Not Approved"
            prediction_class = "rejected"

        # Pass prediction text and class to the template
        return render_template('index.html', prediction_text=f"Loan Status: {result}", prediction_class=prediction_class)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", prediction_class="rejected")


if __name__ == "__main__":
    app.run(debug=True)
