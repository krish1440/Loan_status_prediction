from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('best_catboost_model.pkl')

@app.route('/')
def home():
    return render_template('documentation.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        gender = input_data['Gender']
        married = input_data['Married']
        dependents = input_data['Dependents']
        education = input_data['Education']
        self_employed = input_data['Self_Employed']
        applicant_income = float(input_data['ApplicantIncome'])
        coapplicant_income = float(input_data['CoapplicantIncome'])
        loan_amount = float(input_data['LoanAmount'])
        loan_amount_term = float(input_data['Loan_Amount_Term'])
        credit_history = float(input_data['Credit_History'])
        property_area = input_data['Property_Area']
        input_df = pd.DataFrame({
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

        prediction = model.predict(input_df)

        if prediction[0] == 1:
            result = "Approved"
        else:
            result = "Not Approved"
        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
