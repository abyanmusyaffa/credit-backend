from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature order from model (61 features, one-hot encoded)
FEATURE_NAMES = [
    'month_duration', 'credit_amount', 'payment_to_income_ratio',
    'residence_since', 'age', 'n_credits', 'n_guarantors',
    'status_account_0 to < 200 DM', 'status_account_< 0 DM',
    'status_account_>= 200 DM', 'status_account_no checking account',
    'credit_history_all credits at this bank paid back duly',
    'credit_history_critical account/ other credits existing (not at this bank)',
    'credit_history_delay in paying off in the past',
    'credit_history_existing credits paid back duly till now',
    'credit_history_no credits taken/ all credits paid back duly',
    'purpose_business', 'purpose_car (new)', 'purpose_car (used)',
    'purpose_domestic appliances', 'purpose_education',
    'purpose_furniture/equipment', 'purpose_others',
    'purpose_radio/television', 'purpose_repairs', 'purpose_retraining',
    'status_savings_100 to < 500 DM', 'status_savings_500 to < 1000 DM',
    'status_savings_< 100 DM', 'status_savings_>= 1000 DM',
    'status_savings_unknown/ no savings account',
    'years_employment_1 to < 4 years', 'years_employment_4 to < 7 years',
    'years_employment_< 1 year', 'years_employment_>= 7 years',
    'years_employment_unemployed',
    'status_and_sex_female : divorced/separated/married',
    'status_and_sex_male : divorced/separated',
    'status_and_sex_male : married/widowed',
    'status_and_sex_male : single',
    'secondary_obligor_co-applicant', 'secondary_obligor_guarantor',
    'secondary_obligor_none',
    'collateral_car', 'collateral_none', 'collateral_real estate',
    'collateral_savings agreement/life insurance',
    'other_installment_plans_bank', 'other_installment_plans_none',
    'other_installment_plans_stores',
    'housing_for free', 'housing_own', 'housing_rent',
    'job_management/ self-employed/highly qualified employee',
    'job_skilled employee/ official',
    'job_unemployed/ unskilled - non-resident',
    'job_unskilled - resident',
    'telephone_none',
    'telephone_yes, registered under the customers name',
    'is_foreign_worker_no', 'is_foreign_worker_yes'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Build feature vector (all zeros first)
        features = {name: 0 for name in FEATURE_NAMES}

        # Numerical features
        features['month_duration'] = float(data.get('month_duration', 0))
        features['credit_amount'] = float(data.get('credit_amount', 0))
        features['payment_to_income_ratio'] = float(data.get('payment_to_income_ratio', 0))
        features['residence_since'] = float(data.get('residence_since', 0))
        features['age'] = float(data.get('age', 0))
        features['n_credits'] = float(data.get('n_credits', 0))
        features['n_guarantors'] = float(data.get('n_guarantors', 0))

        # One-hot categorical features
        def set_one_hot(prefix, value):
            key = f"{prefix}{value}"
            if key in features:
                features[key] = 1

        set_one_hot('status_account_', data.get('status_account', ''))
        set_one_hot('credit_history_', data.get('credit_history', ''))
        set_one_hot('purpose_', data.get('purpose', ''))
        set_one_hot('status_savings_', data.get('status_savings', ''))
        set_one_hot('years_employment_', data.get('years_employment', ''))
        set_one_hot('status_and_sex_', data.get('status_and_sex', ''))
        set_one_hot('secondary_obligor_', data.get('secondary_obligor', ''))
        set_one_hot('collateral_', data.get('collateral', ''))
        set_one_hot('other_installment_plans_', data.get('other_installment_plans', ''))
        set_one_hot('housing_', data.get('housing', ''))
        set_one_hot('job_', data.get('job', ''))
        set_one_hot('telephone_', data.get('telephone', ''))
        set_one_hot('is_foreign_worker_', data.get('is_foreign_worker', ''))

        # Create input array in correct order
        input_array = np.array([[features[name] for name in FEATURE_NAMES]])

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0]

        result = {
            'prediction': int(prediction),
            'label': 'Layak' if prediction == 1 else 'Tidak Layak',
            'probability_layak': float(probability[1]) if len(probability) > 1 else float(probability[0]),
            'probability_tidak_layak': float(probability[0]) if len(probability) > 1 else 0.0,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'DecisionTreeClassifier', 'features': len(FEATURE_NAMES)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
