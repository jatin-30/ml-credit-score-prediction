import joblib
import numpy as np
import pandas as pd

# Load saved model, scaler, and metadata
MODEL_PATH = 'artifacts/model_data.joblib'
model_data = joblib.load(MODEL_PATH)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']


def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                  delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                  residence_type, loan_purpose, loan_type):
    """
    Prepares the input for prediction by scaling and encoding features.
    """

    # Basic feature calculations
    loan_to_income = loan_amount / income if income > 0 else 0

    # Construct full input with dummy values for scaling completeness
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_to_income,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,

        # One-hot encodings
        'residence_type_Owned': int(residence_type == 'Owned'),
        'residence_type_Rented': int(residence_type == 'Rented'),
        'loan_purpose_Education': int(loan_purpose == 'Education'),
        'loan_purpose_Home': int(loan_purpose == 'Home'),
        'loan_purpose_Personal': int(loan_purpose == 'Personal'),
        'loan_type_Unsecured': int(loan_type == 'Unsecured'),

        # Dummy placeholders for scaler compatibility
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])

    # Apply scaler only to scale-relevant columns
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Keep only model-required features
    df = df[features]

    return df


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    """
    Calculates probability, credit score, and rating from logistic regression model.
    """

    linear_score = np.dot(input_df.values, model.coef_.T) + model.intercept_
    default_prob = 1 / (1 + np.exp(-linear_score))
    non_default_prob = 1 - default_prob

    credit_score = base_score + non_default_prob.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score[0])

    return float(default_prob.flatten()[0]), int(credit_score[0]), rating


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    """
    Main predict function to be used externally.
    """
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months,
                             avg_dpd_per_delinquency, delinquency_ratio,
                             credit_utilization_ratio, num_open_accounts,
                             residence_type, loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score(input_df)
    return probability, credit_score, rating

