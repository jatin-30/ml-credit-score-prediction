import streamlit as st
from prediction_helper import predict  # Make sure this path is correct

# Set Streamlit page config
st.set_page_config(page_title="Credit Risk Prediction", page_icon="📊")
st.title("Credit Risk Prediction")

# Layout: Input Controls
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

# Row 1: Basic info
with row1[0]:
    age = st.number_input("Age", min_value=18, max_value=100, value=28)
with row1[1]:
    income = st.number_input("Annual Income (₹)", min_value=1, step=10000, value=1200000)
with row1[2]:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, step=10000, value=2560000)

# Row 2: Loan & DPD
loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row2[0]:
    st.markdown("**Loan to Income Ratio**")
    st.markdown(f"{loan_to_income_ratio:.2f}")

with row2[1]:
    loan_tenure_months = st.number_input("Loan Tenure (months)", min_value=1, max_value=360, value=36)
with row2[2]:
    avg_dpd_per_delinquency = st.number_input("Avg DPD per Delinquency", min_value=0, value=20)

# Row 3: Credit behavior
with row3[0]:
    delinquency_ratio = st.slider("Delinquency Ratio (%)", min_value=0, max_value=100, value=30)
with row3[1]:
    credit_utilization_ratio = st.slider("Credit Utilization Ratio (%)", min_value=0, max_value=100, value=30)
with row3[2]:
    num_open_accounts = st.number_input("Open Credit Accounts", min_value=1, max_value=20, value=2)

# Row 4: Categorical fields
with row4[0]:
    residence_type = st.selectbox("Residence Type", ['Owned', 'Rented', 'Mortgage'])
with row4[1]:
    loan_purpose = st.selectbox("Loan Purpose", ['Education', 'Home', 'Auto', 'Personal'])
with row4[2]:
    loan_type = st.selectbox("Loan Type", ['Unsecured', 'Secured'])

# Predict Button
if st.button("Calculate Risk"):
    try:
        # Scale % inputs to 0–1
        credit_util_ratio_scaled = credit_utilization_ratio / 100
        delinquency_ratio_scaled = delinquency_ratio / 100

        # Call prediction
        probability, credit_score, rating = predict(
            age, income, loan_amount, loan_tenure_months,
            avg_dpd_per_delinquency, delinquency_ratio_scaled,
            credit_util_ratio_scaled, num_open_accounts,
            residence_type, loan_purpose, loan_type
        )

        # Output
        st.markdown("---")
        st.subheader("🧾 Prediction Result")
        st.success(f"**Default Probability:** {probability:.2%}")
        st.info(f"**Credit Score:** {credit_score} / 900")
        st.warning(f"**Rating:** {rating}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
