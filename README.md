# Credit Risk Prediction Modelling

A lightweight, interactive Streamlit app to predict credit scores and default risk probabilities using customer financial and credit behavior data.

## Features

- Interactive input form with 12+ customer credit-related features
- Real-time prediction of:
  - Default Probability
  - Credit Score (300–900)
  - Credit Rating (Poor, Average, Good, Excellent)
- Logistic regression–based model using domain-relevant feature engineering
- Scaled features using a trained `MinMaxScaler`
- Clean and intuitive UI built in Streamlit

---

## Tech Stack

- Python 3.10+
- scikit-learn
- joblib
- Streamlit
- pandas & numpy
  
## Project Structure

credit-risk-model/
│
├── artifacts/
│ └── model_data.joblib # Trained model, scaler, features list
│
├── app.py # Streamlit frontend
├── prediction_helper.py # Backend logic (scaling, encoding, prediction)
├── README.md # This file


---

## Run This Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py

