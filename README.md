# ATM Cash Replenishment Prediction

Machine Learning-based system for predicting ATM cash demand and recommending optimal cash replenishment using historical ATM data.

## 📌 Project Overview

The ATM Cash Replenishment Prediction System uses Machine Learning to predict the next-day cash demand of an ATM and recommend an appropriate replenishment amount.

The system helps reduce cash shortages, avoid unnecessary idle cash, and improve ATM cash management.

## 🚀 Features

- Predicts next-day ATM cash demand
- Recommends cash replenishment amount
- Cash gap analysis
- Demand level classification
- Refill priority score
- Interactive data visualizations
- 7-day scenario analysis
- Prediction history
- Downloadable prediction reports

## 🤖 Machine Learning

**Algorithm:** Linear Regression

**Target Variable:** `Cash_Demand_Next_Day`

**Model Performance:**
- Training R²: 0.863
- Testing R²: 0.871

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Streamlit
- Google Colab
- VS Code

## 📊 Project Workflow

Historical ATM Data  
↓  
Data Preprocessing  
↓  
Feature Encoding  
↓  
Train-Test Split  
↓  
Linear Regression Model  
↓  
Model Evaluation  
↓  
Save Trained Model  
↓  
Streamlit Dashboard  
↓  
Cash Demand Prediction  
↓  
Replenishment Recommendation

## 📁 Project Structure

```text
ATM-Cash-Replenishment/
│
├── app.py
├── atm_cash_management_dataset.csv
├── ATM_Cash_Replenishment_Model.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup

```bash
git clone https://github.com/gargi-tiwari312/ATM-Cash-Replenishment.git
cd ATM-Cash-Replenishment
pip install -r requirements.txt
streamlit run app.py
```
