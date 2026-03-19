# customer_churn
This repository contains a Jupyter Notebook for customer churn prediction, including all steps of data preprocessing and model training.
# Customer Churn Prediction Dashboard

This project is an interactive dashboard built with Streamlit to predict customer churn using machine learning models.

## Project Overview
The goal of this project is to analyze customer data and predict whether a customer is likely to churn (leave the company).
The application allows users to input customer information and receive a prediction with a probability score and risk level.

## Models Used
- Logistic Regression
- Decision Tree

## Features
- Interactive dashboard built with Streamlit
- Real-time churn prediction
- Probability and risk level classification (Low / Medium / High)
- User input form for customer data
- Data visualization using Matplotlib and Seaborn

## Files in this Repository
- app.py → Main Streamlit application
- logistic_regression_churn_model.pkl → Trained Logistic Regression model
- decision_tree_churn_model.pkl → Trained Decision Tree model
- scaler.pkl → Scaler used for preprocessing
- Telco-Customer-Churn.csv → Dataset used for training and analysis

## How to Run the Application

1. Clone the repository:
git clone https://github.com/sajadahmadnezhad/customer_churn.git
cd your-repository-name

2. Install the required libraries:
pip install -r requirements.txt

3. Run the Streamlit application:
streamlit run app.py

4. Open the application in your browser:
http://localhost:8501

## Requirements
- Python 3.x
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Purpose
This project demonstrates how machine learning can be applied to predict customer churn and support data-driven decision-making.

## Author
Sajad Ahmadnezhad
