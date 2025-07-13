# -*- coding: utf-8 -*-
# This is a Streamlit app to predict Data Scientist salary

import streamlit as st
import pandas as pd
import joblib  # to load the trained model

# 🔷 Step 1: Load the saved model
model = joblib.load('salary_predictor_rf.pkl')

# 🔷 Step 2: Define the columns the model expects
# These are the features the model was trained with
feature_columns = [
    'Rating', 'Founded', 'job_in_headquarters', 'python_yn', 'sql_yn', 'excel_yn', 'tableau_yn',
    'Type of ownership_College / University', 'Type of ownership_Company - Private',
    'Type of ownership_Company - Public', 'Type of ownership_Contract', 'Type of ownership_Government',
    'Type of ownership_Hospital', 'Type of ownership_Nonprofit Organization',
    'Type of ownership_Other Organization', 'Type of ownership_Private Practice / Firm',
    'Type of ownership_School / School District', 'Type of ownership_Subsidiary or Business Segment',
    'Type of ownership_Unknown', 'Sector_Aerospace & Defense', 'Sector_Biotech & Pharmaceuticals',
    'Sector_Business Services', 'Sector_Education', 'Sector_Finance', 'Sector_Health Care',
    'Sector_Information Technology', 'Sector_Insurance', 'Sector_Manufacturing', 'Sector_Other',
    'Revenue_$10 to $25 million (USD)', 'Revenue_$10+ billion (USD)', 'Revenue_$100 to $500 million (USD)',
    'Revenue_$2 to $5 billion (USD)', 'Revenue_$25 to $50 million (USD)', 'Revenue_$5 to $10 million (USD)',
    'Revenue_$50 to $100 million (USD)', 'Revenue_$500 million to $1 billion (USD)', 'Revenue_Other',
    'Revenue_Unknown / Non-Applicable'
]

# 🔷 Step 3: App title and inputs
st.title("💼 Data Scientist Salary Predictor")
st.write("Enter the job details below to predict the average salary.")

# Get user inputs
rating = st.slider("Company Rating (1-5)", 1.0, 5.0, 3.0)
founded = st.number_input("Company Founded Year", min_value=1800, max_value=2025, value=2000)

ownership = st.selectbox("Type of Ownership", [
    "College / University", "Company - Private", "Company - Public", "Contract", "Government", "Hospital",
    "Nonprofit Organization", "Other Organization", "Private Practice / Firm", "School / School District",
    "Subsidiary or Business Segment", "Unknown"
])

sector = st.selectbox("Sector", [
    "Aerospace & Defense", "Biotech & Pharmaceuticals", "Business Services", "Education", "Finance", "Health Care",
    "Information Technology", "Insurance", "Manufacturing", "Other"
])

revenue = st.selectbox("Revenue", [
    "$10 to $25 million (USD)", "$10+ billion (USD)", "$100 to $500 million (USD)", "$2 to $5 billion (USD)",
    "$25 to $50 million (USD)", "$5 to $10 million (USD)", "$50 to $100 million (USD)",
    "$500 million to $1 billion (USD)", "Other", "Unknown / Non-Applicable"
])

# Yes/No inputs
job_in_hq   = st.radio("Job in Headquarters?", ["Yes", "No"])
python_yn   = st.radio("Python Mentioned?", ["Yes", "No"])
sql_yn      = st.radio("SQL Mentioned?", ["Yes", "No"])
excel_yn    = st.radio("Excel Mentioned?", ["Yes", "No"])
tableau_yn  = st.radio("Tableau Mentioned?", ["Yes", "No"])

# 🔷 Step 4: When user clicks predict
if st.button("Predict Salary"):

    # Create a row of input data with all zeros
    input_row = pd.Series(0, index=feature_columns, dtype=float)

    # Fill the numeric and yes/no fields
    input_row['Rating'] = rating
    input_row['Founded'] = founded
    input_row['job_in_headquarters'] = 1 if job_in_hq == "Yes" else 0
    input_row['python_yn'] = 1 if python_yn == "Yes" else 0
    input_row['sql_yn'] = 1 if sql_yn == "Yes" else 0
    input_row['excel_yn'] = 1 if excel_yn == "Yes" else 0
    input_row['tableau_yn'] = 1 if tableau_yn == "Yes" else 0

    # Set the correct one-hot columns to 1
    ownership_col = f"Type of ownership_{ownership}"
    sector_col    = f"Sector_{sector}"
    revenue_col   = f"Revenue_{revenue}"

    if ownership_col in input_row:
        input_row[ownership_col] = 1
    if sector_col in input_row:
        input_row[sector_col] = 1
    if revenue_col in input_row:
        input_row[revenue_col] = 1

    # Convert to DataFrame (model needs DataFrame)
    input_df = input_row.to_frame().T

    # Make the prediction
    salary = model.predict(input_df)[0]

    # Show result
    st.success(f"💰 Predicted Average Salary: **{salary:.2f}k USD**")
