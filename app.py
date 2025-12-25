import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

tenure = st.number_input("Tenure (months)")
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")
tickets = st.number_input("Support Tickets")
contract = st.selectbox("Contract Type", ["Month-to-Month", "1 Year", "2 Year"])

st.write("Model demo only. Replace UI as needed.")
