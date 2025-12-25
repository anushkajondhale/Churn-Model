# Customer Churn Prediction Model

This project builds a Machine Learning classification model to predict customer churn using historical customer account and usage data. The goal is to identify customers who are likely to stop using a service so that retention actions can be taken in advance.

The project includes data preprocessing, feature engineering, model training, model comparison, and performance evaluation. Special focus is given to improving recall on the minority churn class using resampling techniques and model tuning.

## 🔍 Problem Statement
Customer churn is a key business problem. Retaining an existing customer is often more cost-effective than acquiring a new one. This model helps flag high-risk customers.

## ⚙ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Seaborn  
- Matplotlib  

## 📌 Workflow
1. Load and explore dataset  
2. Handle missing values  
3. Encode categorical features  
4. Split train/test sets  
5. Handle class imbalance  
6. Train ML models  
7. Compare performance  
8. Improve recall  
9. Evaluate using metrics  

## 🚀 How to Run

### Step 1 — Install dependencies
```
pip install -r requirements.txt
```

### Step 2 — Train the model
```
python train.py
```

### Step 3 — Run the prediction app (Streamlit)
```
streamlit run app.py
```

## 📊 Dataset
Sample dataset is provided under `data/churn_data.csv`.

Columns include:
- customer_id  
- tenure  
- monthly_charges  
- total_charges  
- contract_type  
- support_tickets  
- churn (target)  

## 🔮 Future Enhancements
- Deploy with Flask / FastAPI  
- Save model using pickle  
- Add SHAP explainability  
- Create dashboard view  

## 🎯 Project Purpose
This project was developed as part of learning and practicing Data Science techniques on real-world business problems.
