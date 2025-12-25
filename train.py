import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/churn_data.csv")

X = df.drop("churn", axis=1)
y = df["churn"]

numeric = ["tenure", "monthly_charges", "total_charges", "support_tickets"]
categorical = ["contract_type"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(), categorical)
    ]
)

X_processed = preprocess.fit_transform(X)

sm = SMOTE()
X_resampled, y_resampled = sm.fit_resample(X_processed, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
