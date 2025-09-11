# Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
filepath = r"C:\Users\darkp\OneDrive\Desktop\vsc_projects\projects!!\algorithms\sl_lr_salary\salary_dataset.csv"
df = pd.read_csv(filepath)

# One-Hot Encoding for categorical features
categorical_cols = ['Gender', 'Role', 'Education', 'Department', 'Remote']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature Engineering
# Add Experience squared to capture non-linear growth
df_encoded['Experience_Sq'] = df_encoded['Experience'] ** 2

# Add interaction terms (Experience * Years_in_Current_Role)
df_encoded['Exp_YearsInRole'] = df_encoded['Experience'] * df_encoded['Years_in_Current_Role']

# Split features and target
X = df_encoded.drop('Salary', axis=1)
y = df_encoded['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Performance with Feature Engineering:")
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R squared (R²):", round(r2, 4))
