#libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
#load the data
filepath=r"C:\Users\darkp\OneDrive\Desktop\vsc_projects\algorithms\SL_linear_regression-house\house_dataset.csv"
df=pd.read_csv(filepath)
print(df.head())
#label encoding
Le=LabelEncoder()
df["city"]=Le.fit_transform(df["city"])
x=df.drop("price",axis=1)
y=df["price"]
#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#model
model=LinearRegression()
model.fit(x_train,y_train)
#prediction
y_pred=model.predict(x_test)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print("Root Mean Squared Error (RMSE):", rmse)
r2=model.score(x_test,y_test)
print("R squared: ",r2)
