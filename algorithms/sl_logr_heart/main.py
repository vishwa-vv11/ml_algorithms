#libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df=pd.read_csv(r"C:\Users\darkp\OneDrive\Desktop\vsc_projects\projects!!\algorithms\sl_logr_heart\heart.csv")
#split features and target
y=df["target"]
x=df.drop("target",axis=1)
#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#model
model=LogisticRegression()
model.fit(x_train,y_train)
#feature scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)   
x_test=scaler.transform(x_test)
#predict
y_pred=model.predict(x_test)
#evaluate
accuracy=accuracy_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred) 
class_report=classification_report(y_test,y_pred)
print("Model Performance:") 
print("Accuracy:", round(accuracy,4))
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
      