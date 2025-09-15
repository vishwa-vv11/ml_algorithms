import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1️⃣ Load the dataset
path=r"C:\Users\darkp\OneDrive\Desktop\vsc_projects\projects!!\algorithms\sl_dt_loan\loan.csv"
df = pd.read_csv(path)  # replace with your actual file name
print(df.head())
print(df.info())

# 2️⃣ Drop Loan_ID (not useful for prediction)
df.drop("Loan_ID", axis=1, inplace=True)

# 3️⃣ Handle Missing Values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# 4️⃣ Encode Categorical Variables
df = pd.get_dummies(df, drop_first=True)

# 5️⃣ Split Features and Target
X = df.drop("Loan_Status_Y", axis=1)  # Y = approved, N = rejected
y = df["Loan_Status_Y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Train Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 7️⃣ Evaluate Model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8️⃣ Visualize Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=["Rejected", "Approved"], filled=True)
plt.show()

# 9️⃣ Feature Importance
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", feature_importance)
