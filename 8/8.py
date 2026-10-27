import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Resolve dataset path relative to this script so it works from any folder
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "loan_data_100_records.csv")

# Load data
data = pd.read_csv(dataset_path)[["ApplicantIncome","LoanAmount","Credit_History","Loan_Status"]].dropna()

# Encode target
data["Loan_Status"] = LabelEncoder().fit_transform(data["Loan_Status"])

# Features and target
X = data[["ApplicantIncome","LoanAmount","Credit_History"]]
y = data["Loan_Status"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Accuracy
y_pred = knn.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# New applicant prediction
new_applicant = scaler.transform([[5000, 150, 1]])
prediction = knn.predict(new_applicant)
print("Loan Approved" if prediction[0] == 1 else "Loan Not Approved")
