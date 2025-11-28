import pandas as pd
import numpy as np
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("Reading heart.csv...")
df = pd.read_csv("heart.csv")
print(f"✓ CSV loaded: {df.shape}")

X = df.drop("target", axis=1)
y = df["target"]
print(f"✓ Features: {X.shape}, Target: {y.shape}")

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Training set: {X_train.shape}, Test set: {X_test.shape}")

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")

print("Training RandomForestClassifier (300 estimators)...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)
model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully")

print("Saving model to heart_model.pkl...")
with open("heart_model.pkl", "wb") as f:
    pickle.dump((model, scaler, X.columns.tolist()), f)

print("✅ Model trained and heart_model.pkl saved successfully!")
