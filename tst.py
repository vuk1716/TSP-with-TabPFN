import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

# ===== 1. Read your CSV =====
csv_path = "tsp_route.csv"  # change this
df = pd.read_csv(csv_path)

# Separate target columns (x and y) from inputs
target_x = df["x"].astype(float)
target_y = df["y"].astype(float)
feature_cols = [col for col in df.columns if col not in ["x", "y"]]
X = df[feature_cols]

# ===== 2. Train separate models for x and y =====
def train_and_evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    regressor = TabPFNRegressor()
    regressor.fit(X_train, y_train)

    predictions = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"[{label}] MSE:", mse)
    print(f"[{label}] R²:", r2)

    return regressor

print("=== Training for x-coordinate ===")
model_x = train_and_evaluate(X, target_x, "x")

print("\n=== Training for y-coordinate ===")
model_y = train_and_evaluate(X, target_y, "y")

# ===== 3. Create a new 100-row dataset for evaluation =====
# Here we'll just randomly sample from original features
new_X = X.sample(n=100, replace=True, random_state=123).reset_index(drop=True)

# Predict for the new dataset
pred_x_new = model_x.predict(new_X)
pred_y_new = model_y.predict(new_X)

# Combine predictions into a dataframe
pred_df = new_X.copy()
pred_df["predicted_x"] = pred_x_new
pred_df["predicted_y"] = pred_y_new

print("\n=== Predictions for new 100 instances ===")
print(pred_df.head())

# Optionally save
pred_df.to_csv("predictions_new_100.csv", index=False)
