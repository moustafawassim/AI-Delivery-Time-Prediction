import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("preprocessed_data.csv")

X = df.drop("trip_duration", axis=1)
y = df["trip_duration"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = LinearRegression()

mae_scores = []
rmse_scores = []
r2_scores = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_scores.append(mean_absolute_error(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2_scores.append(r2_score(y_test, y_pred))

print("Linear Regression (5-Fold CV)")
print(f"MAE: {np.mean(mae_scores):.3f} hours")
print(f"RMSE: {np.mean(rmse_scores):.3f} hours")
print(f"R²: {np.mean(r2_scores):.3f}")
