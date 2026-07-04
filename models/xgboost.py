import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


df = pd.read_csv("preprocessed_data.csv")

X = df.drop(columns=["trip_duration"])
y = df["trip_duration"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_list   = []
rmse_list  = []
r2_list    = []


# Choose model configuration 
# first model (baseline)
# model_params = dict(
#     n_estimators    = 300,
#     max_depth       = 6,
#     learning_rate   = 0.1,
#     objective       = 'reg:squarederror',
#     random_state    = 42
# )

# second model (slightly better)
# model_params = dict(
#     n_estimators       = 800,
#     max_depth          = 5,
#     learning_rate      = 0.03,
#     subsample          = 0.8,
#     colsample_bytree   = 0.8,
#     reg_alpha          = 0.5,
#     reg_lambda         = 1.0,
#     objective          = 'reg:squarederror',
#     random_state       = 42
# )

# third model (best performer)
model_params = dict(
    n_estimators       = 1200,
    learning_rate      = 0.02,
    max_depth          = 4,
    subsample          = 0.85,
    colsample_bytree   = 0.85,
    reg_alpha          = 1.0,
    reg_lambda         = 2.0,
    gamma              = 0.1,
    min_child_weight   = 3,
    objective          = 'reg:squarederror',
    random_state       = 42,
)


for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)
    

print("XGBoost (5-fold CV)")
print(f"MAE: {np.mean(mae_list):.3f} hours")
print(f"RMSE: {np.mean(rmse_list):.3f} hours")
print(f"R²: {np.mean(r2_list):.3f}")
