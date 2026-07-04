import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



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
#     n_neighbors=5,
#     weights='uniform',
#     p=2
# )

# second model (slightly better)
# model_params = dict(
#     n_neighbors=9,
#     weights='distance',
#     p=2,
#     algorithm='auto',2
#     leaf_size=30,
#     n_jobs=-1
# )

# third model (best performer)
model_params = dict(
    n_neighbors=11,
    weights='distance',
    p=1,                       
    algorithm='auto',
    leaf_size=40,
    n_jobs=-1
)


for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    model = KNeighborsRegressor(**model_params)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    
    mae_list.append(mae)
    rmse_list.append(rmse)
    r2_list.append(r2)
    

print("KNN (5-fold CV)")
print(f"MAE: {np.mean(mae_list):.3f} hours")
print(f"RMSE: {np.mean(rmse_list):.3f} hours")
print(f"R²: {np.mean(r2_list):.3f}")
