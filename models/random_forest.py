import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv("preprocessed_data.csv")

X = data.drop(columns=["trip_duration"])
y = data["trip_duration"]


kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_list, rmse_list, r2_list = [], [], []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Choose model configuration
    
    # first model (baseline)
    # model = RandomForestRegressor(
    #     n_estimators=500,
    #     max_depth=None,             
    #     min_samples_split=4,
    #     min_samples_leaf=2,
    #     max_features='sqrt',        
    #     bootstrap=True,
    #     max_samples=0.85,
    #     min_impurity_decrease=0.00005,       
    #     random_state=42,
    #     n_jobs=-1
    # )
    

    # second model (slightly better)
    #model = RandomForestRegressor(n_estimators=200, random_state=42)

    # third model (best performer)
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=15,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features=0.6,           
        max_samples=0.8,            
        min_impurity_decrease=0.0001,  
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae_list.append(mean_absolute_error(y_test, y_pred))
    rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2_list.append(r2_score(y_test, y_pred))

print("Random Forest (5-fold CV)")
print(f"MAE: {np.mean(mae_list):.3f} hours")
print(f"RMSE: {np.mean(rmse_list):.3f} hours")
print(f"R²: {np.mean(r2_list):.3f}")
