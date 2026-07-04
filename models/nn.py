import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = np.load("nn_data.npz")

X = np.vstack((data["X_train"], data["X_test"]))
y = np.hstack((data["y_train"], data["y_test"]))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
rmse_scores = []
r2_scores = []

fold = 1

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # first model
    # model = MLPRegressor(
    #     hidden_layer_sizes=(512, 256, 128, 64),
    #     activation="relu",
    #     solver="adam",
    #     learning_rate="adaptive",
    #     learning_rate_init=0.0004,
    #     alpha=0.00005,          
    #     batch_size=32,         
    #     max_iter=1200,
    #     early_stopping=True,
    #     validation_fraction=0.15,
    #     n_iter_no_change=30,
    #     random_state=42
    # )

    # second model
    # model = MLPRegressor(
    #     hidden_layer_sizes=(300, 150, 75),
    #     activation="relu",
    #     solver="adam",
    #     learning_rate_init=0.0003,
    #     alpha=0.0005,                 
    #     batch_size=64,
    #     max_iter=800,                 
    #     early_stopping=True,
    #     validation_fraction=0.15,
    #     n_iter_no_change=20,
    #     random_state=42
    # )


    # third model
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.0005,
        alpha=0.0001,              
        batch_size=64,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mae_scores.append(mae)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    fold += 1

print("Neural Network (5-Fold CV)")
print(f"MAE: {np.mean(mae_scores):.3f} hours")
print(f"RMSE: {np.mean(rmse_scores):.3f} hours")
print(f"R²: {np.mean(r2_scores):.3f}")