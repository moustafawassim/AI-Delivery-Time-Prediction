# AI-Based Delivery Time Prediction

This project predicts logistics delivery trip duration using machine learning regression models. The goal is to estimate delivery time based on trip, vehicle, driver, and environmental features.

## Project Overview

Accurate delivery time prediction is important for logistics companies because it helps improve scheduling, driver assignment, fleet planning, customer satisfaction, and operational efficiency.

In this project, multiple machine learning models were implemented and compared to predict trip duration.

## Models Implemented

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors Regressor
- XGBoost Regressor
- Neural Network / MLP Regressor

## Evaluation Metrics

The models were evaluated using 5-fold cross-validation and the following metrics:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R² Score

## Best Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.299 | 0.452 | 0.707 |
| Decision Tree | 0.290 | 0.470 | 0.683 |
| KNN | 0.321 | 0.499 | 0.643 |
| Random Forest | 0.228 | 0.381 | 0.792 |
| Neural Network | 0.241 | 0.393 | 0.778 |
| XGBoost | 0.203 | 0.350 | 0.824 |

XGBoost achieved the best overall performance with the highest R² score and lowest prediction error.

## Dataset

The dataset used for this project was provided for academic purposes and is not included in this repository.

The expected input file for most models is:

```text
preprocessed_data.csv
```

The neural network script expects:

```text
nn_data.npz
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/AI-Delivery-Time-Prediction.git
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Place the required dataset file in the project folder.

4. Run any model file, for example:

```bash
python final_XGboost.py
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## Project Purpose

This project was developed to explore and compare different machine learning approaches for delivery time prediction in logistics operations.
