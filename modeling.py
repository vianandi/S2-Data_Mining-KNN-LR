from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def train_and_evaluate(df):
    X = df.drop("charges", axis=1)
    y = df["charges"]
    splits = [(0.9, 0.1), (0.8, 0.2)]
    results = []

    for train_size, test_size in splits:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)

        results.append({
            'Model': 'Linear Regression',
            'K': '-',
            'Split': f'{int(train_size*100)}/{int(test_size*100)}',
            'MAE': mean_absolute_error(y_test, y_pred_lr),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            'R2': r2_score(y_test, y_pred_lr)
        })

        for k in [3, 5, 7, 9]:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            y_pred_knn = knn.predict(X_test_scaled)

            results.append({
                'Model': 'KNN',
                'K': k,
                'Split': f'{int(train_size*100)}/{int(test_size*100)}',
                'MAE': mean_absolute_error(y_test, y_pred_knn),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_knn)),
                'R2': r2_score(y_test, y_pred_knn)
            })

    return pd.DataFrame(results)