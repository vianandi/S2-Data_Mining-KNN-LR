# regression_analysis.py

# 1. Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Data
def load_data(path="insurance.csv"):
    return pd.read_csv(path)

# 3. Preprocessing
def preprocess_data(df):
    # Remove outliers using Z-score
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z_scores < 3).all(axis=1)]

    # Encode categorical features
    df = pd.get_dummies(df, drop_first=True)
    return df

# 4. Train & Evaluate Models
def evaluate_models(df):
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

        # KNN Regression with K = 3, 5, 7, 9
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

# 5. Main Function
def main():
    df = load_data()
    df = preprocess_data(df)
    results_df = evaluate_models(df)

    print("\n===== Performance Comparison =====")
    print(results_df.sort_values(by='R2', ascending=False))

    # (Optional) Visualisasi
    sns.barplot(data=results_df[results_df["Model"] == "KNN"], x='K', y='R2', hue='Split')
    plt.title("KNN R² Score by K and Split")
    plt.show()

# 6. Run Program
if __name__ == "__main__":
    main()