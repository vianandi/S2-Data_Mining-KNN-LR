import pandas as pd
import numpy as np
from scipy import stats

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Remove outliers
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z_scores < 3).all(axis=1)]

    # Encode kategorikal
    df = pd.get_dummies(df, drop_first=True)
    return df