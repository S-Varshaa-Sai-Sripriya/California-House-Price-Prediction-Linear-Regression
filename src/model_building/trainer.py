# src/model_building/trainer.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.model_building.model import LinearRegressionScratch
from src.evaluation import evaluate

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def train_and_evaluate(
    file_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Train and evaluate the linear regression model from scratch.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
    
    Returns:
        Tuple[float, float, float]: Evaluation metrics (MAE, MSE, RMSE).
    """
    df = load_data(file_path)
    
    # Assuming last column is the target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegressionScratch()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae, mse, rmse = evaluate(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return mae, mse, rmse
