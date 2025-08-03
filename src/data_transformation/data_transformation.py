# src/components/data_transformation.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def get_data_transformer(self, df: pd.DataFrame) -> ColumnTransformer:
        """Creates and returns a preprocessor pipeline."""

        num_features = df.drop("target", axis=1).columns.tolist()

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer([
            ("num_pipeline", num_pipeline, num_features)
        ])

        return preprocessor

    def transform_data(self, data_path: str):
        """Preprocesses the data and saves transformer and split sets."""

        df = pd.read_csv(data_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        preprocessor = self.get_data_transformer(df)
        X_transformed = preprocessor.fit_transform(X)

        # Save preprocessor
        joblib.dump(preprocessor, self.preprocessor_path)

        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test
