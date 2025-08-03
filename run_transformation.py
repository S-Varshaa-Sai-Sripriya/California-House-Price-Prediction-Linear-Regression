# run_transformation.py

from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.transform_data("artifacts/data.csv")
    print("Data transformation complete.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
