import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_PATH = "models/student_performance_model.pkl"


def main():

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Step 1: Load Mathematics dataset
    data = pd.read_csv("data/student-mat.csv", sep=";")

    # Convert target to numeric
    data["G3"] = pd.to_numeric(data["G3"], errors="coerce")

    # Remove rows with missing target
    data = data.dropna(subset=["G3"])

    print("Dataset shape:", data.shape)

    # Step 2: Define features and target
    # G1 and G2 are intentionally excluded
    X = data.drop(columns=["G3", "G1", "G2"])
    y = data["G3"]

    # Step 3: Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Step 4: Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    # Step 5: Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]
    )

    # Step 6: Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ]
    )

    # Step 7: Random Forest model
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ]
    )

    # Step 8: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Step 9: 5-fold cross-validation on training data
    cv_scores = cross_val_score(
        model_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    print("\nCross-validation R² scores:")
    print(cv_scores)

    print(f"Mean CV R²: {cv_scores.mean():.3f}")

    # Step 10: Train final model
    print("\nTraining model...")
    model_pipeline.fit(X_train, y_train)

    # Step 11: Test predictions
    predictions = model_pipeline.predict(X_test)

    # Step 12: Evaluation
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nModel Performance")
    print("-----------------")
    print(f"MAE:  {mae:.3f}")
    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    # Step 13: Save complete pipeline
    joblib.dump(model_pipeline, MODEL_PATH)

    print(f"\nModel saved successfully at: {MODEL_PATH}")


if __name__ == "__main__":
    main()