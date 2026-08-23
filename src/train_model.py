import os
import joblib
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split 

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




MODEL_PATH = "models/student_performance_model.pkl"     # where tuned model will be saved


def main():
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    

    # Step 1: Load dataset
    data_mat = pd.read_csv("data/student-mat.csv", sep=";")
    data_por = pd.read_csv("data/student-por.csv", sep=";")

    # Combine them
    data = pd.concat([data_mat, data_por], ignore_index=True)
    
    # Convert grade columns to integers
    for col in ["G1", "G2", "G3"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    print("Combined dataset shape:", data.shape)

    

    # Step 2: Define features (X) and target (y)
    X = data.drop(columns=["G1", "G2", "G3"])
    y = data["G3"]
    
    

    # Step 3: Separate numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
        ).columns.tolist()
    
    categorical_columns = X.select_dtypes(
        include=["object"]
        ).columns.tolist()



    # Step 4: Build preprocessing pipelines
    # For numerical features: impute missing values + scale
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    

    # For categorical features: impute missing values + one-hot encode
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    
    

    # Step 5: Combine both pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ]
    )
    
    
    # Step 6: Build full pipeline (preprocessing + model)
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=800,      # number of trees
                max_depth=30,          # maximum depth of each tree
                min_samples_split=5,   # minimum samples to split a node
                min_samples_leaf=2,    # minimum samples per leaf
                random_state=42
                )
            )
        ]
    )
    
    
    
    # Step 7: Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    
    # Step 8: Cross-validation on full dataset (10-fold)
    cv_scores = cross_val_score(model_pipeline, X, y, cv=10, scoring="r2")
    
    print("Cross-validation R² scores:", cv_scores)
    print("Mean R²:", cv_scores.mean())
        
        
        
    # Step 9: Train model on training set
    print("Training model...")
    model_pipeline.fit(X_train, y_train)
    


    # Step 10: Predict on test set
    predictions = model_pipeline.predict(X_test)
    print(predictions)
    
    
    
    # Step 11: Evaluate performance
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nModel Performance")
    print("-----------------")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.2f}")
    
    

    # Step 12: Save trained pipeline (preprocessing + model)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"\nModel saved successfully at: {MODEL_PATH}")


if __name__ == "__main__":
    main()
