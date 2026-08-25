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

    os.makedirs("models", exist_ok=True)

    data = pd.read_csv("data/student-merge.csv")


    data["G3_x"] = pd.to_numeric(data["G3_x"], errors="coerce")
    
    data = data.dropna(subset=["G3_x"])

    print("Dataset shape:", data.shape)



    y = data[["G3_x","G3_y"]]
    X = data.drop(columns=["G3_x","id","G3_y"])
    
    

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()



    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()
    
    

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]
    )
    
    

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ]
    )
    
    

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
    
    

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    
    
    print(X_train.columns.to_list())

    cv_scores = cross_val_score(
        model_pipeline,
        X_train,
        y_train,
        cv=10,
        scoring="r2"
    )

    print("\nCross-validation R² scores:")
    print(cv_scores)


    print(f"Mean CV R²: {cv_scores.mean():.3f}")



    print("\nTraining model...")

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)



    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(predictions)

    print("\nModel Performance")
    print("-----------------")
    print(f"MAE:  {mae:.3f}")
    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")



    joblib.dump(model_pipeline, MODEL_PATH)

    print(f"\nModel saved successfully at: {MODEL_PATH}")



if __name__ == "__main__":
    main()