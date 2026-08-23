# Student Performance Prediction using Machine Learning

A simple AI/ML-based real-world application that predicts a student's final academic grade (`G3`) using student demographic, social, family, study, and academic-related attributes.

## PR12: AI/ML-Based Real-World Application

**Selected Application:** Student Performance Prediction

This project follows the required PR12 stages:

1. Problem Definition
2. Dataset Collection
3. Data Preprocessing
4. Model Development
5. Performance Evaluation
6. Model Optimization
7. Result Analysis and Documentation

## Aim

To develop a machine learning system that predicts a student's final performance and presents the predicted grade through a simple web application.

## Objectives

* Collect and use student performance data.
* Clean and preprocess numerical and categorical data.
* Train a machine learning regression model.
* Predict the final grade (`G3`) of a student.
* Evaluate the model using MAE, MSE, RMSE and R².
* Use cross-validation to check model performance.
* Save the trained model for use in the Flask application.
* Display a performance category and a basic recommendation to the user.

## Dataset

The project uses the Student Performance dataset containing two CSV files:

* `data/student-mat.csv` - Mathematics subject data
* `data/student-por.csv` - Portuguese language subject data

The two datasets are combined before model training. The target variable is `G3`, the student's final grade.

## Machine Learning Approach

The project uses a **Random Forest Regressor**. The preprocessing and model are combined into a Scikit-learn `Pipeline`.

### Data Preprocessing

* Numerical missing values are handled using median imputation.
* Numerical features are standardized using `StandardScaler`.
* Categorical missing values are handled using most-frequent imputation.
* Categorical features are converted using `OneHotEncoder`.
* Unknown categorical values are ignored during prediction.

### Model Configuration

The current Random Forest model uses:

* `n_estimators = 800`
* `max_depth = 30`
* `min_samples_split = 5`
* `min_samples_leaf = 2`
* `random_state = 42`

## Performance Evaluation

The model is evaluated using:

### MAE – Mean Absolute Error

Measures the average absolute difference between the actual and predicted grades.

### MSE – Mean Squared Error

Measures the average squared difference between actual and predicted grades.

### RMSE – Root Mean Squared Error

The square root of MSE. It represents the prediction error in the same scale as the grade.

### R² Score

Shows how well the model explains the variation in the final student grade.

### 10-Fold Cross-Validation

The project also uses 10-fold cross-validation to obtain a more reliable estimate of model performance.

> The exact metric values should be taken from the output of `train_model.py` after running the model.

## Project Structure

```text
student_performance_predict_model/
│
├── data/
│   ├── student-mat.csv
│   ├── student-por.csv
│   ├── student-merge.R
│   ├── student.txt
│   └── student.zip
│
├── models/
│   └── student_performance_model.pkl
│
├── src/
│   └── train_model.py
│
├── static/
│   └── ...
│
├── templates/
│   └── index.html
│
├── server.py
├── requirementes.txt
└── README.md
```

## System Working

```text
Student Details
      ↓
DataFrame Creation
      ↓
Data Preprocessing
      ↓
Random Forest Regressor
      ↓
Predicted Final Grade (G3)
      ↓
Performance Category
      ↓
Academic Recommendation
```

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/ZarhanMemon/student_performance_predict_model.git
cd student_performance_predict_model
```

### 2. Install Dependencies

```bash
pip install -r requirementes.txt
```

### 3. Train the Model

```bash
python src/train_model.py
```

The trained model will be saved as:

```text
models/student_performance_model.pkl
```

### 4. Run the Flask Application

```bash
python server.py
```

Open the local Flask address in your browser.

## Web Application

The application accepts student information through a form and displays:

* Predicted final grade out of 20
* Performance category
* Basic academic recommendation

### Performance Categories

| Predicted Grade | Category                 |
| --------------: | ------------------------ |
|          `< 10` | At Risk of Failing       |
|     `10 - < 12` | Pass - Needs Improvement |
|       `12 - 16` | Good Performance         |
|          `> 16` | Excellent Performance    |

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Flask
* HTML
* CSS
* Matplotlib
* Seaborn

The required Python packages are listed in `requirementes.txt`.

## PR12 Requirements Mapping

| PR12 Requirement       | Project Implementation                             |
| ---------------------- | -------------------------------------------------- |
| Problem Definition     | Student final grade prediction                     |
| Dataset Collection     | Student Mathematics and Portuguese datasets        |
| Data Preprocessing     | Imputation, scaling and one-hot encoding           |
| Model Development      | Random Forest Regressor                            |
| Performance Evaluation | MAE, MSE, RMSE and R²                              |
| Model Optimization     | Configured Random Forest parameters and 10-fold CV |
| Result Analysis        | Predicted grade and performance category           |
| Documentation          | README and practical journal report                |

## Result

The Student Performance Prediction system was successfully developed using
Python and Machine Learning.

The Random Forest Regressor produced the following results on the test dataset:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.94 |
| Mean Squared Error (MSE) | 2.87 |
| Root Mean Squared Error (RMSE) | 1.69 |
| R² Score | 0.81 |

### Model Performance
-----------------
MAE:  0.94 |
MSE:  2.87 |
RMSE: 1.69 |
R2:   0.81

## Limitations

* The model is based on the available student dataset.
* Predictions should not be treated as a final judgement of a student's ability.
* The current recommendation system is intentionally simple.
* Model performance should be interpreted using the actual training output.

## Future Scope

* Add more recent student datasets.
* Compare multiple machine learning algorithms.
* Add graphical analysis of student performance.
* Improve academic recommendations.
* Add feature importance visualization.
* Deploy the application online.

## Author

**Zarhan Memon**

GitHub: [ZarhanMemon](https://github.com/ZarhanMemon)

## License

This project is developed for academic practical work under **PR12: AI/ML-Based Real-World Application**.

