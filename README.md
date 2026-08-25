
# Student Performance Prediction using Machine Learning

A simple AI/ML-based real-world application that predicts a student's final academic performance in **Mathematics and Portuguese** using demographic, family, social, study, and academic-related information.

## PR12: AI/ML-Based Real-World Application

**Selected Application:** Student Performance Prediction

This project follows the required PR12 stages:

1. Problem Definition
2. Dataset Collection
3. Data Preprocessing
4. Exploratory Data Analysis
5. Model Development
6. Performance Evaluation
7. Model Optimization
8. Result Analysis and Documentation

---

## Aim

To develop a machine learning system that predicts a student's final Mathematics and Portuguese grades and displays the results through a simple Flask web application.

---

## Objectives

- Collect and use student performance data.
- Combine Mathematics and Portuguese student datasets.
- Perform basic data preprocessing.
- Perform simple Exploratory Data Analysis (EDA).
- Train a machine learning regression model.
- Predict Mathematics and Portuguese final grades.
- Evaluate the model using MAE, MSE, RMSE, and R².
- Use 10-fold cross-validation to check model performance.
- Save the trained model for use in the Flask application.
- Display predicted grades and performance categories.
- Provide a basic academic recommendation.

---

# Dataset

The project uses the Student Performance dataset.

The dataset contains information about students including:

- Demographic information
- Family information
- School information
- Study habits
- Social activities
- Previous failures
- Absences
- Previous grades
- Other student-related attributes

The project uses Mathematics and Portuguese subject data.

### Dataset Files

```text
data/
├── student-mat.csv
├── student-por.csv
└── student-merge.csv
````

The two subject datasets are combined into:

```text
student-merge.csv
```

### Target Variables

The model predicts two final grades:

```text
G3_x → Mathematics Final Grade
G3_y → Portuguese Final Grade
```

Therefore, the project uses a **multi-output regression approach**.

---

# Exploratory Data Analysis

A simple Exploratory Data Analysis (EDA) was performed before model training.

The EDA includes:

* Dataset shape
* Dataset columns
* First five rows
* Dataset information
* Missing value checking
* Duplicate row checking
* Statistical summary
* Final grade analysis
* Grade distribution
* Study time vs final grade
* Previous failures vs final grade
* Absences vs final grade
* Correlation analysis

EDA was performed for both Mathematics and Portuguese.

### Mathematics EDA

The following features were analyzed:

```text
G3_x
studytime_x
failures_x
absences_x
```

Graphs include:

* Mathematics final grade distribution
* Study time vs Mathematics final grade
* Previous failures vs Mathematics final grade
* Absences vs Mathematics final grade

### Portuguese EDA

The following features were analyzed:

```text
G3_y
studytime_y
failures_y
absences_y
```

Graphs include:

* Portuguese final grade distribution
* Study time vs Portuguese final grade
* Previous failures vs Portuguese final grade
* Absences vs Portuguese final grade

### Correlation Analysis

A correlation heatmap is also generated using the numerical features.

The correlation of the features with:

```text
G3_x
```

and

```text
G3_y
```

is displayed to understand the relationship between numerical features and final grades.

---

# Machine Learning Approach

The project uses a:

**Random Forest Regressor**

Random Forest is a machine learning algorithm that combines multiple decision trees to make predictions.

The preprocessing and model are combined using a Scikit-learn `Pipeline`.

The model performs **multi-output regression**, allowing it to predict:

```text
Mathematics Final Grade
+
Portuguese Final Grade
```

---

# Data Preprocessing

The dataset contains both numerical and categorical features.

## Numerical Features

Missing numerical values are handled using:

```text
Median Imputation
```

The median value is used because it is less affected by extreme values.

## Categorical Features

Missing categorical values are handled using:

```text
Most Frequent Imputation
```

Categorical values are converted into numerical features using:

```text
OneHotEncoder
```

The encoder uses:

```python
handle_unknown="ignore"
```

This allows the model to handle unknown categorical values during prediction.

---

# Model Configuration

The Random Forest Regressor is configured as follows:

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

### Parameters

| Parameter             | Value |
| --------------------- | ----: |
| Number of Trees       |   200 |
| Maximum Depth         |    15 |
| Minimum Samples Split |     5 |
| Minimum Samples Leaf  |     2 |
| Random State          |    42 |

---

# Model Training

The dataset is divided into training and testing data.

```text
80% → Training Data
20% → Testing Data
```

The training process includes:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
10-Fold Cross-Validation
   ↓
Random Forest Training
   ↓
Model Evaluation
   ↓
Save Model
```

---

# Cross-Validation

The project uses **10-fold cross-validation**.

The training data is divided into 10 parts.

The model is trained and evaluated multiple times using different parts of the training dataset.

The R² score is used for cross-validation.

This provides a better understanding of how the model performs on different subsets of the data.

---

# Performance Evaluation

The model is evaluated using four metrics.

## MAE - Mean Absolute Error

MAE measures the average absolute difference between the actual and predicted grades.

A lower MAE indicates better prediction performance.

---

## MSE - Mean Squared Error

MSE measures the average squared difference between actual and predicted grades.

A lower MSE indicates better performance.

---

## RMSE - Root Mean Squared Error

RMSE is the square root of MSE.

It represents the prediction error in the same scale as the student's grade.

---

## R² Score

R² measures how well the model explains the variation in the target values.

A value closer to 1 indicates better performance.

---

# Model Performance

The Random Forest Regressor produced the following results on the test dataset:

| Metric                         | Result |
| ------------------------------ | -----: |
| Mean Absolute Error (MAE)      |  0.984 |
| Mean Squared Error (MSE)       |  3.060 |
| Root Mean Squared Error (RMSE) |  1.749 |
| R² Score                       |  0.768 |

### Performance Output

```text
Model Performance
-----------------
MAE:  0.984
MSE:  3.060
RMSE: 1.749
R²:   0.768
```

The R² score of **0.768** indicates that the model explains approximately **76.8% of the variation** in the test data.

The RMSE of **1.749** means that the prediction error is approximately 1.75 grade points on the 0–20 grading scale.

---

# System Working

```text
Student Details
       ↓
HTML Form
       ↓
Flask Application
       ↓
Request Data
       ↓
Pandas DataFrame
       ↓
Saved ML Pipeline
       ↓
Data Preprocessing
       ↓
Random Forest Regressor
       ↓
Predicted Grades
       ↓
Performance Categories
       ↓
Academic Recommendation
```

---

# Web Application

The project includes a simple Flask web application.

The user enters student information through an HTML form.

The form sends the information to the Flask server.

The trained machine learning model processes the input and predicts the student's final grades.

The results are then displayed on the webpage.

### Application Displays

* Mathematics predicted grade
* Portuguese predicted grade
* Mathematics performance category
* Portuguese performance category
* Basic academic recommendation

---

# Performance Categories

The predicted grades are classified into four categories.

| Predicted Grade | Category                 |
| --------------: | ------------------------ |
|          `< 10` | At Risk of Failing       |
|     `10 - < 12` | Pass - Needs Improvement |
|       `12 - 16` | Good Performance         |
|          `> 16` | Excellent Performance    |

---

# Project Structure

```text
student_performance_predict_model/
│
├── data/
│   ├── student-mat.csv
│   ├── student-por.csv
│   ├── student-merge.csv
│   ├── student-merge.R
│   ├── student.txt
│   └── student.zip
│
├── models/
│   └── student_performance_model.pkl
│
├── src/
│   ├── train_model.py
│   └── eda.ipynb
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── server.py
│
├── requirementes.txt
│
└── README.md
```

---

# Technologies Used

## Programming Language

* Python

## Machine Learning

* Pandas
* NumPy
* Scikit-learn
* Joblib

## Web Application

* Flask
* HTML
* CSS

## Data Visualization

* Matplotlib
* Seaborn

The required Python packages are listed in:

```text
requirementes.txt
```

---

# How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/ZarhanMemon/student_performance_predict_model.git
```

Move into the project directory:

```bash
cd student_performance_predict_model
```

---

## 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirementes.txt
```

---

## 3. Run Exploratory Data Analysis

Run the EDA script:

```bash
  src/eda.ipynb
```

The script displays:

* Dataset information
* Missing values
* Duplicate values
* Statistical summary
* Mathematics graphs
* Portuguese graphs
* Correlation heatmap

---

## 4. Train the Model

Run:

```bash
python src/train_model.py
```

The training script:

* Loads the dataset
* Performs preprocessing
* Splits the dataset
* Performs 10-fold cross-validation
* Trains the Random Forest model
* Evaluates the model
* Saves the trained model

The trained model is saved as:

```text
models/student_performance_model.pkl
```

---

## 5. Run the Flask Application

Start the Flask server:

```bash
python server.py
```

Open the local Flask address shown in the terminal in your browser.

---

# PR12 Requirements Mapping

| PR12 Requirement          | Project Implementation                               |
| ------------------------- | ---------------------------------------------------- |
| Problem Definition        | Student final grade prediction                       |
| Dataset Collection        | Mathematics and Portuguese student datasets          |
| Data Preprocessing        | Imputation and one-hot encoding                      |
| Exploratory Data Analysis | Basic statistical and graphical analysis             |
| Model Development         | Random Forest Regressor                              |
| Performance Evaluation    | MAE, MSE, RMSE and R²                                |
| Model Optimization        | Random Forest parameter configuration and 10-fold CV |
| Result Analysis           | Predicted grades and performance categories          |
| Documentation             | README and practical project report                  |

---

# Result

The Student Performance Prediction system was successfully developed using Python, Machine Learning, and Flask.

The Random Forest Regressor achieved the following results on the test dataset:

| Metric | Result |
| ------ | -----: |
| MAE    |  0.984 |
| MSE    |  3.060 |
| RMSE   |  1.749 |
| R²     |  0.768 |

The system can predict both:

```text
Mathematics Final Grade
Portuguese Final Grade
```

The Flask application provides a simple interface for entering student information and viewing the predicted results.

---

# Limitations

* The model depends on the quality and size of the available dataset.
* The dataset may not represent every student or educational environment.
* Predictions should not be treated as a final judgement of a student's ability.
* The recommendation system is intentionally simple.
* Model performance can vary with different datasets and train-test splits.
* The application is developed mainly for academic and demonstration purposes.

---

# Future Scope

The project can be improved by:

* Adding more recent student datasets.
* Comparing different machine learning algorithms.
* Adding feature importance visualization.
* Adding graphical student performance analysis.
* Improving the recommendation system.
* Adding more student-related features.
* Improving the web interface.
* Deploying the application online.

---

# Author

**Zarhan Memon**

GitHub:

[https://github.com/ZarhanMemon/student_performance_predict_model](https://github.com/ZarhanMemon/student_performance_predict_model)

---

# License

This project is developed for academic practical work under:


<img width="715" height="727" alt="Screenshot 2026-08-25 192042" src="https://github.com/user-attachments/assets/e30ac728-f63a-45e5-87ff-953f7efba6b2" />

<img width="391" height="803" alt="Screenshot 2026-08-25 190029" src="https://github.com/user-attachments/assets/3912cbc2-ea6b-439d-8fbc-684684387d6e" />


<img width="397" height="789" alt="Screenshot 2026-08-25 190047" src="https://github.com/user-attachments/assets/fffad637-5837-4b0a-9f15-08d46bc4bc38" />


