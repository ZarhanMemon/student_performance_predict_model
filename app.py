from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

model = joblib.load(
    "models/student_performance_model.pkl"
)


def classify_performance(grade):
    if grade < 10:
        return "At Risk of Failing"
    elif grade < 12:
        return "Pass - Needs Improvement"
    elif grade <= 16:
        return "Good Performance"
    else:
        return "Excellent Performance"


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    category = None
    recommendation = None
    error = None

    if request.method == "POST":
        try:
            student = {
                "school": request.form["school"],
                "sex": request.form["sex"],
                "age": int(request.form["age"]),
                "address": request.form["address"],
                "famsize": request.form["famsize"],
                "Pstatus": request.form["Pstatus"],
                "Medu": int(request.form["Medu"]),
                "Fedu": int(request.form["Fedu"]),
                "Mjob": request.form["Mjob"],
                "Fjob": request.form["Fjob"],
                "reason": request.form["reason"],
                "guardian": request.form["guardian"],
                "traveltime": int(request.form["traveltime"]),
                "studytime": int(request.form["studytime"]),
                "failures": int(request.form["failures"]),
                "schoolsup": request.form["schoolsup"],
                "famsup": request.form["famsup"],
                "paid": request.form["paid"],
                "activities": request.form["activities"],
                "nursery": request.form["nursery"],
                "higher": request.form["higher"],
                "internet": request.form["internet"],
                "romantic": request.form["romantic"],
                "famrel": int(request.form["famrel"]),
                "freetime": int(request.form["freetime"]),
                "goout": int(request.form["goout"]),
                "Dalc": int(request.form["Dalc"]),
                "Walc": int(request.form["Walc"]),
                "health": int(request.form["health"]),
                "absences": int(request.form["absences"])
            }

            input_data = pd.DataFrame([student])

            predicted_grade = model.predict(input_data)[0]

            prediction = round(
                max(0, min(20, predicted_grade)),
                2
            )

            category = classify_performance(prediction)

            recommendation = (
                "Continue monitoring the student's progress "
                "and provide suitable academic support."
            )

        except Exception as exception:
            error = str(exception)

    return render_template(
        "index.html",
        prediction=prediction,
        category=category,
        recommendation=recommendation,
        error=error
    )



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)