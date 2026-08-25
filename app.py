from flask import Flask, render_template, request
import pandas as pd
import joblib

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

    math_prediction = None
    portuguese_prediction = None

    math_category = None
    portuguese_category = None

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

                "guardian_x": request.form["guardian_x"],
                "traveltime_x": int(request.form["traveltime_x"]),
                "studytime_x": int(request.form["studytime_x"]),
                "failures_x": int(request.form["failures_x"]),
                "schoolsup_x": request.form["schoolsup_x"],
                "famsup_x": request.form["famsup_x"],
                "paid_x": request.form["paid_x"],
                "activities_x": request.form["activities_x"],

                "nursery": request.form["nursery"],
                "higher_x": request.form["higher_x"],
                "internet": request.form["internet"],
                "romantic_x": request.form["romantic_x"],

                "famrel_x": int(request.form["famrel_x"]),
                "freetime_x": int(request.form["freetime_x"]),
                "goout_x": int(request.form["goout_x"]),
                "Dalc_x": int(request.form["Dalc_x"]),
                "Walc_x": int(request.form["Walc_x"]),
                "health_x": int(request.form["health_x"]),
                "absences_x": int(request.form["absences_x"]),
                "G1_x": int(request.form["G1_x"]),
                "G2_x": int(request.form["G2_x"]),

                "guardian_y": request.form["guardian_y"],
                "traveltime_y": int(request.form["traveltime_y"]),
                "studytime_y": int(request.form["studytime_y"]),
                "failures_y": int(request.form["failures_y"]),
                "schoolsup_y": request.form["schoolsup_y"],
                "famsup_y": request.form["famsup_y"],
                "paid_y": request.form["paid_y"],
                "activities_y": request.form["activities_y"],

                "higher_y": request.form["higher_y"],
                "romantic_y": request.form["romantic_y"],

                "famrel_y": int(request.form["famrel_y"]),
                "freetime_y": int(request.form["freetime_y"]),
                "goout_y": int(request.form["goout_y"]),
                "Dalc_y": int(request.form["Dalc_y"]),
                "Walc_y": int(request.form["Walc_y"]),
                "health_y": int(request.form["health_y"]),
                "absences_y": int(request.form["absences_y"]),
                "G1_y": int(request.form["G1_y"]),
                "G2_y": int(request.form["G2_y"])
            }


            # Create input DataFrame

            input_data = pd.DataFrame([student])


            # Make prediction

            predictions = model.predict(input_data)[0]

            print("Predictions:", predictions)


            # Mathematics

            math_prediction = round(
                max(0, min(20, predictions[0])),
                2
            )

            math_category = classify_performance(
                math_prediction
            )


            # Portuguese

            portuguese_prediction = round(
                max(0, min(20, predictions[1])),
                2
            )

            portuguese_category = classify_performance(
                portuguese_prediction
            )


            recommendation = (
                "Continue monitoring the student's progress "
                "and provide suitable academic support."
            )


        except Exception as exception:

            print("ERROR:", exception)

            error = str(exception)


    return render_template(
        "index.html",

        math_prediction=math_prediction,
        portuguese_prediction=portuguese_prediction,

        math_category=math_category,
        portuguese_category=portuguese_category,

        recommendation=recommendation,

        error=error
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
    )