from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


# -------------------------------------------------
# LOAD TRAINED MODELS
# -------------------------------------------------

loan_approval_model = joblib.load(
    "models/loan_approval_model.pkl"
)

loan_amount_model = joblib.load(
    "models/loan_amount_model.pkl"
)


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form

    name = request.form["name"]

    education = int(request.form["education"])

    self_employed = int(request.form["self_employed"])

    income_annum = float(
        request.form["income_annum"]
    )

    loan_amount = float(
        request.form["loan_amount"]
    )

    loan_term = float(
        request.form["loan_term"]
    )

    cibil_score = float(
        request.form["cibil_score"]
    )

    total_assets = float(
        request.form["total_assets"]
    )


    # -------------------------------------------------
    # DATA FOR LOAN APPROVAL MODEL
    # -------------------------------------------------

    approval_data = pd.DataFrame([{

        "education": education,

        "self_employed": self_employed,

        "income_annum": income_annum,

        "loan_amount": loan_amount,

        "loan_term": loan_term,

        "cibil_score": cibil_score,

        "total_assets": total_assets

    }])


    # Keep exactly the same column order
    # used during model training

    approval_data = approval_data[
        [
            "education",
            "self_employed",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "total_assets"
        ]
    ]


    # -------------------------------------------------
    # LOAN APPROVAL PREDICTION
    # -------------------------------------------------

    approval_prediction = loan_approval_model.predict(
        approval_data
    )[0]


    # -------------------------------------------------
    # DATA FOR LOAN AMOUNT MODEL
    # -------------------------------------------------

    amount_data = pd.DataFrame([{

        "education": education,

        "self_employed": self_employed,

        "income_annum": income_annum,

        "loan_term": loan_term,

        "cibil_score": cibil_score,

        "total_assets": total_assets

    }])


    # Same order as training

    amount_data = amount_data[
        [
            "education",
            "self_employed",
            "income_annum",
            "loan_term",
            "cibil_score",
            "total_assets"
        ]
    ]


    # -------------------------------------------------
    # LOAN AMOUNT PREDICTION
    # -------------------------------------------------

    predicted_loan_amount = loan_amount_model.predict(
        amount_data
    )[0]


    # Prevent negative value

    if predicted_loan_amount < 0:

        predicted_loan_amount = 0


    # -------------------------------------------------
    # RESULT
    # -------------------------------------------------

    if approval_prediction == 1:

        approval_result = "Loan Approved"

    else:

        approval_result = "Loan Not Approved"


    # -------------------------------------------------
    # SEND RESULT PAGE
    # -------------------------------------------------

    return render_template(
        "result.html",

        name=name,

        approval_result=approval_result,

        predicted_loan_amount=predicted_loan_amount
    )


# -------------------------------------------------
# RUN FLASK
# -------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)