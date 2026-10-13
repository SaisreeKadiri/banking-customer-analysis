# banking-customer-analysis
# Machine Learning-Based Banking Customer Analysis

## About the Project
A machine learning-based banking application that predicts loan approval and estimates loan amounts using customer financial data. It also performs customer segmentation using K-Means clustering and integrates the trained models into a Flask web application.

## Features
- Loan approval prediction using Random Forest Classification.
- Loan amount estimation using Random Forest Regression.
- Customer segmentation using K-Means Clustering.
- Model comparison using Gradient Boosting.
- Flask web application for entering customer details and viewing predictions.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- HTML and CSS
- Machine Learning

## Dataset
The project uses a banking loan approval dataset containing 4,269 records and 13 attributes, including customer income, education, employment status, loan details, CIBIL score, and asset values.

## Project Structure
```text
banking-customer-analysis/
├── app.py
├── train_models.py
├── data/
│   └── loan_approval_dataset.csv
├── models/
├── templates/
├── static/
└── README.md
```

## How to Run
1. Clone or download this repository.
2. Install the required Python libraries.
3. Train the models if the trained model files are not included.
4. Run the Flask application using:

```bash
python app.py
```

5. Open the local URL displayed in the terminal in your browser.

## Disclaimer
The predictions are intended for educational and analytical purposes only and should not be considered actual banking or lending decisions.
