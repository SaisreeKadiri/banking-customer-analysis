# ============================================================
# MACHINE LEARNING-BASED BANKING CUSTOMER ANALYSIS
# ============================================================


# Import pandas for handling the dataset
import pandas as pd

# Import train_test_split to divide data into training and testing
from sklearn.model_selection import train_test_split

# Import classification algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Import regression algorithm
from sklearn.ensemble import RandomForestRegressor

# Import evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Import K-Means clustering
from sklearn.cluster import KMeans

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Import joblib to save trained models
import joblib


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Read the banking dataset
df = pd.read_csv("data/loan_approval_dataset.csv")


# Remove extra spaces from column names
df.columns = df.columns.str.strip()


# Remove extra spaces from text values
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)


# Display cleaned columns
print("Cleaned columns:")
print(df.columns)


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

# Convert education into numbers
# Graduate = 1
# Not Graduate = 0

df["education"] = df["education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})


# Convert self-employed into numbers
# Yes = 1
# No = 0

df["self_employed"] = df["self_employed"].map({
    "Yes": 1,
    "No": 0
})


# Convert loan status into numbers
# Approved = 1
# Rejected = 0

df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})


# ============================================================
# 3. CREATE TOTAL ASSETS
# ============================================================

# Add all four asset columns together
# to create one total asset value.

df["total_assets"] = (
    df["residential_assets_value"]
    + df["commercial_assets_value"]
    + df["luxury_assets_value"]
    + df["bank_asset_value"]
)


# Display some total asset values
print("\nTotal Assets:")
print(df["total_assets"].head())


# ============================================================
# 4. LOAN APPROVAL CLASSIFICATION
# ============================================================

# Select only the features that the customer
# will provide through the website.

classification_features = [
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "total_assets"
]


# Create input data
X = df[classification_features]


# Create target variable
y = df["loan_status"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create Random Forest classification model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Train the classification model
model.fit(X_train, y_train)


# Predict loan approval for test data
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


# Display accuracy
print("\nRandom Forest Classification Results:")
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100)


# Save classification model
joblib.dump(
    model,
    "models/loan_approval_model.pkl"
)


print("Loan approval model saved successfully!")


# ============================================================
# 5. LOAN AMOUNT REGRESSION
# ============================================================

# Features used to predict loan amount
# Requested loan amount is NOT included because
# loan_amount is the target we are predicting.

regression_features = [
    "education",
    "self_employed",
    "income_annum",
    "loan_term",
    "cibil_score",
    "total_assets"
]


# Create regression input
X_reg = df[regression_features]


# Target = loan amount
y_reg = df["loan_amount"]


# Split regression data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


# Create Random Forest Regression model
regression_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=2
)


# Train regression model
regression_model.fit(
    X_train_reg,
    y_train_reg
)


# Predict loan amounts
y_pred_reg = regression_model.predict(X_test_reg)


# Calculate MAE
mae = mean_absolute_error(
    y_test_reg,
    y_pred_reg
)


# Calculate MSE
mse = mean_squared_error(
    y_test_reg,
    y_pred_reg
)


# Calculate RMSE
rmse = mse ** 0.5


# Calculate R2 score
r2 = r2_score(
    y_test_reg,
    y_pred_reg
)


# Display regression results
print("\nLoan Amount Regression Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# Save regression model
joblib.dump(
    regression_model,
    "models/loan_amount_model.pkl"
)


print("Loan amount model saved successfully!")


# ============================================================
# 6. CUSTOMER SEGMENTATION
# ============================================================

# Select customer financial information
# for K-Means clustering.

X_cluster = df[
    [
        "income_annum",
        "loan_amount",
        "cibil_score",
        "total_assets"
    ]
]


# Create StandardScaler
scaler = StandardScaler()


# Scale clustering data
X_cluster_scaled = scaler.fit_transform(
    X_cluster
)


# Create K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)


# Train K-Means and assign clusters
clusters = kmeans.fit_predict(
    X_cluster_scaled
)


# Add cluster number to dataset
df["customer_cluster"] = clusters


# Display cluster counts
print("\nCustomer Segmentation Results:")
print(
    df["customer_cluster"]
    .value_counts()
    .sort_index()
)


# Save K-Means model
joblib.dump(
    kmeans,
    "models/customer_segmentation.pkl"
)


# Save scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)


print("Customer segmentation model saved successfully!")


# ============================================================
# 7. GRADIENT BOOSTING CLASSIFICATION
# ============================================================

# Create Gradient Boosting model
gradient_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)


# Train Gradient Boosting model
gradient_model.fit(
    X_train,
    y_train
)


# Make predictions
y_pred_gradient = gradient_model.predict(
    X_test
)


# Calculate accuracy
gradient_accuracy = accuracy_score(
    y_test,
    y_pred_gradient
)


# Display accuracy
print("\nGradient Boosting Accuracy:")
print(gradient_accuracy)
print(
    "Accuracy Percentage:",
    gradient_accuracy * 100
)


# Save Gradient Boosting model
joblib.dump(
    gradient_model,
    "models/gradient_boosting_model.pkl"
)


print("Gradient Boosting model saved successfully!")


# ============================================================
# 8. TEST DIFFERENT CUSTOMER PROFILES
# ============================================================

print("\n========================================")
print("TESTING LOAN AMOUNT PREDICTIONS")
print("========================================")


# Customer profile 1
customer_1 = pd.DataFrame([{
    "education": 1,
    "self_employed": 0,
    "income_annum": 3000000,
    "loan_term": 10,
    "cibil_score": 600,
    "total_assets": 5000000
}])


# Customer profile 2
customer_2 = pd.DataFrame([{
    "education": 1,
    "self_employed": 0,
    "income_annum": 10000000,
    "loan_term": 15,
    "cibil_score": 750,
    "total_assets": 20000000
}])


# Customer profile 3
customer_3 = pd.DataFrame([{
    "education": 1,
    "self_employed": 1,
    "income_annum": 20000000,
    "loan_term": 20,
    "cibil_score": 850,
    "total_assets": 50000000
}])


# Predict loan amounts
prediction_1 = regression_model.predict(customer_1)[0]
prediction_2 = regression_model.predict(customer_2)[0]
prediction_3 = regression_model.predict(customer_3)[0]


# Display predictions
print("\nCustomer 1 Estimated Loan:")
print(prediction_1)


print("\nCustomer 2 Estimated Loan:")
print(prediction_2)


print("\nCustomer 3 Estimated Loan:")
print(prediction_3)


print("\n========================================")
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("========================================")