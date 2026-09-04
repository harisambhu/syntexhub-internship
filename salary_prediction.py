import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# 1. HELPER FUNCTION
# ==========================================

def _to_dense(x):
    """Convert sparse matrix to dense array when needed."""
    return x.toarray() if hasattr(x, "toarray") else x


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data(path):
    """Load CSV file."""
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Failed to read '{path}': {exc}")
        sys.exit(1)


# ==========================================
# 3. TRAIN AND EVALUATE MODELS
# ==========================================

def build_and_evaluate(df):

    # ---------- SINGLE FEATURE MODEL ----------

    if "Experience" not in df.columns:
        raise KeyError("Column 'Experience' is missing from dataset.")

    if "Salary" not in df.columns:
        raise KeyError("Column 'Salary' is missing from dataset.")

    X_single = df[["Experience"]]
    y = df["Salary"]

    X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
        X_single,
        y,
        test_size=0.2,
        random_state=42
    )

    single_model = LinearRegression()
    single_model.fit(X_train_single, y_train_single)

    single_predictions = single_model.predict(X_test_single)

    single_rmse = mean_squared_error(
        y_test_single,
        single_predictions
    ) ** 0.5

    single_r2 = r2_score(
        y_test_single,
        single_predictions
    )

    # ---------- MULTIPLE FEATURE MODEL ----------

    X_multiple = df.drop("Salary", axis=1)
    y = df["Salary"]

    categorical_features = X_multiple.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    to_dense = FunctionTransformer(
        _to_dense,
        accept_sparse=True
    )

    X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(
        X_multiple,
        y,
        test_size=0.2,
        random_state=42
    )

    multiple_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("to_dense", to_dense),
            ("regressor", LinearRegression())
        ]
    )

    multiple_model.fit(
        X_train_multiple,
        y_train_multiple
    )

    multiple_predictions = multiple_model.predict(
        X_test_multiple
    )

    multiple_rmse = mean_squared_error(
        y_test_multiple,
        multiple_predictions
    ) ** 0.5

    multiple_r2 = r2_score(
        y_test_multiple,
        multiple_predictions
    )

    return {
        "single": {
            "model": single_model,
            "rmse": single_rmse,
            "r2": single_r2,
            "X_test": X_test_single,
            "y_test": y_test_single,
            "predictions": single_predictions
        },
        "multiple": {
            "model": multiple_model,
            "rmse": multiple_rmse,
            "r2": multiple_r2,
            "features": X_multiple.columns.tolist(),
            "X_test": X_test_multiple,
            "y_test": y_test_multiple,
            "predictions": multiple_predictions
        }
    }


# ==========================================
# 4. MAIN PROGRAM
# ==========================================

def main():

    # ---------- LOAD DATA ----------

    src = "salary_prediction_big_dataset.csv"

    df = load_data(src)

    print("\n========== DATASET INFORMATION ==========")
    print("First 5 rows:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nDataset Shape:")
    print(df.shape)

    # ---------- CLEAN DATA ----------

    df = df.dropna()

    print("\n========== AFTER CLEANING ==========")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # ---------- DATA TYPES ----------

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    numerical_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print("\n========== NUMERICAL FEATURES ==========")
    print(numerical_features)

    print("\n========== CATEGORICAL FEATURES ==========")
    print(categorical_features)

    # ---------- TRAIN MODELS ----------

    results = build_and_evaluate(df)

    single_rmse = results["single"]["rmse"]
    single_r2 = results["single"]["r2"]

    multiple_rmse = results["multiple"]["rmse"]
    multiple_r2 = results["multiple"]["r2"]

    # ---------- SINGLE FEATURE RESULTS ----------

    print("\n========== SINGLE FEATURE MODEL ==========")
    print("Feature: Experience")
    print("RMSE:", single_rmse)
    print("R2 Score:", single_r2)

    # ---------- MULTIPLE FEATURE RESULTS ----------

    print("\n========== MULTIPLE FEATURE MODEL ==========")
    print("Features:")
    print(results["multiple"]["features"])
    print("RMSE:", multiple_rmse)
    print("R2 Score:", multiple_r2)

    # ---------- MODEL COMPARISON ----------

    print("\n========== MODEL COMPARISON ==========")
    print("Single Feature RMSE:", single_rmse)
    print("Multiple Feature RMSE:", multiple_rmse)
    print("Single Feature R2:", single_r2)
    print("Multiple Feature R2:", multiple_r2)

    # ---------- SELECT BEST MODEL ----------

    if multiple_rmse < single_rmse:
        best_model = results["multiple"]["model"]
        best_model_name = "Multiple Feature Model"
    else:
        best_model = results["single"]["model"]
        best_model_name = "Single Feature Model"

    print("\n========== BEST MODEL ==========")
    print("Best Model:", best_model_name)

    # ---------- SAVE BEST MODEL ----------

    joblib.dump(
        best_model,
        "best_salary_model.pkl"
    )

    print("\nBest model saved successfully!")
    print("File: best_salary_model.pkl")

    # ==========================================
    # 5. SAMPLE SALARY PREDICTION
    # ==========================================

    new_employee = pd.DataFrame({
        "Experience": [5],
        "Test Score": [80],
        "Interview Score": [85],
        "Education": ["Bachelor"],
        "Department": ["IT"],
        "Gender": ["Male"]
    })

    if best_model_name == "Single Feature Model":
        input_df = new_employee[["Experience"]]
    else:
        input_df = new_employee

    predicted_salary = best_model.predict(input_df)

    print("\n========== SALARY PREDICTION ==========")
    print("Employee Details:")
    print(new_employee)

    print("\nPredicted Salary:")
    print("Predicted Salary: ₹", round(float(predicted_salary[0]), 2))

    # ==========================================
    # 6. VISUALIZATION
    # ==========================================

    # ---------- ACTUAL VS PREDICTED SALARY ----------

    y_test_multiple = results["multiple"]["y_test"]
    multiple_predictions = results["multiple"]["predictions"]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test_multiple,
        multiple_predictions
    )

    # Perfect prediction line
    plt.plot(
        [y_test_multiple.min(), y_test_multiple.max()],
        [y_test_multiple.min(), y_test_multiple.max()]
    )

    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")

    plt.tight_layout()
    plt.show()

    # ---------- RMSE COMPARISON ----------

    models = [
        "Single Feature",
        "Multiple Feature"
    ]

    rmse_values = [
        single_rmse,
        multiple_rmse
    ]

    plt.figure(figsize=(8, 6))

    plt.bar(
        models,
        rmse_values
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Model RMSE Comparison")

    plt.tight_layout()
    plt.show()

    # ---------- R2 SCORE COMPARISON ----------

    r2_values = [
        single_r2,
        multiple_r2
    ]

    plt.figure(figsize=(8, 6))

    plt.bar(
        models,
        r2_values
    )

    plt.xlabel("Model")
    plt.ylabel("R² Score")
    plt.title("Model R² Score Comparison")

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 7. PROJECT COMPLETION
    # ==========================================

    print("\n==========================================")
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("==========================================")


# ==========================================
# 8. RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    main()