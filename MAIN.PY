# ==========================================
# SALARY PREDICTION USING LINEAR REGRESSION
# ==========================================
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "salary_prediction_dataset.csv"
MODEL_PATH = BASE_DIR / "salary_prediction_model.pkl"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = ["experience", "test_score", "education", "location", "salary"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    print("\n========== DATASET ==========")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    df = df.dropna().copy()
    if df.empty:
        raise ValueError("No valid data remains after removing missing values.")

    X = df.drop("salary", axis=1)
    y = df["salary"]

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print("\n========== FEATURES ==========")
    print("Numerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining Data:", X_train.shape)
    print("Testing Data:", X_test.shape)

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

    multiple_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regression", LinearRegression())
        ]
    )

    multiple_model.fit(X_train, y_train)
    multiple_prediction = multiple_model.predict(X_test)
    multiple_rmse = np.sqrt(mean_squared_error(y_test, multiple_prediction))
    multiple_r2 = r2_score(y_test, multiple_prediction)

    print("\n========== MULTIPLE LINEAR REGRESSION ==========")
    print("RMSE:", round(multiple_rmse, 2))
    print("R2 Score:", round(multiple_r2, 4))

    X_experience = df[["experience"]]
    X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
        X_experience,
        y,
        test_size=0.20,
        random_state=42
    )

    experience_model = LinearRegression()
    experience_model.fit(X_train_exp, y_train_exp)
    experience_prediction = experience_model.predict(X_test_exp)
    experience_rmse = np.sqrt(mean_squared_error(y_test_exp, experience_prediction))
    experience_r2 = r2_score(y_test_exp, experience_prediction)

    print("\n========== EXPERIENCE ONLY ==========")
    print("RMSE:", round(experience_rmse, 2))
    print("R2 Score:", round(experience_r2, 4))

    X_score = df[["test_score"]]
    X_train_score, X_test_score, y_train_score, y_test_score = train_test_split(
        X_score,
        y,
        test_size=0.20,
        random_state=42
    )

    score_model = LinearRegression()
    score_model.fit(X_train_score, y_train_score)
    score_prediction = score_model.predict(X_test_score)
    score_rmse = np.sqrt(mean_squared_error(y_test_score, score_prediction))
    score_r2 = r2_score(y_test_score, score_prediction)

    print("\n========== TEST SCORE ONLY ==========")
    print("RMSE:", round(score_rmse, 2))
    print("R2 Score:", round(score_r2, 4))

    results = pd.DataFrame({
        "Model": ["Experience Only", "Test Score Only", "Multiple Features"],
        "RMSE": [experience_rmse, score_rmse, multiple_rmse],
        "R2": [experience_r2, score_r2, multiple_r2]
    })

    print("\n========== MODEL COMPARISON ==========")
    print(results.round(2))

    best_index = results["R2"].idxmax()
    best_model_name = results.loc[best_index, "Model"]

    print("\n========== BEST MODEL ==========")
    print("Best Model:", best_model_name)

    if best_model_name == "Experience Only":
        joblib.dump(experience_model, MODEL_PATH)
    elif best_model_name == "Test Score Only":
        joblib.dump(score_model, MODEL_PATH)
    else:
        joblib.dump(multiple_model, MODEL_PATH)

    print("\nModel saved successfully!")
    print(f"File: {MODEL_PATH.name}")

    new_employee = pd.DataFrame({
        "experience": [4.0],
        "test_score": [85],
        "education": ["MCA"],
        "location": ["Bengaluru"]
    })

    predicted_salary = multiple_model.predict(new_employee)[0]

    print("\n========== SALARY PREDICTION ==========")
    print("Experience: 4 years")
    print("Test Score: 85")
    print("Education: MCA")
    print("Location: Bengaluru")
    print("Predicted Salary:", round(predicted_salary, 2))

    print("\n===================================")
    print("SALARY PREDICTION PROJECT COMPLETED")
    print("===================================")

    plt.figure(figsize=(8, 5))
    plt.scatter(df["experience"], df["salary"])
    plt.xlabel("Experience (Years)")
    plt.ylabel("Salary")
    plt.title("Experience vs Salary")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "experience_vs_salary.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["test_score"], df["salary"])
    plt.xlabel("Test Score")
    plt.ylabel("Salary")
    plt.title("Test Score vs Salary")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "test_score_vs_salary.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, multiple_prediction)
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "actual_vs_predicted_salary.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(results["Model"], results["R2"])
    plt.xlabel("Model")
    plt.ylabel("R2 Score")
    plt.title("R2 Score Comparison")
    plt.xticks(rotation=15)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "r2_score_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(results["Model"], results["RMSE"])
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison")
    plt.xticks(rotation=15)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "rmse_comparison.png", dpi=150)
    plt.close()

    print("\nGenerated graph files:")
    for filename in [
        "experience_vs_salary.png",
        "test_score_vs_salary.png",
        "actual_vs_predicted_salary.png",
        "r2_score_comparison.png",
        "rmse_comparison.png",
    ]:
        print(f"- {BASE_DIR / filename}")

    return predicted_salary


if __name__ == "__main__":
    main()