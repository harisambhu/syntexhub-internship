"""Iris EDA, model comparison, evaluation, and prediction CLI."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
FEATURES = ["sepal length", "sepal width", "petal length", "petal width"]
TARGET = "species"


def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load the built-in Iris dataset into a tidy DataFrame."""
    iris = load_iris(as_frame=True)
    frame = iris.data.rename(
        columns={
            "sepal length (cm)": FEATURES[0],
            "sepal width (cm)": FEATURES[1],
            "petal length (cm)": FEATURES[2],
            "petal width (cm)": FEATURES[3],
        }
    )
    frame[TARGET] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return frame, list(iris.target_names)


def run_eda(frame: pd.DataFrame, class_names: list[str]) -> None:
    """Print basic EDA and save feature-pair visualizations."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Rows: {len(frame)} | Columns: {len(frame.columns)}")
    print(f"Missing values: {int(frame.isna().sum().sum())}")
    print("\nClass counts:")
    print(frame[TARGET].value_counts().sort_index().to_string())
    print("\nFeature summary:")
    print(frame[FEATURES].describe().round(2).to_string())

    pair_plot = sns.pairplot(
        frame,
        vars=FEATURES,
        hue=TARGET,
        hue_order=class_names,
        corner=True,
        diag_kind="hist",
        plot_kws={"alpha": 0.75, "s": 42},
    )
    pair_plot.fig.suptitle("Iris Feature Pairs", y=1.02)
    pair_plot.savefig(OUTPUT_DIR / "feature_pairs.png", dpi=160, bbox_inches="tight")
    plt.close(pair_plot.fig)
    print(f"\nSaved feature-pair plot: {OUTPUT_DIR / 'feature_pairs.png'}")


def build_models() -> dict[str, object]:
    return {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    }


def save_confusion_matrix(matrix, model_name: str, class_names: list[str]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_title(f"{model_name.title()} Confusion Matrix")
    axis.set_xlabel("Predicted species")
    axis.set_ylabel("Actual species")
    figure.tight_layout()
    path = OUTPUT_DIR / f"confusion_matrix_{model_name}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def train_and_evaluate(frame: pd.DataFrame, class_names: list[str]) -> None:
    """Train both classifiers, compare accuracy, and save artifacts."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    features_train, features_test, target_train, target_test = train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=frame[TARGET],
    )

    results: dict[str, float] = {}
    for name, model in build_models().items():
        model.fit(features_train, target_train)
        predictions = model.predict(features_test)
        accuracy = accuracy_score(target_test, predictions)
        results[name] = accuracy
        matrix = confusion_matrix(target_test, predictions, labels=class_names)
        matrix_path = save_confusion_matrix(matrix, name, class_names)
        print(f"\n{name.title()} accuracy: {accuracy:.3f}")
        print(classification_report(target_test, predictions, labels=class_names, zero_division=0))
        print("Confusion matrix (rows = actual, columns = predicted):")
        print(pd.DataFrame(matrix, index=class_names, columns=class_names).to_string())
        print(f"Saved confusion matrix: {matrix_path}")
        with (OUTPUT_DIR / f"{name}_model.pkl").open("wb") as model_file:
            pickle.dump(model, model_file)

    best_model = max(results, key=results.get)
    print("\nAccuracy comparison:")
    for name, accuracy in results.items():
        print(f"- {name.title()}: {accuracy:.3f}")
    print(f"Best model: {best_model.title()}")
    print("Misclassification note: nonzero off-diagonal cells show which species were confused.")
    print(f"Models saved in: {OUTPUT_DIR.resolve()}")


def predict_species(values: list[float], model_name: str) -> None:
    """Predict one species from four measurements."""
    model_path = OUTPUT_DIR / f"{model_name}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist. Run `python iris_classifier.py train` first."
        )
    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)
    prediction = model.predict(pd.DataFrame([values], columns=FEATURES))[0]
    print(f"Predicted species: {prediction}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iris classification workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("eda", help="Print EDA and save a feature-pair plot")
    subparsers.add_parser("train", help="Train classifiers and save evaluation artifacts")
    predict_parser = subparsers.add_parser("predict", help="Predict a species for new measurements")
    predict_parser.add_argument("measurements", nargs=4, type=float, metavar="VALUE")
    predict_parser.add_argument("--model", choices=["logistic", "tree"], default="logistic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, class_names = load_data()
    if args.command == "eda":
        run_eda(frame, class_names)
    elif args.command == "train":
        train_and_evaluate(frame, class_names)
    else:
        predict_species(args.measurements, args.model)


if __name__ == "__main__":
    main()
