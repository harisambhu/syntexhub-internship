# Iris Flower Classification

A small, reproducible machine-learning workflow for the Iris dataset. It includes:

- Exploratory data analysis with class counts, missing-value checks, descriptive statistics, and a feature-pair plot.
- Logistic Regression and Decision Tree classifiers trained on a stratified train/test split.
- Accuracy comparison, classification reports, and confusion matrices.
- A CLI for predicting an Iris species from four measurements.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run EDA

```powershell
python iris_classifier.py eda
```

This prints the dataset summary and creates `outputs/feature_pairs.png`.

## Train and evaluate

```powershell
python iris_classifier.py train
```

This prints accuracy and classification metrics, explains the confusion-matrix layout, and creates one confusion-matrix image plus a model file for each classifier in `outputs/`.

Rows in each confusion matrix are actual species and columns are predicted species. Diagonal values are correct predictions; off-diagonal values are misclassifications. Iris errors typically occur between the visually similar `versicolor` and `virginica` classes.

## Predict a new flower

Measurements are ordered as sepal length, sepal width, petal length, and petal width, all in centimetres. Train first so the model files exist:

```powershell
python iris_classifier.py train
python iris_classifier.py predict 5.1 3.5 1.4 0.2
python iris_classifier.py predict 6.5 3.0 5.2 2.0 --model tree
```
