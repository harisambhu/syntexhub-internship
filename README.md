# Spam Detection

A reusable Python spam/ham text classification project using scikit-learn.

## Features

- Loads CSV, TSV, or the common SMS Spam Collection format
- Cleans text and tokenizes it with a regex tokenizer
- Supports TF-IDF or count vectors
- Supports Multinomial Naive Bayes or Logistic Regression
- Reports accuracy, precision, recall, F1, and a confusion matrix
- Saves the fitted vectorizer and model together in one `joblib` pipeline artifact
- Predicts new messages from the command line

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

The included sample dataset is small and is intended for a smoke test. For useful results, replace it with a larger labeled dataset.

```powershell
python train.py --data data/sms_spam_sample.csv --vectorizer tfidf --model logreg
```

The default artifact is saved to `artifacts/spam_pipeline.joblib`.

For the UCI/Kaggle SMS Spam Collection format (`ham<TAB>message`):

```powershell
python train.py --data data/SMSSpamCollection --format tsv --vectorizer count --model nb
```

## Predict

```powershell
python predict.py "Congratulations! You won a free prize. Call now!"
python predict.py --interactive
```

## Use in Python

```python
import joblib

bundle = joblib.load("artifacts/spam_pipeline.joblib")
pipeline = bundle["pipeline"]
print(pipeline.predict(["Can we meet for lunch tomorrow?"])[0])
```

The saved bundle also stores the training configuration, class labels, and evaluation metrics.
