from __future__ import annotations

import argparse
import json

from src.spam_detector import load_dataset, save_bundle, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a spam/ham classifier.")
    parser.add_argument("--data", default="data/sms_spam_sample.csv", help="CSV or TSV dataset path")
    parser.add_argument("--format", choices=["auto", "csv", "tsv"], default="auto")
    parser.add_argument("--vectorizer", choices=["tfidf", "count"], default="tfidf")
    parser.add_argument("--model", choices=["logreg", "nb"], default="logreg")
    parser.add_argument("--output", default="artifacts/spam_pipeline.joblib", help="Output joblib artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts, labels = load_dataset(args.data, args.format)
    pipeline, metrics = train_and_evaluate(texts, labels, args.vectorizer, args.model)
    save_bundle(args.output, pipeline, metrics, args.vectorizer, args.model)

    print(f"Loaded {len(texts)} messages: {int((labels == 'ham').sum())} ham, {int((labels == 'spam').sum())} spam")
    print(f"Vectorizer: {args.vectorizer} | Model: {args.model}")
    print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))
    print(f"Saved pipeline to {args.output}")


if __name__ == "__main__":
    main()
