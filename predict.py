from __future__ import annotations

import argparse
import json

from src.spam_detector import load_bundle, predict_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify messages with a saved spam pipeline.")
    parser.add_argument("messages", nargs="*", help="Messages to classify")
    parser.add_argument("--model", default="artifacts/spam_pipeline.joblib", help="Saved joblib artifact")
    parser.add_argument("--interactive", action="store_true", help="Read one message per line until Ctrl+Z")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    messages = list(args.messages)
    if args.interactive:
        print("Enter messages, one per line. Press Ctrl+Z then Enter to finish.")
        messages.extend(line.strip() for line in iter(input, "") if line.strip())
    if not messages:
        raise SystemExit("Provide a message or use --interactive.")

    predictions = predict_messages(load_bundle(args.model), messages)
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
