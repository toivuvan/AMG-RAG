import argparse
import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


VALID_LABELS = ["A", "B", "C", "D", "E"]


def normalize_answer(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)
    if text in VALID_LABELS:
        return text
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MEDQA-style multiple-choice QA results."
    )
    parser.add_argument("--input", default="results/medqa_baseline.csv")
    parser.add_argument("--expected-col", default="expected_answer")
    parser.add_argument("--pred-col", default="model_answer")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.expected_col not in df.columns or args.pred_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns {args.expected_col!r} and {args.pred_col!r}. "
            f"Available columns: {list(df.columns)}"
        )

    expected = df[args.expected_col].map(normalize_answer)
    predicted = df[args.pred_col].map(normalize_answer)
    valid_mask = expected.notna()

    if not valid_mask.any():
        raise ValueError("No valid expected labels found.")

    expected_valid = expected[valid_mask]
    predicted_valid = predicted[valid_mask].fillna("NAN")

    labels_for_f1 = VALID_LABELS + (["NAN"] if "NAN" in set(predicted_valid) else [])
    accuracy = accuracy_score(expected_valid, predicted_valid)
    macro_f1 = f1_score(
        expected_valid,
        predicted_valid,
        labels=labels_for_f1,
        average="macro",
        zero_division=0,
    )

    print(f"Rows: {len(df)}")
    print(f"Evaluated rows: {len(expected_valid)}")
    print(f"Invalid predictions: {(predicted_valid == 'NAN').sum()}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    if args.report:
        print()
        print(classification_report(expected_valid, predicted_valid, zero_division=0))


if __name__ == "__main__":
    main()
