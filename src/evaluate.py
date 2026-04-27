import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, Trainer

from .dataset import load_nli_dataset
from .decision import load_decision_threshold
from .infer import load_nli_model, predict_label_with
from .tokenizer_utils import load_tokenizer
from .train import compute_metrics, tokenize_fn


def predict_texts(texts, lang, tokenizer, model, device, positive_margin_threshold: float = 0.0):
    """Predict labels for raw texts using thresholded hypothesis-based NLI."""
    predictions = []
    for text in texts:
        pred, _ = predict_label_with(
            text=text,
            lang=lang,
            tok=tokenizer,
            mdl=model,
            device=device,
            positive_margin_threshold=positive_margin_threshold,
        )
        predictions.append(int(pred))
    return predictions


def evaluate_with_analysis(
    checkpoint_path: str,
    test_csv: str,
    lang: str,
    show_examples: int = 10,
    positive_margin_threshold: float | None = None,
):
    """Evaluate checkpoint with confusion matrix, positive-class metrics and error examples."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Evaluating on: {test_csv}")
    print(f"Language: {lang}")
    print("-" * 80)

    tokenizer, model, device = load_nli_model(checkpoint_path)

    if positive_margin_threshold is None:
        threshold = load_decision_threshold(checkpoint_path)
    else:
        threshold = float(positive_margin_threshold)

    print(f"Decision threshold (margin score_1-score_0): {threshold:.4f}")

    df = pd.read_csv(test_csv)
    df = df.dropna(subset=["Text", "Biased"])

    texts = df["Text"].astype(str).tolist()
    true_labels = df["Biased"].astype(int).tolist()

    print(f"\nOriginal test samples: {len(texts)}")
    print(f"Class distribution: {pd.Series(true_labels).value_counts().to_dict()}")
    print("\nMaking predictions...")

    predictions = predict_texts(texts, lang, tokenizer, model, device, positive_margin_threshold=threshold)

    y_true = np.array(true_labels)
    y_pred = np.array(predictions)

    accuracy = float(np.mean(y_pred == y_true))
    precision_pos = float(precision_score(y_true, y_pred, zero_division=0))
    recall_pos = float(recall_score(y_true, y_pred, zero_division=0))
    f1_pos = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(
        true_labels,
        predictions,
        target_names=["Non-Antisemitic (0)", "Antisemitic (1)"],
        digits=4,
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Positive Precision:{precision_pos:.4f}")
    print(f"Positive Recall:   {recall_pos:.4f}")
    print(f"Positive F1:       {f1_pos:.4f}")
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    print("                    Predicted")
    print("                 Non-AS    AS")
    print(f"Actual Non-AS   {cm[0][0]:6d}  {cm[0][1]:5d}")
    print(f"Actual AS       {cm[1][0]:6d}  {cm[1][1]:5d}")

    print("\n" + "-" * 80)
    print("CLASSIFICATION REPORT")
    print("-" * 80)
    print(report)
    print("=" * 80)

    df["Predicted"] = predictions
    df["Correct"] = df["Biased"] == df["Predicted"]

    tp = df[(df["Biased"] == 1) & (df["Predicted"] == 1)]
    fp = df[(df["Biased"] == 0) & (df["Predicted"] == 1)]
    fn = df[(df["Biased"] == 1) & (df["Predicted"] == 0)]
    tn = df[(df["Biased"] == 0) & (df["Predicted"] == 0)]

    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print(f"True Positives (Antisemitic correctly identified):  {len(tp)}")
    print(f"False Positives (Incorrectly labeled antisemitic): {len(fp)}")
    print(f"False Negatives (Missed antisemitic texts):        {len(fn)}")
    print(f"True Negatives (Non-antisemitic correct):          {len(tn)}")
    print("=" * 80)

    if len(tp) > 0:
        print("\n" + "=" * 80)
        print(
            "TRUE POSITIVES - Antisemitic Texts Correctly Identified "
            f"(showing {min(show_examples, len(tp))}/{len(tp)})"
        )
        print("=" * 80)
        for idx, row in tp.head(show_examples).iterrows():
            print(f"\n[{idx}] {row['Text'][:200]}{'...' if len(row['Text']) > 200 else ''}")

    if len(fp) > 0:
        print("\n" + "=" * 80)
        print(
            "FALSE POSITIVES - Non-Antisemitic Incorrectly Labeled "
            f"(showing {min(show_examples, len(fp))}/{len(fp)})"
        )
        print("=" * 80)
        for idx, row in fp.head(show_examples).iterrows():
            print(f"\n[{idx}] TRUE LABEL: Non-Antisemitic (0)")
            print(f"    {row['Text'][:200]}{'...' if len(row['Text']) > 200 else ''}")

    if len(fn) > 0:
        print("\n" + "=" * 80)
        print(
            "FALSE NEGATIVES - Antisemitic Texts Missed "
            f"(showing {min(show_examples, len(fn))}/{len(fn)})"
        )
        print("=" * 80)
        for idx, row in fn.head(show_examples).iterrows():
            print(f"\n[{idx}] TRUE LABEL: Antisemitic (1)")
            print(f"    {row['Text'][:200]}{'...' if len(row['Text']) > 200 else ''}")

    print("\n" + "=" * 80)

    return {
        "accuracy": accuracy,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "confusion_matrix": cm,
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "true_negatives": len(tn),
        "tp_texts": tp,
        "fp_texts": fp,
        "fn_texts": fn,
        "threshold": threshold,
    }


def evaluate_on_dataset(checkpoint_path: str, test_dataset: Dataset, lang: str | None = None):
    """Pair-level NLI evaluation for debugging/trainer consistency."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    if lang:
        print(f"Language: {lang}")
    print(f"Test samples: {len(test_dataset)}")
    print("-" * 80)

    tokenizer = load_tokenizer(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    tokenized_test = test_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)

    print("Running evaluation...")
    results = trainer.evaluate(eval_dataset=tokenized_test)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {results['eval_accuracy']:.4f} ({results['eval_accuracy'] * 100:.2f}%)")
    print(f"F1-Score:  {results['eval_f1']:.4f} ({results['eval_f1'] * 100:.2f}%)")
    print(f"Loss:      {results['eval_loss']:.4f}")
    print(f"Runtime:   {results['eval_runtime']:.2f}s")
    print(f"Samples/s: {results['eval_samples_per_second']:.2f}")
    print("=" * 80)

    return results


def evaluate_checkpoint(checkpoint_path: str, test_csv: str, lang: str):
    """Pair-level NLI evaluation on expanded test CSV (legacy helper)."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Evaluating on: {test_csv}")
    print(f"Language: {lang}")
    print("-" * 80)

    tokenizer = load_tokenizer(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    test_dataset = load_nli_dataset(test_csv, lang)

    tokenized_test = test_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)

    print("\nRunning evaluation...")
    results = trainer.evaluate(eval_dataset=tokenized_test)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {results['eval_accuracy']:.4f} ({results['eval_accuracy'] * 100:.2f}%)")
    print(f"F1-Score:  {results['eval_f1']:.4f} ({results['eval_f1'] * 100:.2f}%)")
    print(f"Loss:      {results['eval_loss']:.4f}")
    print(f"Runtime:   {results['eval_runtime']:.2f}s")
    print(f"Samples/s: {results['eval_samples_per_second']:.2f}")
    print("=" * 80)

    return results


def main():
    """Parse evaluation CLI arguments and run detailed checkpoint evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint with detailed analysis")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/xlmr-nli/de_ft", help="Path to checkpoint directory")
    parser.add_argument("--test_data", type=str, default="data/de_cleaned.csv", help="Path to test CSV file")
    parser.add_argument("--lang", type=str, default="de", choices=["en", "de"], help="Language of the test data")
    parser.add_argument(
        "--show_examples",
        type=int,
        default=10,
        help="Number of examples to show for each category (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for positive margin threshold (score_1-score_0).",
    )

    args = parser.parse_args()

    results = evaluate_with_analysis(
        args.checkpoint,
        args.test_data,
        args.lang,
        args.show_examples,
        positive_margin_threshold=args.threshold,
    )

    return results


if __name__ == "__main__":
    main()
