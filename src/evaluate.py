import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import evaluate
from sklearn.metrics import confusion_matrix, classification_report
from .dataset import load_nli_dataset
from .train import compute_metrics, tokenize_fn
from .hypotheses import HYPOTHESES, NLI_LABEL2ID
from .tokenizer_utils import load_tokenizer
import argparse

@torch.no_grad()
def predict_texts(texts, lang, tokenizer, model):
    """
    Predict labels for raw texts using NLI approach.
    
    Args:
        texts: List of texts to predict
        lang: Language code ('en' or 'de')
        tokenizer: Tokenizer
        model: Model
    
    Returns:
        List of predicted labels (0 or 1)
    """
    model.eval()
    predictions = []
    entail_id = NLI_LABEL2ID["entailment"]
    
    for text in texts:
        scores = {}
        for cls, hyp_dict in HYPOTHESES.items():
            hyp = hyp_dict[lang]
            inputs = tokenizer(text, hyp, return_tensors="pt", truncation=True)
            logits = model(**inputs).logits[0]
            probs = logits.softmax(-1).cpu().numpy()
            scores[cls] = probs[entail_id]
        
        # Choose class with highest entailment probability
        pred = max(scores.items(), key=lambda x: x[1])[0]
        predictions.append(int(pred))  # "0" or "1" -> 0 or 1
    
    return predictions

def evaluate_with_analysis(checkpoint_path: str, test_csv: str, lang: str, show_examples: int = 10):
    """
    Evaluate checkpoint with detailed analysis: confusion matrix and error examples.
    
    Args:
        checkpoint_path: Path to checkpoint
        test_csv: Path to test CSV file
        lang: Language ('en' or 'de')
        show_examples: Number of examples to show for each category (default: 10)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Evaluating on: {test_csv}")
    print(f"Language: {lang}")
    print("-" * 80)
    
    # Load tokenizer and model
    tokenizer = load_tokenizer(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    # Load original CSV data
    df = pd.read_csv(test_csv)
    df = df.dropna(subset=["Text", "Biased"])
    
    texts = df["Text"].tolist()
    true_labels = df["Biased"].astype(int).tolist()
    
    print(f"\nOriginal test samples: {len(texts)}")
    print(f"Class distribution: {pd.Series(true_labels).value_counts().to_dict()}")
    print("\nMaking predictions...")
    
    # Get predictions
    predictions = predict_texts(texts, lang, tokenizer, model)
    
    # Calculate metrics
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Classification report
    report = classification_report(true_labels, predictions, 
                                   target_names=['Non-Antisemitic (0)', 'Antisemitic (1)'],
                                   digits=4)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
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
    
    # Analyze errors
    df['Predicted'] = predictions
    df['Correct'] = df['Biased'] == df['Predicted']
    
    # True Positives (correctly identified antisemitic)
    tp = df[(df['Biased'] == 1) & (df['Predicted'] == 1)]
    # False Positives (incorrectly labeled as antisemitic)
    fp = df[(df['Biased'] == 0) & (df['Predicted'] == 1)]
    # False Negatives (missed antisemitic texts)
    fn = df[(df['Biased'] == 1) & (df['Predicted'] == 0)]
    # True Negatives (correctly identified non-antisemitic)
    tn = df[(df['Biased'] == 0) & (df['Predicted'] == 0)]
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print(f"True Positives (Antisemitic correctly identified):  {len(tp)}")
    print(f"False Positives (Incorrectly labeled antisemitic): {len(fp)}")
    print(f"False Negatives (Missed antisemitic texts):        {len(fn)}")
    print(f"True Negatives (Non-antisemitic correct):          {len(tn)}")
    print("=" * 80)
    
    # Show examples
    if len(tp) > 0:
        print("\n" + "=" * 80)
        print(f"TRUE POSITIVES - Antisemitic Texts Correctly Identified (showing {min(show_examples, len(tp))}/{len(tp)})")
        print("=" * 80)
        for idx, row in tp.head(show_examples).iterrows():
            print(f"\n[{idx}] {row['Text'][:200]}{'...' if len(row['Text']) > 200 else ''}")
    
    if len(fp) > 0:
        print("\n" + "=" * 80)
        print(f"FALSE POSITIVES - Non-Antisemitic Incorrectly Labeled (showing {min(show_examples, len(fp))}/{len(fp)})")
        print("=" * 80)
        for idx, row in fp.head(show_examples).iterrows():
            print(f"\n[{idx}] TRUE LABEL: Non-Antisemitic (0)")
            print(f"    {row['Text'][:200]}{'...' if len(row['Text']) > 200 else ''}")
    
    if len(fn) > 0:
        print("\n" + "=" * 80)
        print(f"FALSE NEGATIVES - Antisemitic Texts Missed (showing {min(show_examples, len(fn))}/{len(fn)})")
        print("=" * 80)
        for idx, row in fn.head(show_examples).iterrows():
            print(f"\n[{idx}] TRUE LABEL: Antisemitic (1)")
            print(f"    {row['Text'][:200]}{'...' if len(row['Text']) > 200 else ''}")
    
    print("\n" + "=" * 80)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'true_positives': len(tp),
        'false_positives': len(fp),
        'false_negatives': len(fn),
        'true_negatives': len(tn),
        'tp_texts': tp,
        'fp_texts': fp,
        'fn_texts': fn
    }

def evaluate_on_dataset(checkpoint_path: str, test_dataset: Dataset, lang: str = None):
    """
    Evaluates a checkpoint on an existing dataset (not CSV).
    
    Args:
        checkpoint_path: Path to checkpoint
        test_dataset: HuggingFace Dataset with NLI data (already expanded)
        lang: Language (optional, for display only)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    if lang:
        print(f"Language: {lang}")
    print(f"Test samples: {len(test_dataset)}")
    print("-" * 80)
    
    # Load tokenizer and model
    tokenizer = load_tokenizer(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    # Tokenize test data
    tokenized_test = test_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Run evaluation
    print("Running evaluation...")
    results = trainer.evaluate(eval_dataset=tokenized_test)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {results['eval_accuracy']:.4f} ({results['eval_accuracy']*100:.2f}%)")
    print(f"F1-Score:  {results['eval_f1']:.4f} ({results['eval_f1']*100:.2f}%)")
    print(f"Loss:      {results['eval_loss']:.4f}")
    print(f"Runtime:   {results['eval_runtime']:.2f}s")
    print(f"Samples/s: {results['eval_samples_per_second']:.2f}")
    print("=" * 80)
    
    return results

def evaluate_checkpoint(checkpoint_path: str, test_csv: str, lang: str):
    """
    Evaluates a checkpoint on a test dataset.
    
    Args:
        checkpoint_path: Path to checkpoint
        test_csv: Path to test CSV file
        lang: Language ('en' or 'de')
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Evaluating on: {test_csv}")
    print(f"Language: {lang}")
    print("-" * 80)
    
    # Load tokenizer and model
    tokenizer = load_tokenizer(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    
    # Load test data
    test_dataset = load_nli_dataset(test_csv, lang)
    
    # Tokenize test data
    tokenized_test = test_dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = trainer.evaluate(eval_dataset=tokenized_test)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {results['eval_accuracy']:.4f} ({results['eval_accuracy']*100:.2f}%)")
    print(f"F1-Score:  {results['eval_f1']:.4f} ({results['eval_f1']*100:.2f}%)")
    print(f"Loss:      {results['eval_loss']:.4f}")
    print(f"Runtime:   {results['eval_runtime']:.2f}s")
    print(f"Samples/s: {results['eval_samples_per_second']:.2f}")
    print("=" * 80)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint with detailed analysis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/xlmr-nli/de_ft",
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/de_cleaned.csv",
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="de",
        choices=["en", "de"],
        help="Language of the test data"
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=10,
        help="Number of examples to show for each category (default: 10)"
    )
    
    args = parser.parse_args()
    
    results = evaluate_with_analysis(args.checkpoint, args.test_data, args.lang, args.show_examples)
    
    return results

if __name__ == "__main__":
    main()
