import argparse
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from .infer import load_nli_model, predict_label_with


def zero_shot_classify_csv(input_csv: str, output_csv: str, ckpt_dir: str, lang: str = "de"):
    """
    Zero-shot classification of texts without using existing labels.
    
    Reads CSV with 'Text' column, classifies each text using NLI model,
    and saves predictions to a new CSV with predicted class and confidence scores.
    
    Args:
        input_csv: Path to input CSV (must have 'Text' column, 'Biased' column ignored if present)
        output_csv: Path to output CSV with predictions
        ckpt_dir: Path to trained model checkpoint
        lang: Language for hypotheses ('en' or 'de')
    """
    from .hypotheses import HYPOTHESES
    
    # Load data
    df = pd.read_csv(input_csv)
    if 'Text' not in df.columns:
        raise ValueError("Input CSV must have 'Text' column")
    
    print(f"Loaded {len(df)} texts from {input_csv}")
    print(f"Using model: {ckpt_dir}")
    print(f"Language: {lang}")
    print(f"Available classes: {list(HYPOTHESES.keys())}")
    
    # Load model once
    tok, mdl, device = load_nli_model(ckpt_dir)
    
    predictions: List[str] = []
    scores_list: List[Dict[str, float]] = []
    
    # Classify each text
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Zero-shot classifying"):
        text = str(row["Text"])
        pred, scores = predict_label_with(text=text, lang=lang, tok=tok, mdl=mdl, device=device)
        predictions.append(pred)
        scores_list.append(scores)
    
    # Add predictions to dataframe
    df['Predicted_Class'] = predictions
    
    # Add confidence scores for each class
    for class_key in HYPOTHESES.keys():
        df[f'score_{class_key}'] = [scores.get(class_key, 0.0) for scores in scores_list]
    
    # Add prediction confidence (highest score)
    df['confidence'] = [max(scores.values()) for scores in scores_list]
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to {output_csv}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CLASSIFICATION SUMMARY")
    print("="*80)
    
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    for class_key in sorted(pred_counts.index):
        count = pred_counts[class_key]
        percentage = (count / len(predictions)) * 100
        print(f"Class '{class_key}': {count} ({percentage:.1f}%)")
    
    # Print low-confidence predictions
    low_conf_threshold = 0.6
    low_conf = df[df['confidence'] < low_conf_threshold]
    if len(low_conf) > 0:
        print(f"\n{len(low_conf)} predictions with confidence < {low_conf_threshold}:")
        print(low_conf[['Text', 'Predicted_Class', 'confidence']].head(10))
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Zero-shot classification of texts using NLI model")
    parser.add_argument("--input", required=True, help="Input CSV file with 'Text' column")
    parser.add_argument("--output", required=True, help="Output CSV file with predictions")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--lang", default="de", choices=["en", "de"], help="Language for hypotheses")
    args = parser.parse_args()
    
    zero_shot_classify_csv(
        input_csv=args.input,
        output_csv=args.output,
        ckpt_dir=args.ckpt,
        lang=args.lang
    )


if __name__ == "__main__":
    main()
