import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict, List, Optional, Tuple
from .hypotheses import HYPOTHESES, NLI_LABEL2ID, hypotheses_for_class


def load_binary_dataframe(csv_path: str) -> pd.DataFrame:
    """Loads raw tweet-level data and enforces clean binary labels."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Text", "Biased"]).copy()
    df["Biased"] = df["Biased"].astype(int)
    # Prevent train/val overlap from exact duplicate text-label rows.
    df = df.drop_duplicates(subset=["Text", "Biased"]).reset_index(drop=True)
    return df


def oversample_positive_rows(df: pd.DataFrame, factor: float, seed: int = 42) -> pd.DataFrame:
    """
    Oversamples positive (Biased==1) tweet rows before NLI expansion.

    factor=1.0 keeps data unchanged.
    factor=2.0 roughly doubles positive tweet rows.
    """
    if factor <= 1.0:
        return df

    pos_df = df[df["Biased"] == 1]
    if len(pos_df) == 0:
        return df

    extra_count = int(round((factor - 1.0) * len(pos_df)))
    if extra_count <= 0:
        return df

    extra_pos = pos_df.sample(n=extra_count, replace=True, random_state=seed)
    out = pd.concat([df, extra_pos], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def dataframe_to_nli_dataset(df: pd.DataFrame, lang: str, nli_train_mode: str = "both_classes", tweet_id_offset: int = 0) -> Dataset:
    """Converts a tweet-level dataframe to expanded NLI dataset format."""
    nli_df = expand_to_nli(df, lang, nli_train_mode=nli_train_mode, tweet_id_offset=tweet_id_offset)
    return Dataset.from_pandas(nli_df, preserve_index=False)


def _resolve_nli_train_mode(nli_train_mode: str) -> str:
    """Normalize training mode names while preserving backward compatibility."""
    mode = str(nli_train_mode).strip().lower()
    if mode == "both_classes":
        return "both_classes_contradiction"
    valid_modes = {
        "both_classes_contradiction",
        "both_classes_asymmetric_neutral",
        "class1_only",
    }
    if mode not in valid_modes:
        raise ValueError(
            "nli_train_mode must be one of "
            "'both_classes', 'both_classes_contradiction', "
            "'both_classes_asymmetric_neutral', or 'class1_only'"
        )
    return mode


def expand_to_nli(df: pd.DataFrame, lang: str, nli_train_mode: str = "both_classes", tweet_id_offset: int = 0) -> pd.DataFrame:
    """
    Expands a dataset with bias labels into NLI (Natural Language Inference) format.

        nli_train_mode:
            "both_classes" or "both_classes_contradiction"
                                                – all hypotheses from class 0 and 1. Wrong-class pairs are
                                                    always trained as contradiction (current baseline).
            "both_classes_asymmetric_neutral"
                                                – class-1 hypotheses on non-antisemitic tweets are trained as
                                                    neutral, while class-0 hypotheses on antisemitic tweets stay
                                                    contradiction.
            "class1_only"   – only class-1 hypotheses: entailment for antisemitic tweets,
                                                contradiction for non-antisemitic tweets. Class-0 hypotheses
                                                are entirely omitted from training.

    Args:
        df: DataFrame with 'Text' and 'Biased' columns
        lang: Language code ('en' or 'de') for hypothesis selection

    Returns:
        DataFrame with columns: premise, hypothesis, nli_label, target_cls, hyp_cls
    """
    mode = _resolve_nli_train_mode(nli_train_mode)

    # Determine which hypothesis classes to emit.
    active_classes = ["1"] if mode == "class1_only" else list(HYPOTHESES.keys())

    lang_id = 0 if str(lang).lower() == "en" else 1

    rows = []
    for tweet_id, (_, r) in enumerate(df.iterrows()):
        text = str(r["Text"])
        gold = str(r["Biased"])
        for cls in active_classes:
            for hyp in hypotheses_for_class(cls, lang):
                if cls == gold:
                    nli_label = "entailment"
                elif mode == "both_classes_asymmetric_neutral" and gold == "0" and cls == "1":
                    nli_label = "neutral"
                else:
                    nli_label = "contradiction"
                rows.append({
                    "premise": text,
                    "hypothesis": hyp,
                    "nli_label": NLI_LABEL2ID[nli_label],
                    "target_cls": int(gold),
                    "hyp_cls": int(cls),
                    "tweet_id": int(tweet_id) + int(tweet_id_offset),
                    "lang_id": lang_id,
                })
    return pd.DataFrame(rows)

def load_nli_dataset(csv_path: str, lang: str, test_size: Optional[float] = None, seed: int = 42) -> Dataset | DatasetDict:
    """
    Lädt NLI-Dataset aus CSV.
    
    Args:
        csv_path: Pfad zur CSV-Datei
        lang: Sprache ('en' oder 'de')
        test_size: Anteil für Test-Split (z.B. 0.2 für 20%). Wenn None, kein Split.
        seed: Random seed für reproduzierbare Splits
    
    Returns:
        Dataset wenn test_size=None, sonst DatasetDict mit 'train' und 'test'
    """
    df = load_binary_dataframe(csv_path)
    dataset = dataframe_to_nli_dataset(df, lang)
    
    if test_size is not None:
        split = dataset.train_test_split(test_size=test_size, seed=seed)
        return split
    return dataset