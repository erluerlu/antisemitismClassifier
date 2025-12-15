import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict, List, Optional, Tuple
from .hypotheses import HYPOTHESES, NLI_LABEL2ID

def expand_to_nli(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Expands a dataset with bias labels into NLI (Natural Language Inference) format.
    
    For each text and its bias label, creates multiple premise-hypothesis pairs
    where the hypothesis corresponds to each possible bias class.
    
    Args:
        df: DataFrame with 'Text' and 'Biased' columns
        lang: Language code ('en' or 'de') for hypothesis selection
    
    Returns:
        DataFrame with columns: premise, hypothesis, nli_label, target_cls, hyp_cls
    """
    rows = []
    for _, r in df.iterrows():
        text = str(r["Text"])
        gold = str(r["Biased"])
        for cls, hyp_dict in HYPOTHESES.items():
            hyp = hyp_dict[lang]
            
            # Wenn Hypothese der echten Klasse entspricht: entailment
            # Sonst: contradiction (nicht neutral, da wir nur 2 Labels haben)
            nli_label = "entailment" if cls == gold else "contradiction"
            rows.append({
                "premise": text,
                "hypothesis": hyp,
                "nli_label": NLI_LABEL2ID[nli_label],
                "target_cls": gold,
                "hyp_cls": cls
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
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Text", "Biased"])
    nli_df = expand_to_nli(df, lang)
    dataset = Dataset.from_pandas(nli_df, preserve_index=False)
    
    if test_size is not None:
        split = dataset.train_test_split(test_size=test_size, seed=seed)
        return split
    return dataset