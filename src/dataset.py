import pandas as pd
from datasets import Dataset
from typing import Dict, List
from .hypotheses import HYPOTHESES, NLI_LABEL2ID

def expand_to_nli(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        text = str(r["Text"])
        gold = str(r["Biased"])
        for cls, hyp_dict in HYPOTHESES.items():
            hyp = hyp_dict[lang]
            
            nli_label = "entailment" if cls == gold else "neutral"
            rows.append({
                "premise": text,
                "hypothesis": hyp,
                "nli_label": NLI_LABEL2ID[nli_label],
                "target_cls": gold,
                "hyp_cls": cls
            })
    return pd.DataFrame(rows)

def load_nli_dataset(csv_path: str, lang: str) -> Dataset:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Text", "Biased"])
    nli_df = expand_to_nli(df, lang)
    return Dataset.from_pandas(nli_df, preserve_index=False)