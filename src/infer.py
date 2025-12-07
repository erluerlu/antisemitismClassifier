import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .hypotheses import HYPOTHESES, NLI_LABEL2ID

@torch.no_grad()
def predict_label(text: str, lang: str, ckpt_dir: str):
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    mdl.eval()

    entail_id = NLI_LABEL2ID["entailment"]
    scores = {}
    for cls, hyp_dict in HYPOTHESES.items():
        hyp = hyp_dict[lang]
        inputs = tok(text, hyp, return_tensors="pt", truncation=True)
        logits = mdl(**inputs).logits[0]
        probs = logits.softmax(-1).cpu().numpy()
        scores[cls] = probs[entail_id]
    # wähle Klasse mit höchster Entailment-Wahrscheinlichkeit
    pred = max(scores.items(), key=lambda x: x[1])[0]
    return pred, scores