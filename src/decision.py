import json
import os
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from .hypotheses import HYPOTHESES, NLI_LABEL2ID, hypotheses_for_class

DECISION_CONFIG_FILENAME = "decision_config.json"
DEFAULT_POSITIVE_MARGIN_THRESHOLD = 0.0
DEFAULT_HYPOTHESIS_AGGREGATION_TOP_K = 2


def _to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tokenized model inputs to the target torch device."""
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def score_text(text: str, lang: str, tokenizer, model, device: torch.device) -> Dict[str, float]:
    """Returns entailment probability per hypothesis class for one text."""
    entail_id = NLI_LABEL2ID["entailment"]
    scores: Dict[str, float] = {}
    for cls in HYPOTHESES.keys():
        entail_scores = []
        for hyp in hypotheses_for_class(cls, lang):
            inputs = tokenizer(text, hyp, return_tensors="pt", truncation=True)
            inputs = _to_device(inputs, device)
            logits = model(**inputs).logits[0]
            probs = logits.softmax(-1).detach().cpu().numpy()
            entail_scores.append(float(probs[entail_id]))

        if not entail_scores:
            scores[cls] = 0.0
            continue

        top_k = min(DEFAULT_HYPOTHESIS_AGGREGATION_TOP_K, len(entail_scores))
        # Use top-k mean instead of max to reduce one-hypothesis spikes.
        top_scores = sorted(entail_scores, reverse=True)[:top_k]
        scores[cls] = float(sum(top_scores) / len(top_scores))
    return scores


def predict_from_scores(scores: Dict[str, float], positive_margin_threshold: float = DEFAULT_POSITIVE_MARGIN_THRESHOLD) -> str:
    """
    Predicts binary class from class-wise entailment scores.

    Uses margin rule: predict class 1 if score_1 - score_0 >= threshold.
    """
    margin = float(scores.get("1", 0.0) - scores.get("0", 0.0))
    return "1" if margin >= positive_margin_threshold else "0"


@torch.no_grad()
def predict_texts_with_threshold(
    texts: Iterable[str],
    lang: str,
    tokenizer,
    model,
    device: torch.device,
    positive_margin_threshold: float = DEFAULT_POSITIVE_MARGIN_THRESHOLD,
    progress_label: str | None = None,
    log_every: int = 100,
) -> List[int]:
    preds: List[int] = []
    texts_list = list(texts)
    total = len(texts_list)
    for idx, text in enumerate(texts_list, start=1):
        scores = score_text(text, lang, tokenizer, model, device)
        preds.append(int(predict_from_scores(scores, positive_margin_threshold=positive_margin_threshold)))
        if progress_label and (idx % log_every == 0 or idx == total):
            print(f"{progress_label}: {idx}/{total}")
    return preds


@torch.no_grad()
def tune_positive_margin_threshold(
    texts: List[str],
    true_labels: List[int],
    lang: str,
    tokenizer,
    model,
    device: torch.device,
    objective: str = "f1",
    threshold_min: float = -1.0,
    threshold_max: float = 1.0,
    threshold_steps: int = 401,
    min_precision: float = 0.0,
    progress_label: str | None = None,
    log_every: int = 100,
) -> Dict[str, float]:
    """Finds decision threshold maximizing positive-class metric on validation tweets."""
    margins: List[float] = []
    total = len(texts)
    for idx, text in enumerate(texts, start=1):
        scores = score_text(text, lang, tokenizer, model, device)
        margins.append(float(scores.get("1", 0.0) - scores.get("0", 0.0)))
        if progress_label and (idx % log_every == 0 or idx == total):
            print(f"{progress_label}: {idx}/{total}")

    y_true = np.asarray(true_labels, dtype=int)
    margins_np = np.asarray(margins, dtype=float)

    objective = objective.lower().strip()
    if objective not in {"f1", "recall", "recall_at_precision"}:
        raise ValueError("objective must be 'f1', 'recall', or 'recall_at_precision'")

    min_precision = float(min_precision)
    if min_precision < 0.0 or min_precision > 1.0:
        raise ValueError("min_precision must be within [0, 1]")

    best = None
    fallback_best = None
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)
    for thr in thresholds:
        y_pred = (margins_np >= thr).astype(int)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        predicted_positive = int(y_pred.sum())

        fallback_rank = (recall, f1, precision, -abs(float(thr)))
        if fallback_best is None or fallback_rank > fallback_best["rank"]:
            fallback_best = {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positive": predicted_positive,
                "rank": fallback_rank,
            }

        if objective == "recall_at_precision" and precision < min_precision:
            continue

        if objective == "f1":
            rank = (f1, recall, precision, -abs(float(thr)))
        elif objective == "recall":
            rank = (recall, f1, precision, -abs(float(thr)))
        else:
            rank = (recall, f1, precision, -abs(float(thr)))

        if best is None or rank > best["rank"]:
            best = {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positive": predicted_positive,
                "rank": rank,
            }

    if best is None:
        # If precision constraint cannot be met, return the recall-optimal fallback
        # and mark that the constraint was not satisfied.
        assert fallback_best is not None
        best = fallback_best
        best["constraint_satisfied"] = False
    else:
        best["constraint_satisfied"] = True

    best.pop("rank", None)
    best["objective"] = objective
    best["min_precision"] = min_precision
    return best


def decision_config_path(ckpt_dir: str) -> str:
    """Return the full path to the decision-threshold config file for a checkpoint."""
    return os.path.join(ckpt_dir, DECISION_CONFIG_FILENAME)


def save_decision_config(
    ckpt_dir: str,
    positive_margin_threshold: float,
    objective: str,
    validation_metrics: Dict[str, float],
) -> str:
    path = decision_config_path(ckpt_dir)
    payload = {
        "positive_margin_threshold": float(positive_margin_threshold),
        "objective": objective,
        "validation_metrics": validation_metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return path


def load_decision_threshold(ckpt_dir: str, default: float = DEFAULT_POSITIVE_MARGIN_THRESHOLD) -> float:
    path = decision_config_path(ckpt_dir)
    if not os.path.exists(path):
        return float(default)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("positive_margin_threshold", default))
