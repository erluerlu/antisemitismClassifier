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
DEFAULT_CONTRADICTION_WEIGHT = 1.0
DEFAULT_CLASS_0_WEIGHT = 0.7  # Dampen non-antisemitic hypotheses to reduce false negatives
DEFAULT_SCORE_MODE = "entailment_only"
DEFAULT_NLI_TRAIN_MODE = "both_classes"


def _hypothesis_score(
    entail_prob: float,
    contradiction_prob: float,
    contradiction_weight: float,
    score_mode: str,
) -> float:
    """Compute per-hypothesis score according to selected score mode."""
    mode = str(score_mode).strip().lower()
    if mode == "entailment_only":
        return float(entail_prob)
    if mode == "entailment_minus_contradiction":
        return float(entail_prob - float(contradiction_weight) * contradiction_prob)
    raise ValueError("score_mode must be 'entailment_only' or 'entailment_minus_contradiction'")


def _to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tokenized model inputs to the target torch device."""
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def score_text(
    text: str,
    lang: str,
    tokenizer,
    model,
    device: torch.device,
    contradiction_weight: float = DEFAULT_CONTRADICTION_WEIGHT,
    class_0_weight: float = DEFAULT_CLASS_0_WEIGHT,
    score_mode: str = DEFAULT_SCORE_MODE,
    nli_train_mode: str = DEFAULT_NLI_TRAIN_MODE,
) -> Dict[str, float]:
    """Returns class score per hypothesis class for one text."""
    entail_id = NLI_LABEL2ID["entailment"]
    contradiction_id = NLI_LABEL2ID["contradiction"]
    # In class1_only mode, class-0 hypotheses were never trained on – skip them
    # and fix score_0 = 0 so the margin equals score_1 directly.
    if str(nli_train_mode).strip().lower() == "class1_only":
        return _score_text_class1_only(
            text, lang, tokenizer, model, device,
            contradiction_weight=contradiction_weight,
            score_mode=score_mode,
        )
    scores: Dict[str, float] = {}
    for cls in HYPOTHESES.keys():
        class_scores: list[float] = []
        for hyp in hypotheses_for_class(cls, lang):
            inputs = tokenizer(text, hyp, return_tensors="pt", truncation=True)
            inputs = _to_device(inputs, device)
            logits = model(**inputs).logits[0]
            probs = logits.softmax(-1).detach().cpu().numpy()
            entail_prob = float(probs[entail_id])
            contradiction_prob = float(probs[contradiction_id])
            class_scores.append(
                _hypothesis_score(
                    entail_prob=entail_prob,
                    contradiction_prob=contradiction_prob,
                    contradiction_weight=contradiction_weight,
                    score_mode=score_mode,
                )
            )

        if not class_scores:
            scores[cls] = 0.0
            continue

        # Aggregate consistently for both classes: take the maximum hypothesis score.
        # Using max for class 0 too prevents weak class-0 hypotheses from being averaged out
        # by strong ones (which previously caused score_0 to dominate score_1).
        scores[cls] = float(max(class_scores))
    # Apply class_0_weight dampening on the final aggregated class-0 score.
    if "0" in scores:
        scores["0"] = scores["0"] * float(class_0_weight)
    return scores


@torch.no_grad()
def _score_text_class1_only(
    text: str,
    lang: str,
    tokenizer,
    model,
    device: torch.device,
    contradiction_weight: float = DEFAULT_CONTRADICTION_WEIGHT,
    score_mode: str = DEFAULT_SCORE_MODE,
) -> Dict[str, float]:
    """Scores using only class-1 hypotheses (class1_only training mode)."""
    entail_id = NLI_LABEL2ID["entailment"]
    contradiction_id = NLI_LABEL2ID["contradiction"]
    class_scores: list[float] = []
    for hyp in hypotheses_for_class("1", lang):
        inputs = tokenizer(text, hyp, return_tensors="pt", truncation=True)
        inputs = _to_device(inputs, device)
        logits = model(**inputs).logits[0]
        probs = logits.softmax(-1).detach().cpu().numpy()
        class_scores.append(
            _hypothesis_score(
                entail_prob=float(probs[entail_id]),
                contradiction_prob=float(probs[contradiction_id]),
                contradiction_weight=contradiction_weight,
                score_mode=score_mode,
            )
        )
    score_1 = float(max(class_scores)) if class_scores else 0.0
    # score_0 fixed at 0 – margin = score_1 directly
    return {"1": score_1, "0": 0.0}


@torch.no_grad()
def score_text_detailed(
    text: str,
    lang: str,
    tokenizer,
    model,
    device: torch.device,
    contradiction_weight: float = DEFAULT_CONTRADICTION_WEIGHT,
    class_0_weight: float = DEFAULT_CLASS_0_WEIGHT,
    score_mode: str = DEFAULT_SCORE_MODE,
    nli_train_mode: str = DEFAULT_NLI_TRAIN_MODE,
) -> tuple[Dict[str, float], Dict[str, list[tuple[str, float]]]]:
    """
    Like score_text, but also returns per-hypothesis entailment scores.

    Returns:
        scores: class score (cls1=max hypothesis score, cls0=mean entailment), same as score_text
        hyp_scores: {cls: [(hypothesis_text, score), ...]} sorted descending
    """
    entail_id = NLI_LABEL2ID["entailment"]
    contradiction_id = NLI_LABEL2ID["contradiction"]
    scores: Dict[str, float] = {}
    hyp_scores: Dict[str, list[tuple[str, float]]] = {}

    for cls in HYPOTHESES.keys():
        hyps = hypotheses_for_class(cls, lang)
        class_scores: list[tuple[str, float]] = []
        for hyp in hyps:
            inputs = tokenizer(text, hyp, return_tensors="pt", truncation=True)
            inputs = _to_device(inputs, device)
            logits = model(**inputs).logits[0]
            probs = logits.softmax(-1).detach().cpu().numpy()
            entail_prob = float(probs[entail_id])
            contradiction_prob = float(probs[contradiction_id])
            score = _hypothesis_score(
                entail_prob=entail_prob,
                contradiction_prob=contradiction_prob,
                contradiction_weight=contradiction_weight,
                score_mode=score_mode,
            )
            class_scores.append((hyp, score))

        if not class_scores:
            scores[cls] = 0.0
            hyp_scores[cls] = []
            continue

        sorted_hyps = sorted(class_scores, key=lambda x: x[1], reverse=True)
        hyp_scores[cls] = sorted_hyps
        # Same aggregation as score_text: max over hypotheses for both classes.
        scores[cls] = float(max(s for _, s in sorted_hyps))

    # Skip class-0 hypotheses entirely in class1_only mode
    mode = str(nli_train_mode).strip().lower()
    if mode == "class1_only":
        scores["0"] = 0.0
        hyp_scores["0"] = []
    elif "0" in scores:
        scores["0"] = scores["0"] * float(class_0_weight)

    return scores, hyp_scores


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
    contradiction_weight: float = DEFAULT_CONTRADICTION_WEIGHT,
    class_0_weight: float = DEFAULT_CLASS_0_WEIGHT,
    score_mode: str = DEFAULT_SCORE_MODE,
    nli_train_mode: str = DEFAULT_NLI_TRAIN_MODE,
    progress_label: str | None = None,
    log_every: int = 100,
) -> List[int]:
    preds: List[int] = []
    texts_list = list(texts)
    total = len(texts_list)
    for idx, text in enumerate(texts_list, start=1):
        scores = score_text(
            text,
            lang,
            tokenizer,
            model,
            device,
            contradiction_weight=contradiction_weight,
            class_0_weight=class_0_weight,
            score_mode=score_mode,
            nli_train_mode=nli_train_mode,
        )
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
    contradiction_weight: float = DEFAULT_CONTRADICTION_WEIGHT,
    class_0_weight: float = DEFAULT_CLASS_0_WEIGHT,
    score_mode: str = DEFAULT_SCORE_MODE,
    nli_train_mode: str = DEFAULT_NLI_TRAIN_MODE,
    progress_label: str | None = None,
    log_every: int = 100,
) -> Dict[str, float]:
    """Finds decision threshold maximizing positive-class metric on validation tweets."""
    margins: List[float] = []
    total = len(texts)
    for idx, text in enumerate(texts, start=1):
        scores = score_text(
            text,
            lang,
            tokenizer,
            model,
            device,
            contradiction_weight=contradiction_weight,
            class_0_weight=class_0_weight,
            score_mode=score_mode,
            nli_train_mode=nli_train_mode,
        )
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
        if objective == "recall_at_precision":
            best["constraint_satisfied"] = True
        else:
            # For unconstrained objectives, still expose whether min_precision was met
            # (useful for debugging accidental threshold collapse).
            best["constraint_satisfied"] = bool(best["precision"] >= min_precision)

    best.pop("rank", None)
    best["objective"] = objective
    best["min_precision"] = min_precision
    best["contradiction_weight"] = float(contradiction_weight)
    best["nli_train_mode"] = str(nli_train_mode)
    return best


def decision_config_path(ckpt_dir: str) -> str:
    """Return the full path to the decision-threshold config file for a checkpoint."""
    return os.path.join(ckpt_dir, DECISION_CONFIG_FILENAME)


def save_decision_config(
    ckpt_dir: str,
    positive_margin_threshold: float,
    contradiction_weight: float,
    score_mode: str,
    objective: str,
    validation_metrics: Dict[str, float],
    nli_train_mode: str = DEFAULT_NLI_TRAIN_MODE,
) -> str:
    path = decision_config_path(ckpt_dir)
    payload = {
        "positive_margin_threshold": float(positive_margin_threshold),
        "contradiction_weight": float(contradiction_weight),
        "score_mode": str(score_mode),
        "nli_train_mode": str(nli_train_mode),
        "objective": objective,
        "validation_metrics": validation_metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return path


def load_decision_params(
    ckpt_dir: str,
    default_threshold: float = DEFAULT_POSITIVE_MARGIN_THRESHOLD,
    default_contradiction_weight: float = DEFAULT_CONTRADICTION_WEIGHT,
    default_score_mode: str = DEFAULT_SCORE_MODE,
    default_nli_train_mode: str = DEFAULT_NLI_TRAIN_MODE,
) -> tuple[float, float, str, str]:
    path = decision_config_path(ckpt_dir)
    if not os.path.exists(path):
        return float(default_threshold), float(default_contradiction_weight), str(default_score_mode), str(default_nli_train_mode)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        float(data.get("positive_margin_threshold", default_threshold)),
        float(data.get("contradiction_weight", default_contradiction_weight)),
        str(data.get("score_mode", default_score_mode)),
        str(data.get("nli_train_mode", default_nli_train_mode)),
    )


def load_decision_threshold(ckpt_dir: str, default: float = DEFAULT_POSITIVE_MARGIN_THRESHOLD) -> float:
    threshold, _, _, _ = load_decision_params(ckpt_dir, default_threshold=default)
    return float(threshold)
