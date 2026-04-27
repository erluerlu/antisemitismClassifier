import argparse
from typing import TypedDict

import torch
from transformers import AutoModelForSequenceClassification

from .decision import load_decision_threshold, predict_from_scores, score_text
from .tokenizer_utils import load_tokenizer


class ClassProbabilities(TypedDict):
    """Binary class probabilities for labels 0 and 1."""

    _0: float
    _1: float


class ClassificationResult(TypedDict):
    """Single-text inference output."""

    label: str
    label_int: int
    label_name: str
    probability: float
    class_probabilities: ClassProbabilities
    scores: dict[str, float]
    margin: float
    threshold: float


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a torch device from user input or pick CUDA/CPU automatically."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_nli_model(ckpt_dir: str, device: str | torch.device | None = None):
    """Loads tokenizer/model once for repeated predictions."""
    resolved_device = _resolve_device(device)
    tok = load_tokenizer(ckpt_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    mdl.to(resolved_device)
    mdl.eval()
    return tok, mdl, resolved_device


@torch.no_grad()
def predict_label_with(
    text: str,
    lang: str,
    tok,
    mdl,
    device: str | torch.device | None = None,
    positive_margin_threshold: float = 0.0,
):
    """Predicts one label from a preloaded model using thresholded NLI margin."""
    resolved_device = _resolve_device(device)
    mdl.eval()
    scores = score_text(text=text, lang=lang, tokenizer=tok, model=mdl, device=resolved_device)
    pred = predict_from_scores(scores, positive_margin_threshold=positive_margin_threshold)
    return pred, scores


@torch.no_grad()
def predict_label(
    text: str,
    lang: str,
    ckpt_dir: str,
    positive_margin_threshold: float | None = None,
):
    """
    One-shot prediction API.

    If threshold is None, uses checkpoint decision_config.json when available.
    """
    tok, mdl, device = load_nli_model(ckpt_dir)
    threshold = (
        load_decision_threshold(ckpt_dir) if positive_margin_threshold is None else float(positive_margin_threshold)
    )
    pred, scores = predict_label_with(
        text=text,
        lang=lang,
        tok=tok,
        mdl=mdl,
        device=device,
        positive_margin_threshold=threshold,
    )
    return pred, scores


def _binary_class_probabilities_from_scores(scores: dict[str, float]) -> tuple[float, float]:
    """
    Converts class entailment scores to binary probabilities.

    Note: this is a relative confidence over classes {0, 1}, not a calibrated
    real-world probability.
    """
    s0 = float(scores.get("0", 0.0))
    s1 = float(scores.get("1", 0.0))
    probs = torch.softmax(torch.tensor([s0, s1], dtype=torch.float32), dim=0)
    return float(probs[0].item()), float(probs[1].item())


@torch.no_grad()
def classify_text(
    text: str,
    lang: str,
    ckpt_dir: str,
    positive_margin_threshold: float | None = None,
) -> ClassificationResult:
    """
    Classifies one raw text as antisemitic (1) or non-antisemitic (0).

    Returns label, class probabilities, raw class entailment scores and margin.
    """
    tok, mdl, device = load_nli_model(ckpt_dir)
    threshold = (
        load_decision_threshold(ckpt_dir) if positive_margin_threshold is None else float(positive_margin_threshold)
    )

    pred, scores = predict_label_with(
        text=text,
        lang=lang,
        tok=tok,
        mdl=mdl,
        device=device,
        positive_margin_threshold=threshold,
    )

    prob_0, prob_1 = _binary_class_probabilities_from_scores(scores)
    margin = float(scores.get("1", 0.0) - scores.get("0", 0.0))
    pred_int = int(pred)

    return {
        "label": pred,
        "label_int": pred_int,
        "label_name": "antisemitic" if pred_int == 1 else "non_antisemitic",
        "probability": prob_1 if pred_int == 1 else prob_0,
        "class_probabilities": {"_0": prob_0, "_1": prob_1},
        "scores": {"0": float(scores.get("0", 0.0)), "1": float(scores.get("1", 0.0))},
        "margin": margin,
        "threshold": float(threshold),
    }


def main():
    """Parse CLI arguments and print a human-readable prediction for one text."""
    parser = argparse.ArgumentParser(description="Classify one text with a trained antisemitism NLI model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint directory")
    parser.add_argument("--lang", type=str, required=True, choices=["en", "de"], help="Input text language")
    parser.add_argument("--text", type=str, required=True, help="Input text to classify")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for positive margin threshold (score_1-score_0)",
    )
    args = parser.parse_args()

    out = classify_text(
        text=args.text,
        lang=args.lang,
        ckpt_dir=args.checkpoint,
        positive_margin_threshold=args.threshold,
    )

    print("Prediction:", out["label_name"], f"({out['label_int']})")
    print(f"Confidence (relative): {out['probability']:.4f}")
    print(
        "Class probs (0/1): "
        f"{out['class_probabilities']['_0']:.4f} / {out['class_probabilities']['_1']:.4f}"
    )
    print(f"Score margin (1-0): {out['margin']:.4f}")
    print(f"Decision threshold: {out['threshold']:.4f}")


if __name__ == "__main__":
    main()
