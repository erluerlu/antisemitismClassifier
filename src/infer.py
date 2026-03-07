import torch
from transformers import AutoModelForSequenceClassification

from .decision import load_decision_threshold, predict_from_scores, score_text
from .tokenizer_utils import load_tokenizer


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
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
