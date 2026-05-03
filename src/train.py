import os
import time
import json
from tkinter import NONE
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import evaluate
from datasets import Dataset, DatasetDict
from .dataset import load_binary_dataframe, dataframe_to_nli_dataset, oversample_positive_rows
from .hypotheses import NLI_LABEL2ID
from .tokenizer_utils import load_tokenizer
from .decision import (
    DEFAULT_CONTRADICTION_WEIGHT,
    DEFAULT_NLI_TRAIN_MODE,
    DEFAULT_POSITIVE_MARGIN_THRESHOLD,
    DEFAULT_SCORE_MODE,
    predict_texts_with_threshold,
    save_decision_config,
    tune_positive_margin_threshold,
)
from .logging_utils import redirect_output_to_log
import torch


@dataclass
class Config:
    """Training configuration parameters."""

    model_name: str = "xlm-roberta-base"
    output_dir: str = "checkpoints/xlmr-nli"
    lr: float = 2e-5
    epochs: int = 4
    batch_size: int = 16
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    pos_oversample_factor: float = 2.0
    positive_sample_weight: float = 1.0
    en_pos_oversample_factor: float = 1.5
    de_pos_oversample_factor: float = 4.0
    en_positive_sample_weight: float = 1.5
    de_positive_sample_weight: float = 2.5
    threshold_objective: str = "f1"
    threshold_min_precision: float = 0.5
    threshold_search_min: float = -0.2
    threshold_search_max: float = 0.8
    contradiction_weight: float = DEFAULT_CONTRADICTION_WEIGHT
    score_mode: str = DEFAULT_SCORE_MODE
    nli_train_mode: str = DEFAULT_NLI_TRAIN_MODE
    class_0_weight: float = 0.7
    class_0_loss_weight: float = 0.7
    model_selection_objective: str = "f1"
    model_selection_min_precision: float = 0.4
    model_selection_last_k_checkpoints: int = 0
    tweet_level_loss_weight: float = 0.2
    tweet_level_margin: float = 0.0
    de_lang_loss_weight: float = 2.0
    joint_eval_lang_for_best: str = "de"


def _print_class_distribution(name: str, df: pd.DataFrame):
    """Print sample count and binary class distribution for a dataframe split."""
    dist = df["Biased"].value_counts().sort_index().to_dict()
    print(f"{name}: {len(df)} samples | class dist: {dist}")


def _prepare_tweet_level_splits(
    train_csv: str,
    val_csv: str,
    val_size: float | None,
    hold_test_size: float | None,
    seed: int,
):
    """
    Creates leak-free splits on original tweet rows before NLI expansion.
    """
    df_original = load_binary_dataframe(train_csv)

    test_df = None
    test_csv_path = None

    if hold_test_size is not None:
        print(f"\n{'=' * 80}")
        print(f"Holding out {hold_test_size * 100:.0f}% of data as test set (NOT used for training)")
        print(f"{'=' * 80}")
        train_val_df, test_df = train_test_split(
            df_original,
            test_size=hold_test_size,
            random_state=seed,
            stratify=df_original["Biased"],
        )

        test_csv_path = train_csv.replace(".csv", "_test_holdout.csv")
        test_df.to_csv(test_csv_path, index=False)
        print(f"Test CSV saved: {test_csv_path} ({len(test_df)} original samples)")

        train_val_csv_path = train_csv.replace(".csv", "_train_val.csv")
        train_val_df.to_csv(train_val_csv_path, index=False)
    else:
        train_val_df = df_original

    if val_size is not None:
        if hold_test_size is not None:
            relative_val_size = val_size / (1 - hold_test_size)
        else:
            relative_val_size = val_size

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=seed,
            stratify=train_val_df["Biased"],
        )
    else:
        train_df = train_val_df
        val_df = load_binary_dataframe(val_csv) if val_csv else None

    return train_df.reset_index(drop=True), (val_df.reset_index(drop=True) if val_df is not None else None), (
        test_df.reset_index(drop=True) if test_df is not None else None
    ), test_csv_path


def tokenize_fn(examples, tokenizer):
    """Tokenizes premise-hypothesis pairs and keeps tweet-level metadata for weighted loss."""
    encoded = tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
    encoded["labels"] = examples["nli_label"]
    encoded["target_cls"] = examples["target_cls"]
    encoded["hyp_cls"] = examples["hyp_cls"]
    encoded["tweet_id"] = examples["tweet_id"]
    if "lang_id" in examples:
        encoded["lang_id"] = examples["lang_id"]
    return encoded


def tokenize_binary_fn(examples, tokenizer):
    """Tokenize single-text binary classification samples."""
    encoded = tokenizer(examples["text"], truncation=True)
    encoded["labels"] = examples["labels"]
    return encoded


def dataframe_to_binary_dataset(df: pd.DataFrame) -> Dataset:
    """Convert tweet-level dataframe to a 2-class dataset for direct classification."""
    out_df = pd.DataFrame(
        {
            "text": df["Text"].astype(str).tolist(),
            "labels": df["Biased"].astype(int).tolist(),
        }
    )
    return Dataset.from_pandas(out_df, preserve_index=False)


def compute_metrics_binary(eval_pred):
    """Metrics for direct 2-class tweet classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float(np.mean(preds == labels))
    p = float(precision_score(labels, preds, zero_division=0))
    r = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    return {
        "accuracy": acc,
        "precision_pos": p,
        "recall_pos": r,
        "f1": f1,
    }


@torch.no_grad()
def evaluate_binary_with_analysis(
    checkpoint_path: str,
    test_csv: str,
    show_examples: int = 10,
):
    """Evaluate a direct binary text classifier (non-NLI) on held-out tweet data."""
    print(f"Loading binary checkpoint from: {checkpoint_path}")
    print(f"Evaluating on: {test_csv}")
    print("Language: de")
    print("-" * 80)

    tokenizer = load_tokenizer(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = pd.read_csv(test_csv).dropna(subset=["Text", "Biased"]).copy()
    texts = df["Text"].astype(str).tolist()
    y_true = df["Biased"].astype(int).to_numpy()

    y_pred = []
    for i, text in enumerate(texts, start=1):
        enc = tokenizer(text, return_tensors="pt", truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[0]
        pred = int(torch.argmax(logits).item())
        y_pred.append(pred)
        if i % 100 == 0 or i == len(texts):
            print(f"Binary prediction scoring: {i}/{len(texts)}")

    y_pred = np.array(y_pred)
    accuracy = float(np.mean(y_pred == y_true))
    precision_pos = float(precision_score(y_true, y_pred, zero_division=0))
    recall_pos = float(recall_score(y_true, y_pred, zero_division=0))
    f1_pos = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Non-Antisemitic (0)", "Antisemitic (1)"],
        digits=4,
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS (DIRECT BINARY)")
    print("=" * 80)
    print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Positive Precision:{precision_pos:.4f}")
    print(f"Positive Recall:   {recall_pos:.4f}")
    print(f"Positive F1:       {f1_pos:.4f}")

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

    df["Predicted"] = y_pred
    tp = df[(df["Biased"] == 1) & (df["Predicted"] == 1)]
    fp = df[(df["Biased"] == 0) & (df["Predicted"] == 1)]
    fn = df[(df["Biased"] == 1) & (df["Predicted"] == 0)]
    tn = df[(df["Biased"] == 0) & (df["Predicted"] == 0)]

    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print(f"True Positives (Antisemitic correctly identified):  {len(tp)}")
    print(f"False Positives (Incorrectly labeled antisemitic): {len(fp)}")
    print(f"False Negatives (Missed antisemitic texts):        {len(fn)}")
    print(f"True Negatives (Non-antisemitic correct):          {len(tn)}")

    if len(tp) > 0:
        print("\n" + "=" * 80)
        print(
            "TRUE POSITIVES - Antisemitic Texts Correctly Identified "
            f"(showing {min(show_examples, len(tp))}/{len(tp)})"
        )
        print("=" * 80)
        for idx, row in tp.head(show_examples).iterrows():
            print(f"\n[{idx}] {row['Text']}")

    if len(fp) > 0:
        print("\n" + "=" * 80)
        print(
            "FALSE POSITIVES - Non-Antisemitic Incorrectly Labeled "
            f"(showing {min(show_examples, len(fp))}/{len(fp)})"
        )
        print("=" * 80)
        for idx, row in fp.head(show_examples).iterrows():
            print(f"\n[{idx}] TRUE LABEL: Non-Antisemitic (0)")
            print(f"    {row['Text']}")

    if len(fn) > 0:
        print("\n" + "=" * 80)
        print(
            "FALSE NEGATIVES - Antisemitic Texts Missed "
            f"(showing {min(show_examples, len(fn))}/{len(fn)})"
        )
        print("=" * 80)
        for idx, row in fn.head(show_examples).iterrows():
            print(f"\n[{idx}] TRUE LABEL: Antisemitic (1)")
            print(f"    {row['Text']}")

    return {
        "accuracy": accuracy,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "true_negatives": len(tn),
    }


_F1_METRIC = None
_ACC_METRIC = None


def compute_metrics(eval_pred):
    """Computes pair-level metrics for trainer logging."""
    global _F1_METRIC, _ACC_METRIC
    if _F1_METRIC is None:
        _F1_METRIC = evaluate.load("f1")
    if _ACC_METRIC is None:
        _ACC_METRIC = evaluate.load("accuracy")
    metric_f1 = _F1_METRIC
    metric_acc = _ACC_METRIC
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
    labels_list = labels.tolist() if hasattr(labels, "tolist") else list(labels)

    acc_res = metric_acc.compute(predictions=preds_list, references=labels_list)
    f1_macro = metric_f1.compute(predictions=preds_list, references=labels_list, average="macro")
    # Robust against 3-class predictions (e.g., neutral): compute entailment as one-vs-rest.
    entailment_id = NLI_LABEL2ID["entailment"]
    y_true_entail = [1 if y == entailment_id else 0 for y in labels_list]
    y_pred_entail = [1 if y == entailment_id else 0 for y in preds_list]
    recall_entail = recall_score(y_true_entail, y_pred_entail, zero_division=0)
    f1_entail = f1_score(y_true_entail, y_pred_entail, zero_division=0)

    out = {k: float(v) for k, v in acc_res.items()}
    out["f1"] = float(f1_macro["f1"])
    out["recall_entailment"] = float(recall_entail)
    out["f1_entailment"] = float(f1_entail)
    return out


def _tweet_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Computes positive-class tweet-level metrics."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _rank_tweet_metrics(metrics: dict[str, float], objective: str, min_precision: float) -> tuple:
    objective = objective.lower().strip()
    p = float(metrics["precision"])
    r = float(metrics["recall"])
    f1 = float(metrics["f1"])

    if objective == "f1":
        return (f1, r, p)
    if objective == "recall":
        return (r, f1, p)
    if objective == "recall_at_precision":
        constraint = 1 if p >= min_precision else 0
        return (constraint, r, f1, p)
    raise ValueError("model_selection_objective must be 'f1', 'recall', or 'recall_at_precision'")


def _append_tweet_level_checkpoint_metrics(output_dir: str, payload: dict[str, float | int | str | bool]) -> str:
    """Append per-checkpoint tweet-level metrics to a JSONL log file."""
    path = os.path.join(output_dir, "tweet_level_checkpoint_metrics.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return path


class TweetLevelCheckpointEvalCallback(TrainerCallback):
    """Run tweet-level validation scoring at each evaluation step."""

    def __init__(self, val_texts: list[str], val_labels: list[int], lang: str, tokenizer, cfg: Config):
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.lang = lang
        self.tokenizer = tokenizer
        self.cfg = cfg

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs.get("model")
        output_dir = args.output_dir or self.cfg.output_dir
        if model is None or not self.val_texts or not output_dir:
            return

        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{state.global_step}")
        print("\n" + "-" * 80)
        print(f"Tweet-level validation scoring at eval step: {checkpoint_dir}")
        print("-" * 80)

        preds = predict_texts_with_threshold(
            texts=self.val_texts,
            lang=self.lang,
            tokenizer=self.tokenizer,
            model=model,
            device=model.device,
            positive_margin_threshold=DEFAULT_POSITIVE_MARGIN_THRESHOLD,
            contradiction_weight=self.cfg.contradiction_weight,
            class_0_weight=self.cfg.class_0_weight,
            score_mode=self.cfg.score_mode,
            nli_train_mode=self.cfg.nli_train_mode,
            progress_label=f"Tweet-level scoring {os.path.basename(checkpoint_dir)}",
        )
        tweet_metrics = _tweet_metrics(self.val_labels, preds)
        tweet_metrics["constraint_satisfied"] = bool(
            tweet_metrics["precision"] >= self.cfg.model_selection_min_precision
        )

        # Inject tweet-level metrics into Trainer eval metrics so metric_for_best_model
        # and EarlyStoppingCallback can operate on the actual task metric.
        sink = metrics if isinstance(metrics, dict) else kwargs.get("metrics")
        if not isinstance(sink, dict):
            sink = None
        if sink is not None:
            sink["eval_tweet_precision"] = float(tweet_metrics["precision"])
            sink["eval_tweet_recall"] = float(tweet_metrics["recall"])
            sink["eval_tweet_f1"] = float(tweet_metrics["f1"])
            sink["eval_tweet_constraint_satisfied"] = bool(tweet_metrics["constraint_satisfied"])

        payload = {
            "checkpoint": checkpoint_dir,
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else -1.0,
            "precision": float(tweet_metrics["precision"]),
            "recall": float(tweet_metrics["recall"]),
            "f1": float(tweet_metrics["f1"]),
            "constraint_satisfied": bool(tweet_metrics["constraint_satisfied"]),
            "score_mode": self.cfg.score_mode,
            "class_0_weight": float(self.cfg.class_0_weight),
            "contradiction_weight": float(self.cfg.contradiction_weight),
        }
        metrics_path = _append_tweet_level_checkpoint_metrics(output_dir, payload)
        print(
            "Tweet-level checkpoint metrics: "
            f"p={payload['precision']:.4f} "
            f"r={payload['recall']:.4f} "
            f"f1={payload['f1']:.4f} "
            f"constraint_satisfied={payload['constraint_satisfied']}"
        )
        print(f"Saved tweet-level checkpoint metrics: {metrics_path}")
        return


def _list_candidate_checkpoint_dirs(output_dir: str) -> list[str]:
    """Returns candidate model directories (all checkpoint-* plus output_dir itself)."""
    candidates: list[str] = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path) and name.startswith("checkpoint-"):
                candidates.append(path)
        candidates.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]))
        has_model = os.path.exists(os.path.join(output_dir, "config.json")) and (
            os.path.exists(os.path.join(output_dir, "model.safetensors"))
            or os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))
        )
        if has_model:
            candidates.append(output_dir)
    return candidates


def _checkpoint_last_write_time(checkpoint_dir: str) -> float:
    """Return latest relevant file mtime from a checkpoint directory (fallback: directory mtime)."""
    priority_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "trainer_state.json",
        "config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ]

    mtimes: list[float] = []
    for filename in priority_files:
        file_path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(file_path):
            try:
                mtimes.append(os.path.getmtime(file_path))
            except OSError:
                pass

    if mtimes:
        return max(mtimes)

    try:
        return os.path.getmtime(checkpoint_dir)
    except OSError:
        return 0.0


def _select_best_checkpoint_by_tweet_objective(
    output_dir: str,
    val_texts: list[str],
    val_labels: list[int],
    lang: str,
    tokenizer,
    cfg: Config,
    num_labels: int,
    candidate_checkpoints: list[str] | None = None,
) -> tuple[str, dict[str, float]]:
    """Selects checkpoint by tweet-level objective at fixed threshold 0.0."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = candidate_checkpoints if candidate_checkpoints is not None else _list_candidate_checkpoint_dirs(output_dir)
    if not candidates:
        return output_dir, {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "constraint_satisfied": False,
        }

    best_path: str | None = None
    best_metrics: dict[str, float] | None = None
    best_rank = None

    for idx, ckpt in enumerate(candidates, start=1):
        try:
            print(f"Evaluating checkpoint {idx}/{len(candidates)}: {ckpt}")
            eval_model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=num_labels)
            eval_model.to(device)
            eval_model.eval()

            preds = predict_texts_with_threshold(
                texts=val_texts,
                lang=lang,
                tokenizer=tokenizer,
                model=eval_model,
                device=device,
                positive_margin_threshold=DEFAULT_POSITIVE_MARGIN_THRESHOLD,
                contradiction_weight=cfg.contradiction_weight,
                class_0_weight=cfg.class_0_weight,
                score_mode=cfg.score_mode,
                nli_train_mode=cfg.nli_train_mode,
                progress_label=f"Scoring validation tweets for {os.path.basename(ckpt)}",
            )
            metrics = _tweet_metrics(val_labels, preds)
            metrics["constraint_satisfied"] = bool(metrics["precision"] >= cfg.model_selection_min_precision)
            rank = _rank_tweet_metrics(metrics, cfg.model_selection_objective, cfg.model_selection_min_precision)

            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_path = ckpt
                best_metrics = metrics
        except Exception as exc:
            print(f"Skipping checkpoint '{ckpt}' during tweet-level selection: {exc}")

    if best_path is None or best_metrics is None:
        return output_dir, {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "constraint_satisfied": False,
        }
    return best_path, best_metrics


def train_stage(
    train_csv: str,
    val_csv: str,
    lang: str,
    cfg: Config,
    resume_from: str | None,
    val_size: float | None = None,
    hold_test_size: float | None = None,
):
    """
    Trains or fine-tunes an NLI model for bias detection.

    Returns:
        Tuple of (checkpoint_path, test_csv_path, lang)
    """
    tokenizer = load_tokenizer(cfg.model_name)

    train_df, val_df, test_df, test_csv_path = _prepare_tweet_level_splits(
        train_csv=train_csv,
        val_csv=val_csv,
        val_size=val_size,
        hold_test_size=hold_test_size,
        seed=cfg.seed,
    )

    if lang == "en":
        oversample_factor = float(cfg.en_pos_oversample_factor)
        positive_sample_weight = float(cfg.en_positive_sample_weight)
    else:
        oversample_factor = float(cfg.de_pos_oversample_factor)
        positive_sample_weight = float(cfg.de_positive_sample_weight)

    if oversample_factor > 1.0:
        train_df = oversample_positive_rows(train_df, factor=oversample_factor, seed=cfg.seed)
        print(f"Applied positive oversampling factor: {oversample_factor:.2f}")

    print(f"Positive sample loss weight: {positive_sample_weight:.2f}")
    print(
        "Tweet-level auxiliary loss: "
        f"weight={cfg.tweet_level_loss_weight:.3f} "
        f"margin={cfg.tweet_level_margin:.3f}"
    )

    print("\nTweet-level split summary")
    _print_class_distribution("Train", train_df)
    if val_df is not None:
        _print_class_distribution("Validation", val_df)
    if test_df is not None:
        _print_class_distribution("Test", test_df)

    train_ds = dataframe_to_nli_dataset(train_df, lang, nli_train_mode=cfg.nli_train_mode)
    val_ds = dataframe_to_nli_dataset(val_df, lang, nli_train_mode=cfg.nli_train_mode) if val_df is not None else None
    val_texts = [str(x) for x in val_df["Text"].tolist()] if val_df is not None else []
    val_labels = [int(x) for x in val_df["Biased"].tolist()] if val_df is not None else []

    if test_df is not None:
        test_nli_len = len(dataframe_to_nli_dataset(test_df, lang, nli_train_mode=cfg.nli_train_mode))
        print(f"Test set: {test_nli_len} NLI samples (held out for final evaluation)")

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds}) if val_ds else DatasetDict({"train": train_ds})
    tokenized = ds_dict.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        remove_columns=ds_dict["train"].column_names,
    )
    num_labels = len(NLI_LABEL2ID)

    model_path = resume_from if resume_from else cfg.model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Weighted loss at tweet-level: rows from positive tweets get more weight.
    def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute cross-entropy loss with optional upweighting for positive tweets."""
        labels = inputs.pop("labels", None)
        target_cls = inputs.pop("target_cls", None)
        hyp_cls = inputs.pop("hyp_cls", None)
        tweet_id = inputs.pop("tweet_id", None)
        lang_id = inputs.pop("lang_id", None)

        if labels is None:
            raise ValueError("'labels' not found in batch inputs. Available keys: " + str(list(inputs.keys())))

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        sample_losses = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        # Build per-sample weights: upweight positive tweets, downweight class-0 hypothesis rows
        # so the CE loss is consistent with the dampening applied during decision scoring.
        weights = torch.ones_like(sample_losses)
        if target_cls is not None and positive_sample_weight != 1.0:
            pos_mask = target_cls.view(-1).to(weights.device) == 1
            weights = torch.where(pos_mask, weights * float(positive_sample_weight), weights)
        if hyp_cls is not None and cfg.class_0_loss_weight != 1.0:
            hyp0_mask = hyp_cls.view(-1).to(weights.device) == 0
            weights = torch.where(hyp0_mask, weights * float(cfg.class_0_loss_weight), weights)
        if lang_id is not None and cfg.de_lang_loss_weight != 1.0:
            de_mask = lang_id.view(-1).to(weights.device) == 1
            weights = torch.where(de_mask, weights * float(cfg.de_lang_loss_weight), weights)

        ce_loss = (sample_losses * weights).sum() / weights.sum().clamp(min=1.0)

        loss = ce_loss
        # Optional tweet-level margin loss aligned with decision scoring.
        # Operates on raw entailment logits (not probs) so gradients are well-scaled.
        if (
            cfg.tweet_level_loss_weight > 0.0
            and target_cls is not None
            and hyp_cls is not None
            and tweet_id is not None
        ):
            entail_logits = logits[:, NLI_LABEL2ID["entailment"]]

            tweet_ids = tweet_id.view(-1).to(logits.device)
            hyp_classes = hyp_cls.view(-1).to(logits.device)
            tweet_targets = target_cls.view(-1).to(logits.device).float()

            unique_tweet_ids = torch.unique(tweet_ids)
            tweet_margins: list[torch.Tensor] = []
            tweet_labels: list[torch.Tensor] = []
            neg_inf = torch.tensor(-1e4, device=logits.device)
            for tid in unique_tweet_ids:
                tweet_mask = tweet_ids == tid
                if not torch.any(tweet_mask):
                    continue

                tweet_entail = entail_logits[tweet_mask]
                tweet_hyp = hyp_classes[tweet_mask]
                label = tweet_targets[tweet_mask][0]

                pos_mask = tweet_hyp == 1
                neg_mask = tweet_hyp == 0

                pos_logit = torch.max(tweet_entail[pos_mask]) if torch.any(pos_mask) else neg_inf
                neg_logit = torch.max(tweet_entail[neg_mask]) if torch.any(neg_mask) else neg_inf
                margin = pos_logit - neg_logit

                tweet_margins.append(margin)
                tweet_labels.append(label)

            if tweet_margins:
                margin_tensor = torch.stack(tweet_margins)
                label_tensor = torch.stack(tweet_labels)
                bias = float(cfg.tweet_level_margin)
                tweet_loss = torch.nn.functional.binary_cross_entropy_with_logits(margin_tensor - bias, label_tensor)
                loss = ce_loss + float(cfg.tweet_level_loss_weight) * tweet_loss

        return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="steps" if val_ds else "no",
        save_strategy="steps" if val_ds else "no",
        logging_steps=100,
        save_steps=300,
        eval_steps=300,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="tweet_f1",
        greater_is_better=True,
        save_total_limit=10,  # Keep last 5 checkpoints for model-selection flexibility
        seed=cfg.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if val_ds else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if val_ds else None,
    )
    trainer.compute_loss = compute_loss

    if val_ds:
        trainer.add_callback(TweetLevelCheckpointEvalCallback(val_texts, val_labels, lang, tokenizer, cfg))
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    # Keep run start timestamp so we can detect checkpoints written/updated by this run.
    run_start_ts = time.time()

    trainer.train()
    best_dir = args.output_dir

    # Select best checkpoint by tweet-level objective to align with final task goal.
    if val_df is not None and len(val_df) > 0:
        current_checkpoints: list[str] = []
        if os.path.isdir(args.output_dir):
            for name in os.listdir(args.output_dir):
                path = os.path.join(args.output_dir, name)
                if not (os.path.isdir(path) and name.startswith("checkpoint-")):
                    continue

                # Use checkpoint file mtimes (not only directory mtime) to detect writes in this run.
                # This is more robust on Windows when existing checkpoint-* directories are reused.
                mtime = _checkpoint_last_write_time(path)
                if mtime >= run_start_ts - 1.0:
                    current_checkpoints.append(path)
        current_checkpoints.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]))

        # Prefer checkpoints produced in this run; only score the latest K to reduce runtime.
        if cfg.model_selection_last_k_checkpoints > 0 and len(current_checkpoints) > cfg.model_selection_last_k_checkpoints:
            current_checkpoints = current_checkpoints[-cfg.model_selection_last_k_checkpoints :]

        # Add HF-selected best checkpoint as a safety candidate (if not already in latest K).
        run_candidates: list[str] = []
        if trainer.state.best_model_checkpoint and os.path.isdir(trainer.state.best_model_checkpoint):
            run_candidates.append(trainer.state.best_model_checkpoint)
        for p in current_checkpoints:
            if p not in run_candidates:
                run_candidates.append(p)

        if run_candidates:
            print(
                "Selecting best checkpoint from current run candidates: "
                f"{len(run_candidates)} checkpoint(s)"
            )
        else:
            print(
                "No new checkpoint-* directories found for this run; "
                "falling back to all available candidates in output_dir."
            )

        selected_ckpt, selected_metrics = _select_best_checkpoint_by_tweet_objective(
            output_dir=args.output_dir,
            val_texts=val_texts,
            val_labels=val_labels,
            lang=lang,
            tokenizer=tokenizer,
            cfg=cfg,
            num_labels=num_labels,
            candidate_checkpoints=run_candidates if run_candidates else None,
        )
        print(
            "Tweet-level checkpoint selection: "
            f"objective={cfg.model_selection_objective} "
            f"min_precision={cfg.model_selection_min_precision:.2f} "
            f"selected='{selected_ckpt}' "
            f"p={selected_metrics['precision']:.4f} "
            f"r={selected_metrics['recall']:.4f} "
            f"f1={selected_metrics['f1']:.4f} "
            f"constraint_satisfied={selected_metrics['constraint_satisfied']}"
        )

        if selected_ckpt != args.output_dir:
            trainer.model = AutoModelForSequenceClassification.from_pretrained(selected_ckpt, num_labels=num_labels)
            trainer.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Calibrate decision threshold on tweet-level validation data.
    if val_df is not None and len(val_df) > 0:
        print("\nCalibrating positive decision threshold on validation tweets...")
        val_texts = [str(x) for x in val_df["Text"].tolist()]
        val_labels = [int(x) for x in val_df["Biased"].tolist()]
        threshold_result = tune_positive_margin_threshold(
            texts=val_texts,
            true_labels=val_labels,
            lang=lang,
            tokenizer=tokenizer,
            model=trainer.model,
            device=trainer.model.device,
            objective=cfg.threshold_objective,
            threshold_min=cfg.threshold_search_min,
            threshold_max=cfg.threshold_search_max,
            min_precision=cfg.threshold_min_precision,
            contradiction_weight=cfg.contradiction_weight,
            class_0_weight=cfg.class_0_weight,
            score_mode=cfg.score_mode,
            nli_train_mode=cfg.nli_train_mode,
            progress_label="Threshold calibration scoring",
        )
        cfg_path = save_decision_config(
            ckpt_dir=best_dir,
            positive_margin_threshold=threshold_result["threshold"],
            contradiction_weight=cfg.contradiction_weight,
            score_mode=cfg.score_mode,
            nli_train_mode=cfg.nli_train_mode,
            objective=cfg.threshold_objective,
            validation_metrics=threshold_result,
        )
        print(
            "Threshold calibration complete: "
            f"objective={cfg.threshold_objective} "
            f"min_precision={cfg.threshold_min_precision:.2f} "
            f"thr={threshold_result['threshold']:.4f} "
            f"p={threshold_result['precision']:.4f} "
            f"r={threshold_result['recall']:.4f} "
            f"f1={threshold_result['f1']:.4f} "
            f"contradiction_weight={cfg.contradiction_weight:.4f} "
            f"score_mode={cfg.score_mode} "
            f"nli_train_mode={cfg.nli_train_mode} "
            f"constraint_satisfied={threshold_result.get('constraint_satisfied', True)}"
        )
        print(f"Saved decision config: {cfg_path}")
    else:
        cfg_path = save_decision_config(
            ckpt_dir=best_dir,
            positive_margin_threshold=DEFAULT_POSITIVE_MARGIN_THRESHOLD,
            contradiction_weight=cfg.contradiction_weight,
            score_mode=cfg.score_mode,
            nli_train_mode=cfg.nli_train_mode,
            objective="default",
            validation_metrics={},
        )
        print(f"Saved default decision config: {cfg_path}")

    return best_dir, test_csv_path, lang


def train_joint_stage(
    en_train_csv: str,
    de_train_csv: str,
    cfg: Config,
    val_size: float | None = None,
    hold_test_size: float | None = None,
):
    """
    Joint EN+DE training in a single stage. Avoids EN-knowledge being overwritten
    during a separate DE fine-tuning step. Validation/test stay language-separated;
    DE F1 drives best-model selection.
    """
    tokenizer = load_tokenizer(cfg.model_name)

    print("Preparing EN splits...")
    en_train_df, en_val_df, en_test_df, en_test_csv = _prepare_tweet_level_splits(
        train_csv=en_train_csv, val_csv="", val_size=val_size,
        hold_test_size=hold_test_size, seed=cfg.seed,
    )
    print("Preparing DE splits...")
    de_train_df, de_val_df, de_test_df, de_test_csv = _prepare_tweet_level_splits(
        train_csv=de_train_csv, val_csv="", val_size=val_size,
        hold_test_size=hold_test_size, seed=cfg.seed,
    )

    if cfg.en_pos_oversample_factor > 1.0:
        en_train_df = oversample_positive_rows(en_train_df, factor=cfg.en_pos_oversample_factor, seed=cfg.seed)
        print(f"EN positive oversampling factor: {cfg.en_pos_oversample_factor:.2f}")
    if cfg.de_pos_oversample_factor > 1.0:
        de_train_df = oversample_positive_rows(de_train_df, factor=cfg.de_pos_oversample_factor, seed=cfg.seed)
        print(f"DE positive oversampling factor: {cfg.de_pos_oversample_factor:.2f}")

    print(
        f"DE language loss weight: {cfg.de_lang_loss_weight:.2f} | "
        f"EN positive sample weight: {cfg.en_positive_sample_weight:.2f} | "
        f"DE positive sample weight: {cfg.de_positive_sample_weight:.2f}"
    )

    print("\nTweet-level split summary (joint)")
    _print_class_distribution("EN Train", en_train_df)
    _print_class_distribution("DE Train", de_train_df)
    if en_val_df is not None:
        _print_class_distribution("EN Validation", en_val_df)
    if de_val_df is not None:
        _print_class_distribution("DE Validation", de_val_df)

    # Build NLI datasets per language with non-overlapping tweet_ids.
    en_train_ds = dataframe_to_nli_dataset(en_train_df, "en", nli_train_mode=cfg.nli_train_mode, tweet_id_offset=0)
    de_train_offset = len(en_train_df) + 10
    de_train_ds = dataframe_to_nli_dataset(de_train_df, "de", nli_train_mode=cfg.nli_train_mode, tweet_id_offset=de_train_offset)

    en_val_ds = dataframe_to_nli_dataset(en_val_df, "en", nli_train_mode=cfg.nli_train_mode) if en_val_df is not None else None
    de_val_ds = dataframe_to_nli_dataset(de_val_df, "de", nli_train_mode=cfg.nli_train_mode) if de_val_df is not None else None

    from datasets import concatenate_datasets
    train_ds = concatenate_datasets([en_train_ds, de_train_ds]).shuffle(seed=cfg.seed)

    # Use DE val set as the primary HF eval_dataset (DE-priority); we still log EN tweet metrics via callback.
    primary_val_ds = de_val_ds if de_val_ds is not None else en_val_ds

    ds_dict = DatasetDict({"train": train_ds, "validation": primary_val_ds}) if primary_val_ds else DatasetDict({"train": train_ds})
    tokenized = ds_dict.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        remove_columns=ds_dict["train"].column_names,
    )
    num_labels = len(NLI_LABEL2ID)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=num_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Use sample_weight for positives = avg of EN and DE positive weights; DE lang weight is applied separately in compute_loss.
    positive_sample_weight = float((cfg.en_positive_sample_weight + cfg.de_positive_sample_weight) / 2.0)

    def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        target_cls = inputs.pop("target_cls", None)
        hyp_cls = inputs.pop("hyp_cls", None)
        tweet_id = inputs.pop("tweet_id", None)
        lang_id = inputs.pop("lang_id", None)
        if labels is None:
            raise ValueError("'labels' missing in batch")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        sample_losses = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        weights = torch.ones_like(sample_losses)
        if target_cls is not None and positive_sample_weight != 1.0:
            pos_mask = target_cls.view(-1).to(weights.device) == 1
            weights = torch.where(pos_mask, weights * float(positive_sample_weight), weights)
        if hyp_cls is not None and cfg.class_0_loss_weight != 1.0:
            hyp0_mask = hyp_cls.view(-1).to(weights.device) == 0
            weights = torch.where(hyp0_mask, weights * float(cfg.class_0_loss_weight), weights)
        if lang_id is not None and cfg.de_lang_loss_weight != 1.0:
            de_mask = lang_id.view(-1).to(weights.device) == 1
            weights = torch.where(de_mask, weights * float(cfg.de_lang_loss_weight), weights)

        ce_loss = (sample_losses * weights).sum() / weights.sum().clamp(min=1.0)

        loss = ce_loss
        if (
            cfg.tweet_level_loss_weight > 0.0
            and target_cls is not None
            and hyp_cls is not None
            and tweet_id is not None
        ):
            entail_logits = logits[:, NLI_LABEL2ID["entailment"]]
            tweet_ids = tweet_id.view(-1).to(logits.device)
            hyp_classes = hyp_cls.view(-1).to(logits.device)
            tweet_targets = target_cls.view(-1).to(logits.device).float()
            unique_tweet_ids = torch.unique(tweet_ids)
            tweet_margins: list[torch.Tensor] = []
            tweet_labels: list[torch.Tensor] = []
            neg_inf = torch.tensor(-1e4, device=logits.device)
            for tid in unique_tweet_ids:
                tweet_mask = tweet_ids == tid
                if not torch.any(tweet_mask):
                    continue
                tweet_entail = entail_logits[tweet_mask]
                tweet_hyp = hyp_classes[tweet_mask]
                label = tweet_targets[tweet_mask][0]
                pos_mask = tweet_hyp == 1
                neg_mask = tweet_hyp == 0
                pos_logit = torch.max(tweet_entail[pos_mask]) if torch.any(pos_mask) else neg_inf
                neg_logit = torch.max(tweet_entail[neg_mask]) if torch.any(neg_mask) else neg_inf
                margin = pos_logit - neg_logit
                tweet_margins.append(margin)
                tweet_labels.append(label)
            if tweet_margins:
                margin_tensor = torch.stack(tweet_margins)
                label_tensor = torch.stack(tweet_labels)
                tweet_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    margin_tensor - float(cfg.tweet_level_margin), label_tensor
                )
                loss = ce_loss + float(cfg.tweet_level_loss_weight) * tweet_loss

        return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="steps" if primary_val_ds else "no",
        save_strategy="steps" if primary_val_ds else "no",
        logging_steps=100,
        save_steps=300,
        eval_steps=300,
        load_best_model_at_end=True if primary_val_ds else False,
        metric_for_best_model="tweet_f1_de" if cfg.joint_eval_lang_for_best == "de" else "tweet_f1_en",
        greater_is_better=True,
        save_total_limit=10,
        seed=cfg.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if primary_val_ds else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if primary_val_ds else None,
    )
    trainer.compute_loss = compute_loss

    if primary_val_ds:
        trainer.add_callback(JointTweetLevelCheckpointEvalCallback(
            en_val_df=en_val_df, de_val_df=de_val_df, tokenizer=tokenizer, cfg=cfg,
        ))
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    # Track run start so we can identify checkpoints written by this run.
    run_start_ts = time.time()

    trainer.train()
    best_dir = args.output_dir
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Collect candidate checkpoints from this run for per-language selection.
    current_checkpoints: list[str] = []
    if os.path.isdir(args.output_dir):
        for name in os.listdir(args.output_dir):
            path = os.path.join(args.output_dir, name)
            if not (os.path.isdir(path) and name.startswith("checkpoint-")):
                continue
            mtime = _checkpoint_last_write_time(path)
            if mtime >= run_start_ts - 1.0:
                current_checkpoints.append(path)
    current_checkpoints.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]))
    if cfg.model_selection_last_k_checkpoints > 0 and len(current_checkpoints) > cfg.model_selection_last_k_checkpoints:
        current_checkpoints = current_checkpoints[-cfg.model_selection_last_k_checkpoints:]

    run_candidates: list[str] = []
    if trainer.state.best_model_checkpoint and os.path.isdir(trainer.state.best_model_checkpoint):
        run_candidates.append(trainer.state.best_model_checkpoint)
    for p in current_checkpoints:
        if p not in run_candidates:
            run_candidates.append(p)

    # Per-language: select best checkpoint by tweet-level objective on that
    # language's validation set, copy it into best_<lang>/ subdir, calibrate the
    # decision threshold there, and save a per-language decision config.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_lang_best_dirs: dict[str, str] = {}
    for lang_code, val_df_l in [("en", en_val_df), ("de", de_val_df)]:
        if val_df_l is None or len(val_df_l) == 0:
            continue
        val_texts_l = [str(x) for x in val_df_l["Text"].tolist()]
        val_labels_l = [int(x) for x in val_df_l["Biased"].tolist()]

        print(
            f"\nSelecting best checkpoint for {lang_code.upper()} from "
            f"{len(run_candidates)} candidate(s)..."
        )
        selected_ckpt, selected_metrics = _select_best_checkpoint_by_tweet_objective(
            output_dir=args.output_dir,
            val_texts=val_texts_l,
            val_labels=val_labels_l,
            lang=lang_code,
            tokenizer=tokenizer,
            cfg=cfg,
            num_labels=num_labels,
            candidate_checkpoints=run_candidates if run_candidates else None,
        )
        print(
            f"[{lang_code.upper()}] Tweet-level checkpoint selection: "
            f"objective={cfg.model_selection_objective} "
            f"min_precision={cfg.model_selection_min_precision:.2f} "
            f"selected='{selected_ckpt}' "
            f"p={selected_metrics['precision']:.4f} "
            f"r={selected_metrics['recall']:.4f} "
            f"f1={selected_metrics['f1']:.4f} "
            f"constraint_satisfied={selected_metrics['constraint_satisfied']}"
        )

        lang_best_dir = os.path.join(best_dir, f"best_{lang_code}")
        os.makedirs(lang_best_dir, exist_ok=True)
        chosen_model = AutoModelForSequenceClassification.from_pretrained(selected_ckpt, num_labels=num_labels)
        chosen_model.to(device)
        chosen_model.eval()
        chosen_model.save_pretrained(lang_best_dir)
        tokenizer.save_pretrained(lang_best_dir)

        print(
            f"\nCalibrating {lang_code.upper()} positive decision threshold on validation tweets "
            f"(best ckpt: {os.path.basename(selected_ckpt)})..."
        )
        thr_res = tune_positive_margin_threshold(
            texts=val_texts_l, true_labels=val_labels_l, lang=lang_code,
            tokenizer=tokenizer, model=chosen_model, device=device,
            objective=cfg.threshold_objective,
            threshold_min=cfg.threshold_search_min, threshold_max=cfg.threshold_search_max,
            min_precision=cfg.threshold_min_precision,
            contradiction_weight=cfg.contradiction_weight,
            class_0_weight=cfg.class_0_weight, score_mode=cfg.score_mode,
            nli_train_mode=cfg.nli_train_mode,
            progress_label=f"{lang_code.upper()} threshold calibration",
        )
        # Save the canonical decision_config.json inside the per-language best dir.
        save_decision_config(
            ckpt_dir=lang_best_dir,
            positive_margin_threshold=thr_res["threshold"],
            contradiction_weight=cfg.contradiction_weight,
            score_mode=cfg.score_mode, nli_train_mode=cfg.nli_train_mode,
            objective=cfg.threshold_objective, validation_metrics=thr_res,
        )
        # Default config in joint root = primary language (for backward compat).
        if lang_code == cfg.joint_eval_lang_for_best:
            save_decision_config(
                ckpt_dir=best_dir,
                positive_margin_threshold=thr_res["threshold"],
                contradiction_weight=cfg.contradiction_weight,
                score_mode=cfg.score_mode, nli_train_mode=cfg.nli_train_mode,
                objective=cfg.threshold_objective, validation_metrics=thr_res,
            )
        # Per-lang reference config in joint root.
        per_lang_path = os.path.join(best_dir, f"decision_config_{lang_code}.json")
        with open(per_lang_path, "w", encoding="utf-8") as f:
            json.dump({
                "positive_margin_threshold": float(thr_res["threshold"]),
                "contradiction_weight": float(cfg.contradiction_weight),
                "score_mode": str(cfg.score_mode),
                "nli_train_mode": str(cfg.nli_train_mode),
                "objective": cfg.threshold_objective,
                "lang": lang_code,
                "selected_checkpoint": selected_ckpt,
                "best_lang_dir": lang_best_dir,
                "validation_metrics": thr_res,
            }, f, indent=2, ensure_ascii=True)
        print(
            f"[{lang_code.upper()}] threshold: thr={thr_res['threshold']:.4f} "
            f"p={thr_res['precision']:.4f} r={thr_res['recall']:.4f} f1={thr_res['f1']:.4f} "
            f"constraint_satisfied={thr_res.get('constraint_satisfied', True)}"
        )
        per_lang_best_dirs[lang_code] = lang_best_dir

        del chosen_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_dir, en_test_csv, de_test_csv, per_lang_best_dirs


def train_binary_stage(
    train_csv: str,
    val_csv: str,
    lang: str,
    cfg: Config,
    resume_from: str | None = None,
    val_size: float | None = None,
    hold_test_size: float | None = None,
):
    """
    Direct 2-class tweet classifier training (non-NLI baseline).

    Returns:
        Tuple of (checkpoint_path, test_csv_path, lang)
    """
    source_model = resume_from if resume_from else cfg.model_name
    tokenizer = load_tokenizer(source_model)

    train_df, val_df, test_df, test_csv_path = _prepare_tweet_level_splits(
        train_csv=train_csv,
        val_csv=val_csv,
        val_size=val_size,
        hold_test_size=hold_test_size,
        seed=cfg.seed,
    )

    if lang == "en":
        oversample_factor = float(cfg.en_pos_oversample_factor)
    else:
        oversample_factor = float(cfg.de_pos_oversample_factor)

    if oversample_factor > 1.0:
        train_df = oversample_positive_rows(train_df, factor=oversample_factor, seed=cfg.seed)
        print(f"Applied positive oversampling factor: {oversample_factor:.2f}")

    print("\nTweet-level split summary (direct binary)")
    _print_class_distribution("Train", train_df)
    if val_df is not None:
        _print_class_distribution("Validation", val_df)
    if test_df is not None:
        _print_class_distribution("Test", test_df)

    train_ds = dataframe_to_binary_dataset(train_df)
    val_ds = dataframe_to_binary_dataset(val_df) if val_df is not None else None
    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds}) if val_ds else DatasetDict({"train": train_ds})
    tokenized = ds_dict.map(
        lambda ex: tokenize_binary_fn(ex, tokenizer),
        batched=True,
        remove_columns=ds_dict["train"].column_names,
    )

    model = AutoModelForSequenceClassification.from_pretrained(source_model, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="steps" if val_ds else "no",
        save_strategy="steps" if val_ds else "no",
        logging_steps=100,
        save_steps=300,
        eval_steps=300,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=10,
        seed=cfg.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if val_ds else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_binary if val_ds else None,
    )

    if val_ds:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=5))

    trainer.train()
    best_dir = args.output_dir
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    return best_dir, test_csv_path, lang


class JointTweetLevelCheckpointEvalCallback(TrainerCallback):
    """Tweet-level validation per language during joint training."""

    def __init__(self, en_val_df, de_val_df, tokenizer, cfg: Config):
        self.en_texts = [str(x) for x in en_val_df["Text"].tolist()] if en_val_df is not None else []
        self.en_labels = [int(x) for x in en_val_df["Biased"].tolist()] if en_val_df is not None else []
        self.de_texts = [str(x) for x in de_val_df["Text"].tolist()] if de_val_df is not None else []
        self.de_labels = [int(x) for x in de_val_df["Biased"].tolist()] if de_val_df is not None else []
        self.tokenizer = tokenizer
        self.cfg = cfg

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs.get("model")
        output_dir = args.output_dir or self.cfg.output_dir
        if model is None:
            return
        sink = metrics if isinstance(metrics, dict) else kwargs.get("metrics")
        if not isinstance(sink, dict):
            sink = None

        for lang_code, texts, labels in [("en", self.en_texts, self.en_labels), ("de", self.de_texts, self.de_labels)]:
            if not texts:
                continue
            preds = predict_texts_with_threshold(
                texts=texts, lang=lang_code, tokenizer=self.tokenizer, model=model,
                device=model.device,
                positive_margin_threshold=DEFAULT_POSITIVE_MARGIN_THRESHOLD,
                contradiction_weight=self.cfg.contradiction_weight,
                class_0_weight=self.cfg.class_0_weight,
                score_mode=self.cfg.score_mode,
                nli_train_mode=self.cfg.nli_train_mode,
                progress_label=f"Joint tweet-level {lang_code.upper()} step {state.global_step}",
            )
            m = _tweet_metrics(labels, preds)
            print(
                f"[{lang_code.upper()}] step={state.global_step} "
                f"p={m['precision']:.4f} r={m['recall']:.4f} f1={m['f1']:.4f}"
            )
            if sink is not None:
                sink[f"eval_tweet_precision_{lang_code}"] = float(m["precision"])
                sink[f"eval_tweet_recall_{lang_code}"] = float(m["recall"])
                sink[f"eval_tweet_f1_{lang_code}"] = float(m["f1"])
                sink[f"tweet_f1_{lang_code}"] = float(m["f1"])
            payload = {
                "checkpoint": os.path.join(output_dir, f"checkpoint-{state.global_step}"),
                "global_step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else -1.0,
                "lang": lang_code,
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1": float(m["f1"]),
            }
            _append_tweet_level_checkpoint_metrics(output_dir, payload)
        return


if __name__ == "__main__":
    import argparse
    from .evaluate import evaluate_with_analysis

    parser = argparse.ArgumentParser()
    parser.add_argument("--en_train", required=False, default="", help="Path to English training CSV")
    parser.add_argument(
        "--en_val",
        required=False,
        default="",
        help="Path to English validation CSV (optional if --val_size is used)",
    )
    parser.add_argument("--de_train", required=True, help="Path to German training CSV")
    parser.add_argument(
        "--de_val",
        required=False,
        default="",
        help="Path to German validation CSV (optional if --val_size is used)",
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs per stage.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--class_0_loss_weight",
        type=float,
        default=None,
        help="CE-loss weight for class-0 hypothesis rows (matches decision-time class_0_weight).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Ratio for validation split (e.g., 0.15 = 15%%). Ignores *_val arguments.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Ratio for test set hold-out (e.g., 0.15 = 15%%). This data is NOT used for training.",
    )
    parser.add_argument(
        "--pos_oversample_factor",
        type=float,
        default=None,
        help="Oversampling factor for positive tweets in training split (1.0 disables oversampling).",
    )
    parser.add_argument(
        "--positive_sample_weight",
        type=float,
        default=None,
        help="Loss weight multiplier for NLI rows originating from positive tweets.",
    )
    parser.add_argument(
        "--en_pos_oversample_factor",
        type=float,
        default=None,
        help="Oversampling factor for positive tweets in EN training split.",
    )
    parser.add_argument(
        "--de_pos_oversample_factor",
        type=float,
        default=None,
        help="Oversampling factor for positive tweets in DE training split.",
    )
    parser.add_argument(
        "--en_positive_sample_weight",
        type=float,
        default=None,
        help="Loss weight multiplier for positive tweets in EN stage.",
    )
    parser.add_argument(
        "--de_positive_sample_weight",
        type=float,
        default=None,
        help="Loss weight multiplier for positive tweets in DE stage.",
    )
    parser.add_argument(
        "--threshold_objective",
        type=str,
        default=None,
        choices=["f1", "recall", "recall_at_precision"],
        help="Objective for decision-threshold calibration on validation tweets.",
    )
    parser.add_argument(
        "--threshold_min_precision",
        type=float,
        default=None,
        help="Minimum precision constraint for threshold objective recall_at_precision.",
    )
    parser.add_argument(
        "--threshold_search_min",
        type=float,
        default=None,
        help="Lower bound for threshold calibration search range.",
    )
    parser.add_argument(
        "--threshold_search_max",
        type=float,
        default=None,
        help="Upper bound for threshold calibration search range.",
    )
    parser.add_argument(
        "--contradiction_weight",
        type=float,
        default=None,
        help="Hypothesis score weight for contradiction in score = entailment - weight * contradiction.",
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        default=None,
        choices=["entailment_only", "entailment_minus_contradiction"],
        help="Hypothesis score mode used in decision scoring.",
    )
    parser.add_argument(
        "--model_selection_objective",
        type=str,
        default=None,
        choices=["f1", "recall", "recall_at_precision"],
        help="Objective to choose best checkpoint using tweet-level validation metrics.",
    )
    parser.add_argument(
        "--model_selection_min_precision",
        type=float,
        default=None,
        help="Minimum precision constraint for model selection objective recall_at_precision.",
    )
    parser.add_argument(
        "--model_selection_last_k_checkpoints",
        type=int,
        default=None,
        help="Only score the latest K checkpoint-* directories from current run (0 disables limit).",
    )
    parser.add_argument(
        "--tweet_level_loss_weight",
        type=float,
        default=None,
        help="Auxiliary tweet-level margin loss weight added to NLI CE loss (0 disables).",
    )
    parser.add_argument(
        "--tweet_level_margin",
        type=float,
        default=None,
        help="Bias term for tweet-level margin loss; positive values require larger margin for class 1.",
    )
    parser.add_argument(
        "--class_0_weight",
        type=float,
        default=None,
        help="Weight for non-antisemitic (class 0) hypotheses (default: 0.7). Lower values reduce false negatives.",
    )
    parser.add_argument(
        "--nli_train_mode",
        type=str,
        default=None,
        choices=[
            "both_classes",
            "both_classes_contradiction",
            "both_classes_asymmetric_neutral",
            "class1_only",
        ],
        help=(
            "NLI training mode. 'both_classes'/'both_classes_contradiction' (default): all hypotheses "
            "from class 0 and 1, wrong-class pairs are contradiction. "
            "'both_classes_asymmetric_neutral': non-antisemitic tweets paired with class-1 hypotheses "
            "become neutral, while antisemitic tweets paired with class-0 hypotheses stay contradiction. "
            "'class1_only': only class-1 hypotheses; class-0 hypotheses are skipped during training "
            "and inference (score_0 is fixed at 0, margin = score_1)."
        ),
    )
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip automatic evaluation on test set after training")
    parser.add_argument(
        "--show_examples",
        type=int,
        default=10,
        help="Number of error examples to show in evaluation (default: 10)",
    )
    parser.add_argument(
        "--binary_train",
        action="store_true",
        help="Train direct binary classifier in two stages: EN pretraining then DE fine-tuning (no NLI).",
    )
    parser.add_argument(
        "--de_only",
        action="store_true",
        help="Train only on German data (no English pretraining stage).",
    )
    parser.add_argument(
        "--de_only_roberta",
        action="store_true",
        help="Train DE-only direct binary classifier (no NLI hypotheses, RoBERTa/XLM-R baseline).",
    )
    parser.add_argument(
        "--de_only_output_dir",
        type=str,
        default=os.path.join("checkpoints", "xlmr-nli", "de_only"),
        help="Output directory used when --de_only is enabled.",
    )
    parser.add_argument(
        "--de_only_roberta_output_dir",
        type=str,
        default=os.path.join("checkpoints", "xlmr-binary", "de_only"),
        help="Output directory used when --de_only_roberta is enabled.",
    )
    parser.add_argument(
        "--binary_output_dir",
        type=str,
        default=os.path.join("checkpoints", "xlmr-binary"),
        help="Root output directory used when --binary_train is enabled.",
    )
    parser.add_argument(
        "--joint_train",
        action="store_true",
        help="Train EN and DE jointly in a single stage (instead of EN -> DE fine-tuning).",
    )
    parser.add_argument(
        "--joint_output_dir",
        type=str,
        default=os.path.join("checkpoints", "xlmr-nli", "joint"),
        help="Output directory used when --joint_train is enabled.",
    )
    parser.add_argument(
        "--de_lang_loss_weight",
        type=float,
        default=None,
        help="Loss multiplier for DE rows in joint training (defaults to 2.0).",
    )
    parser.add_argument(
        "--joint_eval_lang_for_best",
        type=str,
        default=None,
        choices=["en", "de"],
        help="Which language tweet-F1 drives best-model selection in joint training.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path for the training/evaluation log file.",
    )
    args_ = parser.parse_args()

    if not args_.de_train:
        parser.error("--de_train is required")
    if not args_.de_only and not args_.de_only_roberta and not args_.binary_train and not args_.en_train:
        parser.error("--en_train is required unless --de_only, --de_only_roberta, or --binary_train is set")
    if args_.binary_train and not args_.en_train:
        parser.error("--en_train is required when --binary_train is set")

    cfg_overrides = {
        "model_name": args_.model_name,
        "epochs": args_.epochs,
        "lr": args_.lr,
        "batch_size": args_.batch_size,
        "pos_oversample_factor": args_.pos_oversample_factor,
        "positive_sample_weight": args_.positive_sample_weight,
        "en_pos_oversample_factor": args_.en_pos_oversample_factor,
        "de_pos_oversample_factor": args_.de_pos_oversample_factor,
        "en_positive_sample_weight": args_.en_positive_sample_weight,
        "de_positive_sample_weight": args_.de_positive_sample_weight,
        "threshold_objective": args_.threshold_objective,
        "threshold_min_precision": args_.threshold_min_precision,
        "threshold_search_min": args_.threshold_search_min,
        "threshold_search_max": args_.threshold_search_max,
        "contradiction_weight": args_.contradiction_weight,
        "score_mode": args_.score_mode,
        "nli_train_mode": args_.nli_train_mode,
        "class_0_weight": args_.class_0_weight,
        "class_0_loss_weight": args_.class_0_loss_weight,
        "model_selection_objective": args_.model_selection_objective,
        "model_selection_min_precision": args_.model_selection_min_precision,
        "model_selection_last_k_checkpoints": args_.model_selection_last_k_checkpoints,
        "tweet_level_loss_weight": args_.tweet_level_loss_weight,
        "tweet_level_margin": args_.tweet_level_margin,
        "de_lang_loss_weight": args_.de_lang_loss_weight,
        "joint_eval_lang_for_best": args_.joint_eval_lang_for_best,
    }
    cfg = Config(**{k: v for k, v in cfg_overrides.items() if v is not None})

    default_log_dir = (
        args_.binary_output_dir
        if args_.binary_train
        else (
            args_.de_only_roberta_output_dir
            if args_.de_only_roberta
            else (args_.de_only_output_dir if args_.de_only else (args_.joint_output_dir if args_.joint_train else cfg.output_dir))
        )
    )
    log_file = args_.log_file if args_.log_file else os.path.join(default_log_dir, "train.log")

    with redirect_output_to_log(log_file):
        print(f"Writing training/evaluation logs to: {os.path.abspath(log_file)}")

        if args_.joint_train:
            print("\n" + "=" * 80)
            print("JOINT EN+DE TRAINING (single stage)")
            print("=" * 80)
            cfg.output_dir = args_.joint_output_dir
            joint_out, en_test_csv, de_test_csv, per_lang_best_dirs = train_joint_stage(
                en_train_csv=args_.en_train,
                de_train_csv=args_.de_train,
                cfg=cfg,
                val_size=args_.val_size,
                hold_test_size=args_.test_size,
            )
            if not args_.skip_evaluation:
                print("\n" + "=" * 80)
                print("FINAL EVALUATION ON HELD-OUT TEST SETS (JOINT)")
                print("=" * 80)
                if en_test_csv is not None:
                    print("\n" + "-" * 80)
                    print("Evaluating EN test set...")
                    print("-" * 80)
                    en_eval_dir = per_lang_best_dirs.get("en", joint_out)
                    print(f"Using EN best checkpoint dir: {en_eval_dir}")
                    evaluate_with_analysis(
                        en_eval_dir, en_test_csv, "en",
                        show_examples=args_.show_examples,
                        class_0_weight=cfg.class_0_weight,
                    )
                if de_test_csv is not None:
                    print("\n" + "-" * 80)
                    print("Evaluating DE test set...")
                    print("-" * 80)
                    de_eval_dir = per_lang_best_dirs.get("de", joint_out)
                    print(f"Using DE best checkpoint dir: {de_eval_dir}")
                    evaluate_with_analysis(
                        de_eval_dir, de_test_csv, "de",
                        show_examples=args_.show_examples,
                        class_0_weight=cfg.class_0_weight,
                    )
        elif args_.binary_train:
            print("\n" + "=" * 80)
            print("STAGE 1: ENGLISH TRAINING (DIRECT BINARY, NO NLI)")
            print("=" * 80)
            cfg.output_dir = os.path.join(args_.binary_output_dir, "en")
            en_out, en_test_csv, en_lang = train_binary_stage(
                args_.en_train,
                args_.en_val,
                "en",
                cfg,
                resume_from=None,
                val_size=args_.val_size,
                hold_test_size=args_.test_size,
            )

            print("\n" + "=" * 80)
            print("STAGE 2: GERMAN FINE-TUNING (DIRECT BINARY, NO NLI)")
            print("=" * 80)
            cfg.output_dir = os.path.join(args_.binary_output_dir, "de_ft")
            de_out, de_test_csv, de_lang = train_binary_stage(
                args_.de_train,
                args_.de_val,
                "de",
                cfg,
                resume_from=en_out,
                val_size=args_.val_size,
                hold_test_size=args_.test_size,
            )

            if not args_.skip_evaluation:
                print("\n" + "=" * 80)
                print("FINAL EVALUATION ON HELD-OUT TEST SETS")
                print("=" * 80)
                if en_test_csv is not None:
                    print("\n" + "-" * 80)
                    print("Evaluating English direct binary model on held-out test set...")
                    print("-" * 80)
                    evaluate_binary_with_analysis(
                        en_out,
                        en_test_csv,
                        show_examples=args_.show_examples,
                    )
                if de_test_csv is not None:
                    print("\n" + "-" * 80)
                    print("Evaluating German direct binary model on held-out test set...")
                    print("-" * 80)
                    evaluate_binary_with_analysis(
                        de_out,
                        de_test_csv,
                        show_examples=args_.show_examples,
                    )
        elif args_.de_only_roberta:
            print("\n" + "=" * 80)
            print("GERMAN TRAINING (DE-ONLY DIRECT BINARY, NO NLI)")
            print("=" * 80)
            cfg.output_dir = args_.de_only_roberta_output_dir
            de_out, de_test_csv, de_lang = train_binary_stage(
                args_.de_train,
                args_.de_val,
                "de",
                cfg,
                resume_from=None,
                val_size=args_.val_size,
                hold_test_size=args_.test_size,
            )

            if not args_.skip_evaluation and de_test_csv is not None:
                print("\n" + "=" * 80)
                print("FINAL EVALUATION ON HELD-OUT TEST SET")
                print("=" * 80)
                print("\n" + "-" * 80)
                print("Evaluating German direct binary model on held-out test set...")
                print("-" * 80)
                evaluate_binary_with_analysis(
                    de_out,
                    de_test_csv,
                    show_examples=args_.show_examples,
                )
        elif args_.de_only:
            print("\n" + "=" * 80)
            print("GERMAN TRAINING (DE-ONLY, NO ENGLISH PRETRAINING)")
            print("=" * 80)
            cfg.output_dir = args_.de_only_output_dir
            de_out, de_test_csv, de_lang = train_stage(
                args_.de_train,
                args_.de_val,
                "de",
                cfg,
                resume_from=None,
                val_size=args_.val_size,
                hold_test_size=args_.test_size,
            )

            if not args_.skip_evaluation and de_test_csv is not None:
                print("\n" + "=" * 80)
                print("FINAL EVALUATION ON HELD-OUT TEST SET")
                print("=" * 80)
                print("\n" + "-" * 80)
                print("Evaluating German model on held-out test set...")
                print("-" * 80)
                evaluate_with_analysis(
                    de_out,
                    de_test_csv,
                    de_lang,
                    show_examples=args_.show_examples,
                    class_0_weight=cfg.class_0_weight,
                )
        else:
            print("\n" + "=" * 80)
            print("STAGE 1: ENGLISH TRAINING")
            print("=" * 80)
            en_out, en_test_csv, en_lang = train_stage(
                args_.en_train,
                args_.en_val,
                "en",
                cfg,
                resume_from=None,
                val_size=args_.val_size,
                hold_test_size=args_.test_size,
            )

            print("\n" + "=" * 80)
            print("STAGE 2: GERMAN FINE-TUNING")
            print("=" * 80)
            cfg.output_dir = os.path.join(cfg.output_dir, "de_ft")
            de_out, de_test_csv, de_lang = train_stage(
                args_.de_train,
                args_.de_val,
                "de",
                cfg,
                resume_from=en_out,
                val_size=args_.val_size,
                hold_test_size=args_.test_size,
            )

            if not args_.skip_evaluation:
                print("\n" + "=" * 80)
                print("FINAL EVALUATION ON HELD-OUT TEST SETS")
                print("=" * 80)

                if en_test_csv is not None:
                    print("\n" + "-" * 80)
                    print("Evaluating English model on held-out test set...")
                    print("-" * 80)
                    evaluate_with_analysis(
                        en_out,
                        en_test_csv,
                        en_lang,
                        show_examples=args_.show_examples,
                        class_0_weight=cfg.class_0_weight,
                    )

                if de_test_csv is not None:
                    print("\n" + "-" * 80)
                    print("Evaluating German model on held-out test set...")
                    print("-" * 80)
                    evaluate_with_analysis(
                        de_out,
                        de_test_csv,
                        de_lang,
                        show_examples=args_.show_examples,
                        class_0_weight=cfg.class_0_weight,
                    )

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
