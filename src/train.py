import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import evaluate
from datasets import DatasetDict
from .dataset import load_binary_dataframe, dataframe_to_nli_dataset, oversample_positive_rows
from .hypotheses import NLI_LABEL2ID
from .tokenizer_utils import load_tokenizer
from .decision import (
    DEFAULT_POSITIVE_MARGIN_THRESHOLD,
    predict_texts_with_threshold,
    save_decision_config,
    tune_positive_margin_threshold,
)
import torch


@dataclass
class Config:
    """Training configuration parameters."""

    model_name: str = "xlm-roberta-base"
    output_dir: str = "checkpoints/xlmr-nli"
    lr: float = 2e-5
    epochs: int = 8
    batch_size: int = 16
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    pos_oversample_factor: float = 5.0
    positive_sample_weight: float = 5.0
    threshold_objective: str = "recall"
    threshold_min_precision: float = 0.20
    model_selection_objective: str = "recall_at_precision"
    model_selection_min_precision: float = 0.20


def _print_class_distribution(name: str, df: pd.DataFrame):
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
    return encoded


def compute_metrics(eval_pred):
    """Computes pair-level metrics for trainer logging."""
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
    labels_list = labels.tolist() if hasattr(labels, "tolist") else list(labels)

    acc_res = metric_acc.compute(predictions=preds_list, references=labels_list)
    f1_macro = metric_f1.compute(predictions=preds_list, references=labels_list, average="macro")
    f1_entail = metric_f1.compute(predictions=preds_list, references=labels_list, average="binary", pos_label=NLI_LABEL2ID["entailment"])

    out = {k: float(v) for k, v in acc_res.items()}
    out["f1"] = float(f1_macro["f1"])
    out["f1_entailment"] = float(f1_entail["f1"])
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

    if cfg.pos_oversample_factor > 1.0:
        train_df = oversample_positive_rows(train_df, factor=cfg.pos_oversample_factor, seed=cfg.seed)
        print(f"Applied positive oversampling factor: {cfg.pos_oversample_factor:.2f}")

    print("\nTweet-level split summary")
    _print_class_distribution("Train", train_df)
    if val_df is not None:
        _print_class_distribution("Validation", val_df)
    if test_df is not None:
        _print_class_distribution("Test", test_df)

    train_ds = dataframe_to_nli_dataset(train_df, lang)
    val_ds = dataframe_to_nli_dataset(val_df, lang) if val_df is not None else None

    if test_df is not None:
        test_nli_len = len(dataframe_to_nli_dataset(test_df, lang))
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
        labels = inputs.pop("labels", None)
        target_cls = inputs.pop("target_cls", None)
        _ = inputs.pop("hyp_cls", None)

        if labels is None:
            raise ValueError("'labels' not found in batch inputs. Available keys: " + str(list(inputs.keys())))

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        sample_losses = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        if target_cls is not None and cfg.positive_sample_weight != 1.0:
            cls_tensor = target_cls.view(-1).to(sample_losses.device)
            sample_weights = torch.where(
                cls_tensor == 1,
                torch.full_like(sample_losses, float(cfg.positive_sample_weight)),
                torch.ones_like(sample_losses),
            )
            loss = (sample_losses * sample_weights).mean()
        else:
            loss = sample_losses.mean()

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
        save_steps=500,
        eval_steps=100,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # Keep run start timestamp so we can detect checkpoints written/updated by this run.
    run_start_ts = time.time()

    trainer.train()
    best_dir = args.output_dir

    # Select best checkpoint by tweet-level objective to align with final task goal.
    if val_df is not None and len(val_df) > 0:
        val_texts = [str(x) for x in val_df["Text"].tolist()]
        val_labels = [int(x) for x in val_df["Biased"].tolist()]

        current_checkpoints: list[str] = []
        if os.path.isdir(args.output_dir):
            for name in os.listdir(args.output_dir):
                path = os.path.join(args.output_dir, name)
                if not (os.path.isdir(path) and name.startswith("checkpoint-")):
                    continue

                # Treat checkpoint dirs as part of this run when they were modified after run start.
                # This correctly includes overwritten folders like checkpoint-500 from repeated runs.
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if mtime >= run_start_ts - 1.0:
                    current_checkpoints.append(path)
        current_checkpoints.sort(key=lambda p: int(os.path.basename(p).split("-")[-1]))

        # Prefer checkpoints produced in this run; add HF-selected best checkpoint as a safety candidate.
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
            min_precision=cfg.threshold_min_precision,
            progress_label="Threshold calibration scoring",
        )
        cfg_path = save_decision_config(
            ckpt_dir=best_dir,
            positive_margin_threshold=threshold_result["threshold"],
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
            f"constraint_satisfied={threshold_result.get('constraint_satisfied', True)}"
        )
        print(f"Saved decision config: {cfg_path}")
    else:
        cfg_path = save_decision_config(
            ckpt_dir=best_dir,
            positive_margin_threshold=DEFAULT_POSITIVE_MARGIN_THRESHOLD,
            objective="default",
            validation_metrics={},
        )
        print(f"Saved default decision config: {cfg_path}")

    return best_dir, test_csv_path, lang


if __name__ == "__main__":
    import argparse
    from .evaluate import evaluate_with_analysis

    parser = argparse.ArgumentParser()
    parser.add_argument("--en_train", required=True, help="Path to English training CSV")
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
    parser.add_argument("--model_name", default="xlm-roberta-base")
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
        default=5.0,
        help="Oversampling factor for positive tweets in training split (1.0 disables oversampling).",
    )
    parser.add_argument(
        "--positive_sample_weight",
        type=float,
        default=5.0,
        help="Loss weight multiplier for NLI rows originating from positive tweets.",
    )
    parser.add_argument(
        "--threshold_objective",
        type=str,
        default="recall_at_precision",
        choices=["f1", "recall", "recall_at_precision"],
        help="Objective for decision-threshold calibration on validation tweets.",
    )
    parser.add_argument(
        "--threshold_min_precision",
        type=float,
        default=0.20,
        help="Minimum precision constraint for threshold objective recall_at_precision.",
    )
    parser.add_argument(
        "--model_selection_objective",
        type=str,
        default="recall_at_precision",
        choices=["f1", "recall", "recall_at_precision"],
        help="Objective to choose best checkpoint using tweet-level validation metrics.",
    )
    parser.add_argument(
        "--model_selection_min_precision",
        type=float,
        default=0.20,
        help="Minimum precision constraint for model selection objective recall_at_precision.",
    )
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip automatic evaluation on test set after training")
    parser.add_argument(
        "--show_examples",
        type=int,
        default=10,
        help="Number of error examples to show in evaluation (default: 10)",
    )
    args_ = parser.parse_args()

    cfg = Config(
        model_name=args_.model_name,
        pos_oversample_factor=args_.pos_oversample_factor,
        positive_sample_weight=args_.positive_sample_weight,
        threshold_objective=args_.threshold_objective,
        threshold_min_precision=args_.threshold_min_precision,
        model_selection_objective=args_.model_selection_objective,
        model_selection_min_precision=args_.model_selection_min_precision,
    )

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
            evaluate_with_analysis(en_out, en_test_csv, en_lang, show_examples=args_.show_examples)

        if de_test_csv is not None:
            print("\n" + "-" * 80)
            print("Evaluating German model on held-out test set...")
            print("-" * 80)
            evaluate_with_analysis(de_out, de_test_csv, de_lang, show_examples=args_.show_examples)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
