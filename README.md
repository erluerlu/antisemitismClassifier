# antisemitismClassifier

# Antisemitism Classifier - Training & Evaluation Guide

## Overview

This guide describes the current end-to-end training pipeline implemented in `src/train.py` and the tweet-level decision logic in `src/decision.py`.

Core design goals:
- Train an NLI model for binary tweet-level antisemitism detection (`0`/`1`).
- Select checkpoints using tweet-level metrics, not only pair-level NLI loss.
- Tune a margin threshold on validation tweets with configurable precision constraints.
- Keep EN and DE class-imbalance handling separate.

Pipeline modes:
- Two-stage mode (default):
    - Stage 1: English training (`en`)
    - Stage 2: German fine-tuning (`de`) from Stage 1 output
- German-only mode:
    - Single stage on DE data (`--de_only`)

## Data Flow

### 1. Tweet-level split before NLI expansion

Splits are created on original tweet rows before premise-hypothesis expansion.

Default behavior:
- `--test_size 0.15`: creates a held-out test split and writes:
    - `*_test_holdout.csv`
    - `*_train_val.csv`
- `--val_size 0.15`: creates validation split from train/val pool.

Dynamic split detail:
- If both `test_size` and `val_size` are used, validation size is adjusted relative to the remaining train/val pool:
    - `relative_val_size = val_size / (1 - test_size)`

### 2. NLI expansion and tokenization

Each tweet is expanded into NLI examples using hypotheses per class and language from `src/hypotheses.py`.

Tokenized fields include:
- model inputs (`input_ids`, `attention_mask`, ...)
- `labels` (NLI class)
- `target_cls` (tweet class `0/1`, used for weighted loss)
- `hyp_cls`

## Class Imbalance Handling

The pipeline uses language-specific settings during training stage execution.

### Dynamic value determination by stage language

In `train_stage(...)`, values are selected by `lang`:
- For `en`:
    - oversampling factor = `en_pos_oversample_factor`
    - positive weight = `en_positive_sample_weight`
- For `de`:
    - oversampling factor = `de_pos_oversample_factor`
    - positive weight = `de_positive_sample_weight`

Defaults:
- `en_pos_oversample_factor = 1.5`
- `de_pos_oversample_factor = 2.0`
- `en_positive_sample_weight = 3.0`
- `de_positive_sample_weight = 10.0`

Mechanisms:
- Oversampling (data-level): duplicates positive tweets in training data.
- Weighted loss (loss-level): multiplies CE loss for samples with `target_cls == 1`.

Note:
- Legacy CLI args `--pos_oversample_factor` and `--positive_sample_weight` still exist for compatibility and are stored in config, but stage execution currently uses the EN/DE-specific values above.

## Training Configuration

Fixed defaults from `Config`:
- `model_name = xlm-roberta-base`
- `lr = 2e-5`
- `epochs = 8`
- `batch_size = 16`
- `gradient_accumulation_steps = 8`
- `weight_decay = 0.01`
- `warmup_ratio = 0.06`
- `seed = 42`

Trainer runtime settings:
- `eval_strategy = steps` (if validation exists)
- `save_strategy = steps` (if validation exists)
- `eval_steps = 200`
- `save_steps = 200`
- `logging_steps = 100`
- `save_total_limit = 5`
- `load_best_model_at_end = True` (if validation exists)
- HF best model criterion:
    - `metric_for_best_model = eval_loss`
    - `greater_is_better = False`

Early stopping:
- Enabled when validation exists with `EarlyStoppingCallback(early_stopping_patience=3)`.

## Dynamic Checkpoint Candidate Selection

After `trainer.train()` finishes, checkpoint selection for final model uses tweet-level metrics:

1. Collect current-run checkpoints dynamically via directory modification time:
- Include `checkpoint-*` directories in output dir where `mtime >= run_start_ts - 1.0`.

2. Apply candidate limiting:
- Keep latest K current-run checkpoints (`model_selection_last_k_checkpoints`, default `3`).
- If `K=0`, no limit is applied.

3. Safety candidate:
- Add HF best checkpoint (`trainer.state.best_model_checkpoint`) if available.

4. Fallback behavior:
- If no run candidates are detected, the selector falls back to all available candidate directories in output dir (including root model dir when present).

### Tweet-level checkpoint objective

Supported objectives:
- `f1`
- `recall`
- `recall_at_precision`

Ranking logic:
- `f1`: rank by `(f1, recall, precision)`
- `recall`: rank by `(recall, f1, precision)`
- `recall_at_precision`: rank by `(constraint, recall, f1, precision)` where `constraint = 1 if precision >= min_precision else 0`

Default selection objective settings (effective CLI defaults):
- `--model_selection_objective recall_at_precision`
- `--model_selection_min_precision 0.20`

## Decision Scoring and Threshold Calibration

### Per-class score aggregation (dynamic over hypotheses)

For each class (`0`, `1`):
- Compute entailment probability for each hypothesis.
- Sort class hypothesis scores descending.
- Aggregate with top-k mean (`k=2`, or fewer if fewer hypotheses exist).

This replaces strict max aggregation and reduces one-hypothesis spikes.

### Threshold tuning

Prediction rule:
- margin = `score_1 - score_0`
- predict class `1` iff `margin >= threshold`

Threshold search defaults:
- `threshold_min = -0.5`
- `threshold_max = 0.5`
- `threshold_steps = 401`

Dynamic threshold objective:
- `--threshold_objective` in `{f1, recall, recall_at_precision}`
- `--threshold_min_precision` used when objective is `recall_at_precision`

Fallback behavior:
- If no threshold satisfies precision constraint, fallback returns recall-optimal threshold and marks `constraint_satisfied = false`.

Saved artifact:
- `decision_config.json` in checkpoint dir stores selected threshold and validation metrics.

## Evaluation Flow

If `--skip_evaluation` is not set:
- Two-stage mode evaluates EN and DE held-out sets.
- DE-only mode evaluates only DE held-out set.

Evaluation script (`src.evaluate`) reports:
- Accuracy
- Positive precision/recall/F1
- Confusion matrix
- Error examples (TP/FP/FN samples)

## CLI Reference (Current)

### Required inputs
- `--de_train`
- `--en_train` required unless `--de_only` is set

### Data and split
- `--en_train`
- `--de_train`
- `--en_val`
- `--de_val`
- `--val_size` (default `0.15`)
- `--test_size` (default `0.15`)

### Base model and stage mode
- `--model_name` (default `xlm-roberta-base`)
- `--de_only`
- `--de_only_output_dir` (default `checkpoints/xlmr-nli/de_only`)

### Imbalance controls
- Legacy/general:
    - `--pos_oversample_factor` (default `5.0`)
    - `--positive_sample_weight` (default `5.0`)
- Stage-specific (actively used by current train loop):
    - `--en_pos_oversample_factor` (default `1.5`)
    - `--de_pos_oversample_factor` (default `2.0`)
    - `--en_positive_sample_weight` (default `3.0`)
    - `--de_positive_sample_weight` (default `10.0`)

### Threshold calibration
- `--threshold_objective` (`f1|recall|recall_at_precision`, default `recall_at_precision`)
- `--threshold_min_precision` (default `0.20`)
- `--threshold_search_min` (default `-0.5`)
- `--threshold_search_max` (default `0.5`)

### Checkpoint model selection
- `--model_selection_objective` (`f1|recall|recall_at_precision`, default `recall_at_precision`)
- `--model_selection_min_precision` (default `0.20`)
- `--model_selection_last_k_checkpoints` (default `3`, `0` = no K-limit)

### Output and reporting
- `--skip_evaluation`
- `--show_examples` (default `10`)

## Recommended Commands

### Full EN -> DE pipeline

```bash
C:/Users/erikk/miniconda3/envs/antisemitism-nli/python.exe -m src.train --en_train data/en_cleaned.csv --de_train data/de_cleaned.csv --threshold_objective recall_at_precision --threshold_min_precision 0.20 --threshold_search_min -0.5 --threshold_search_max 0.5 --model_selection_objective recall_at_precision --model_selection_min_precision 0.20 --en_pos_oversample_factor 1.5 --de_pos_oversample_factor 2.0 --en_positive_sample_weight 3.0 --de_positive_sample_weight 10.0
```

### Fast experiment (skip final evaluation)

```bash
C:/Users/erikk/miniconda3/envs/antisemitism-nli/python.exe -m src.train --en_train data/en_cleaned.csv --de_train data/de_cleaned.csv --skip_evaluation
```

### German-only baseline

```bash
C:/Users/erikk/miniconda3/envs/antisemitism-nli/python.exe -m src.train --de_only --de_train data/de_cleaned.csv --de_only_output_dir checkpoints/xlmr-nli/de_only
```

### Manual evaluation

```bash
C:/Users/erikk/miniconda3/envs/antisemitism-nli/python.exe -m src.evaluate --checkpoint checkpoints/xlmr-nli/de_ft --test_data data/de_cleaned_test_holdout.csv --lang de --show_examples 10
```

## Practical Interpretation Notes

- Accuracy alone is not enough for imbalanced antisemitism detection.
- Track positive precision and recall jointly.
- Very negative thresholds push recall up and can explode false positives.
- The precision-constrained objectives and threshold range bounds are the key controls for this trade-off.

## Outputs and Artifacts

- Stage 1 model dir: `checkpoints/xlmr-nli/`
- Stage 2 model dir: `checkpoints/xlmr-nli/de_ft/`
- DE-only model dir: configurable via `--de_only_output_dir`
- Holdout/train-val split CSVs: written next to source CSVs
- Decision config: `decision_config.json` in each stage output dir

## Troubleshooting

- Training seems stuck:
    - Wait for progress logs during checkpoint scoring and threshold calibration.
    - These phases run full tweet-level scoring loops and can take time.

- Unexpected checkpoint selected:
    - Check objective/min_precision settings.
    - Check `model_selection_last_k_checkpoints` and fallback behavior.

- Need current runtime flags:

```bash
C:/Users/erikk/miniconda3/envs/antisemitism-nli/python.exe -m src.train --help
```
