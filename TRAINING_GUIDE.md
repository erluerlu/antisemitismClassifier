# Antisemitism Classifier - Training & Evaluation Guide

## Overview

The current training pipeline is optimized for tweet-level detection quality on the positive class (antisemitic = `1`), with explicit recall/precision control.

Pipeline stages:
- Stage 1: English training (`en`)
- Stage 2: German fine-tuning (`de`) from stage 1 checkpoint
- Optional automatic evaluation on held-out test sets

## Current Training Logic

### 1. Leak-free splitting
- Splits are done on original tweet rows BEFORE NLI expansion.
- Default split settings:
    - `--test_size 0.15`
    - `--val_size 0.15`

### 2. Imbalance handling (current defaults)
- Positive oversampling: `--pos_oversample_factor 5.0`
- Positive loss weighting: `--positive_sample_weight 5.0`

### 3. Checkpoint selection (tweet-level)
- During each stage, checkpoints are scored on tweet-level metrics on validation data.
- Best checkpoint is selected using:
    - `--model_selection_objective` (default: `recall_at_precision`)
    - `--model_selection_min_precision` (default: `0.20`)

### 4. Threshold calibration (tweet-level)
- A decision threshold on margin `(score_1 - score_0)` is tuned on validation tweets.
- Controlled by:
    - `--threshold_objective` (default: `recall_at_precision`)
    - `--threshold_min_precision` (default: `0.20`)

### 5. Progress logs for long scoring phases
- Long-running checkpoint scoring and threshold calibration now print progress.
- This avoids the "looks stuck" behavior during expensive validation scoring.

## Recommended Training Command

Use PowerShell and run as one command line:

```bash
C:/Users/erikk/miniconda3/envs/antisemitism-nli/python.exe -m src.train --en_train data/en_cleaned.csv --de_train data/de_cleaned.csv --threshold_objective recall_at_precision --threshold_min_precision 0.20 --model_selection_objective recall_at_precision --model_selection_min_precision 0.20
```

If you want to skip final test evaluation during experiments:

```bash
C:/Users/erikk/miniconda3/envs/antisemitism-nli/python.exe -m src.train --en_train data/en_cleaned.csv --de_train data/de_cleaned.csv --skip_evaluation
```

## Full CLI Options (Current)

- `--en_train` (required): path to English training CSV
- `--de_train` (required): path to German training CSV
- `--en_val` (optional): English validation CSV (ignored when `--val_size` is used)
- `--de_val` (optional): German validation CSV (ignored when `--val_size` is used)
- `--model_name` (default: `xlm-roberta-base`)
- `--val_size` (default: `0.15`)
- `--test_size` (default: `0.15`)
- `--pos_oversample_factor` (default: `5.0`)
- `--positive_sample_weight` (default: `5.0`)
- `--threshold_objective` (`f1|recall|recall_at_precision`, default: `recall_at_precision`)
- `--threshold_min_precision` (default: `0.20`)
- `--model_selection_objective` (`f1|recall|recall_at_precision`, default: `recall_at_precision`)
- `--model_selection_min_precision` (default: `0.20`)
- `--skip_evaluation` (flag)
- `--show_examples` (default: `10`)

## Evaluation

Manual detailed evaluation:

```bash
C:/Users/erikk/miniconda3/envs/antisemitism-nli/python.exe -m src.evaluate --checkpoint checkpoints/xlmr-nli/de_ft --test_data data/de_cleaned_test_holdout.csv --lang de --show_examples 10
```

Key tweet-level metrics to watch:
- `Positive Recall`: how many antisemitic tweets are found
- `Positive Precision`: false-positive control
- `Positive F1`: balance between precision and recall
- Confusion matrix (`TP/FP/FN/TN`)

## Notes on Metric Interpretation

- High accuracy can be misleading on highly imbalanced datasets.
- For deployment goals like "catch as many antisemitic tweets as possible", prioritize positive-class recall under a minimum precision constraint.
- NLI pair-level metrics alone are not sufficient; use tweet-level metrics for model selection and decision threshold tuning.

## Output Locations

- Stage 1 checkpoint: `checkpoints/xlmr-nli/`
- Stage 2 checkpoint: `checkpoints/xlmr-nli/de_ft/`
- Threshold config per stage: `decision_config.json` inside the checkpoint directory

## Troubleshooting

- If training appears frozen:
    - Wait for progress prints during validation scoring.
    - Verify you are running in PowerShell (not inside an interactive Python REPL).
    - Ensure the command is not accidentally concatenated with another command.
- If you need to inspect available flags quickly:

```bash
C:/Users/erikk/miniconda3/envs/antisemitism-nli/python.exe -m src.train --help
```
