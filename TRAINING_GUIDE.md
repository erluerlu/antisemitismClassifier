# Antisemitism Classifier - Training & Evaluation Guide

## Automated Train/Val/Test Pipeline

The training pipeline now automatically performs the following steps:

### 1. Data Splitting
- **Test Set Hold-out**: 15% of data is separated BEFORE training
- **Validation Set**: 15% of remaining data for early stopping
- **Training Set**: 70% of data for actual training

### 2. Training
- Stage 1: English base training
- Stage 2: German fine-tuning

### 3. Automatic Evaluation
After training, the models are automatically evaluated on **unseen test sets**.

## Usage

### Training with Automatic Evaluation

```bash
python -m src.train \
    --en_train data/en_cleaned.csv \
    --de_train data/de_cleaned.csv \
    --val_size 0.15 \
    --test_size 0.15
```

**Parameters:**
- `--val_size`: Validation split ratio (e.g., 0.15 = 15%)
- `--test_size`: Test hold-out ratio (e.g., 0.15 = 15%)
- `--skip_evaluation`: Optional - skip automatic test evaluation

### Data Flow Example

With 8,048 German samples:
```
Original Data (8,048 samples)
    ↓
[Test Hold-out: 15%]
    ├─→ Test Set: 1,207 samples (held out)
    └─→ Train+Val: 6,841 samples
            ↓
        [Val Split: 15%]
            ├─→ Validation: 1,026 samples
            └─→ Training: 5,815 samples
```

After training → Automatic evaluation on 1,207 held-out test samples

### Manual Evaluation

For later manual evaluation:

```bash
python -m src.evaluate \
    --checkpoint checkpoints/xlmr-nli/de_ft \
    --test_data data/de_cleaned.csv \
    --lang de
```

## Pipeline Benefits

✅ **No Data Leakage**: Test set never used for training  
✅ **Automated**: Everything in one run  
✅ **Reproducible**: Fixed seeds for consistent splits  
✅ **Transparent**: Clear output about data sizes  

## Latest Results (February 2026)

Training completed successfully with the following performance on held-out test sets:

| Model | Test Samples | Accuracy | F1-Score | Loss | Runtime |
|-------|--------------|----------|----------|------|---------|
| **English (Stage 1)** | 3,394 NLI | **90.45%** | **90.45%** | 0.2976 | 5.17s |
| **German (Stage 2)** | 2,415 NLI | **97.60%** | **97.60%** | 0.1120 | 4.13s |

**Training Duration:**
- Stage 1 (English): ~7.8 minutes (469s)
- Stage 2 (German): ~6.2 minutes (372s)
- **Total**: ~14 minutes

**Key Findings:**
- German fine-tuning shows **7.15 percentage points better performance** than the English base model
- Excellent generalization on unseen data
- No data leakage - test sets were completely held out from training
- The German model achieves 97.60% accuracy for antisemitism detection

## Sample Output

```
================================================================================
STAGE 1: ENGLISH TRAINING
================================================================================

================================================================================
Holding out 15% of data as test set (NOT used for training)
================================================================================
Test set: 3394 NLI samples (held out for final evaluation)
Splitting remaining 19228 NLI samples into train/val (82%/18%)
Train set: 15834 NLI samples
Validation set: 3394 NLI samples

[Training in progress...]

================================================================================
STAGE 2: GERMAN FINE-TUNING
================================================================================

================================================================================
Holding out 15% of data as test set (NOT used for training)
================================================================================
Test set: 2415 NLI samples (held out for final evaluation)
Splitting remaining 13681 NLI samples into train/val (82%/18%)
Train set: 11266 NLI samples
Validation set: 2415 NLI samples

[Training in progress...]

================================================================================
FINAL EVALUATION ON HELD-OUT TEST SETS
================================================================================

--------------------------------------------------------------------------------
Evaluating German model on held-out test set...
--------------------------------------------------------------------------------
Loading checkpoint from: checkpoints/xlmr-nli/de_ft
Language: de
Test samples: 2415
--------------------------------------------------------------------------------
Running evaluation...

================================================================================
EVALUATION RESULTS
================================================================================
Accuracy:  0.9760 (97.60%)
F1-Score:  0.9760 (97.60%)
Loss:      0.1120
Runtime:   4.13s
Samples/s: 584.19
================================================================================

================================================================================
TRAINING COMPLETE
================================================================================
```

## Model Checkpoints

Trained models are saved in:
- English model: `checkpoints/xlmr-nli/`
- German model: `checkpoints/xlmr-nli/de_ft/`

These checkpoints can be used for inference or further evaluation.
