# Antisemitism Classifier — Training & Evaluation Guide

End-to-end pipeline for binary tweet-level antisemitism detection (`Biased ∈ {0,1}`)
based on an NLI-formulation: each tweet is paired with a set of natural-language
hypotheses, the model outputs entailment / neutral / contradiction probabilities,
and a margin-based decision rule produces the final label.

Backbone: `xlm-roberta-base` (3-class NLI head). Multilingual: EN + DE.

---

## 1. Pipeline Modes

The CLI supports three training modes. Pick exactly one.

| Mode | Flag | What it does | Output dir |
|---|---|---|---|
| Two-stage  | (default) | Train on EN, then fine-tune on DE | `checkpoints/xlmr-nli/` (EN), `checkpoints/xlmr-nli/de_ft/` (DE) |
| German only | `--de_only` | Single stage on DE | `checkpoints/xlmr-nli/de_only/` |
| Joint EN+DE | `--joint_train` | Single stage on EN + DE concatenated | `checkpoints/xlmr-nli/joint/` |

---

## 2. Data Pipeline

### 2.1 Tweet-level splits (leakage-free)

Splits happen on raw tweet rows BEFORE NLI expansion, otherwise a single tweet's pair-rows would leak into both train and val.

- `--test_size 0.15` — held-out test set (never seen during training/threshold calibration). Saved as `*_test_holdout.csv` next to the source CSV.
- `--val_size 0.15` — validation split from the remaining train pool. Adjusted to `val_size / (1 - test_size)` so the absolute fraction matches `val_size`.
- Stratified by `Biased` to preserve class balance in each split.

### 2.2 NLI expansion ([`src/dataset.py`](src/dataset.py))

For each tweet, the row is expanded into one NLI pair `(premise=text, hypothesis=h)` per hypothesis from [`src/hypotheses.py`](src/hypotheses.py).

Hypothesis classes:
- Class `1` (antisemitic): 6 hypotheses each in DE/EN, covering dehumanization, conspiracy, collective blame, Israel delegitimization/Nazi-equation, calls for violence, and Holocaust denial.
- Class `0` (non-antisemitic): 3 hypotheses each, covering factual statements, government criticism without anti-Jewish framing, and respectful discussion of Jewish life.

`nli_label` per pair is determined by `--nli_train_mode`:

| Mode | Tweet=0, Hyp class=1 | Tweet=1, Hyp class=0 | Tweet=cls, Hyp=cls |
|---|---|---|---|
| `both_classes_contradiction` (default) | contradiction | contradiction | entailment |
| `both_classes_asymmetric_neutral` (recommended) | **neutral** | contradiction | entailment |
| `class1_only` | contradiction | (skipped) | entailment |

Asymmetric neutral works best in practice: a non-antisemitic tweet does not actively contradict the hypothesis "the text dehumanizes Jews" — it is simply unrelated. Training that as `neutral` matches the semantics and prevents the model from being pushed to contradict everything.

Each NLI row carries:
- `labels` (NLI class id)
- `target_cls` (tweet class 0/1) — for positive-tweet upweighting
- `hyp_cls` (hypothesis class 0/1) — for class-0 hypothesis dampening
- `tweet_id` — for tweet-level margin loss
- `lang_id` (0=EN, 1=DE) — for joint-training language weighting

---

## 3. Loss & Fine-tuning

The standard HuggingFace `Trainer` is used with a custom `compute_loss` that combines two terms.

### 3.1 Weighted cross-entropy (per NLI pair)

Per-sample weight (multiplied together):

```
w = 1.0
  * positive_sample_weight   if target_cls == 1
  * class_0_loss_weight      if hyp_cls == 0
  * de_lang_loss_weight      if lang_id == 1   (joint mode only)
```

Then `ce_loss = sum(sample_loss * w) / sum(w)`.

Why three factors:
- **Positive upweighting** counters the ~4 % DE positive rate.
- **Class-0 hypothesis dampening** mirrors the decision-time `class_0_weight` so training and inference optimize the same signal.
- **DE language weight** (joint mode) prevents the larger EN dataset from dominating gradients.

### 3.2 Tweet-level margin loss (auxiliary)

Independent of CE, a margin objective enforces the actual decision rule on raw entailment logits within each batch's tweet group:

```
For each tweet in batch:
  pos_logit = max(entail_logits over rows with hyp_cls==1)
  neg_logit = max(entail_logits over rows with hyp_cls==0)
  margin    = pos_logit - neg_logit
loss_tweet = BCEWithLogits(margin - tweet_level_margin, tweet_label)

total_loss = ce_loss + tweet_level_loss_weight * loss_tweet
```

This pushes the model to make `pos_logit > neg_logit` whenever the tweet is positive, directly aligning the training objective with the inference-time decision rule. We use raw logits (not softmax probabilities) so gradients are well-scaled.

Default `tweet_level_loss_weight = 0.2`. Set to `0` to disable.

### 3.3 Optimizer & schedule

Defaults from `Config`:

| Param | Value |
|---|---|
| `lr` | `2e-5` |
| `epochs` | `4` |
| `batch_size` | `16` |
| `gradient_accumulation_steps` | `8` (effective batch 128) |
| `weight_decay` | `0.01` |
| `warmup_ratio` | `0.06` |
| `seed` | `42` |
| `eval_steps` / `save_steps` | `300` |
| `save_total_limit` | `10` |

### 3.4 Best-checkpoint selection

Every `eval_steps`, a callback runs **tweet-level** scoring on the validation set (full decision rule, including hypotheses + margin computation, **at threshold 0.0**) and exposes `tweet_precision`, `tweet_recall`, `tweet_f1` to the trainer.

The best checkpoint is selected by:
- Two-stage / DE-only: `metric_for_best_model="tweet_f1"`
- Joint: `metric_for_best_model="tweet_f1_de"` (or `tweet_f1_en` via `--joint_eval_lang_for_best en`)

Early stopping: `EarlyStoppingCallback(patience=5)` — counts evaluations without improvement on the chosen metric.

After training, in two-stage mode an additional explicit checkpoint sweep ([`_select_best_checkpoint_by_tweet_objective`](src/train.py)) reranks the last K candidates by `model_selection_objective` (`f1`, `recall`, or `recall_at_precision`) on the validation set; this is largely redundant with `load_best_model_at_end` but acts as a safety net.

---

## 4. Decision Rule (How a Label Is Computed)

Given a fine-tuned model, predicting the label for one tweet works as follows ([`src/decision.py`](src/decision.py)).

### 4.1 Per-hypothesis score

For each hypothesis `h` of class `c ∈ {0,1}`:
1. Tokenize `(premise=text, hypothesis=h)`, run through model.
2. Softmax over the 3 NLI classes → `(p_contradiction, p_neutral, p_entailment)`.
3. Hypothesis score:
   - `score_mode = "entailment_only"` (default): `s = p_entailment`
   - `score_mode = "entailment_minus_contradiction"`: `s = p_entailment - contradiction_weight * p_contradiction`

### 4.2 Per-class aggregation

Both classes use **max** over their hypotheses (uniform aggregation prevents class 0 from dominating via averaging):

```
score_1 = max_h s(h)  for h in class-1 hypotheses
score_0 = max_h s(h)  for h in class-0 hypotheses
score_0 = score_0 * class_0_weight     # dampening (default 0.7)
```

In `class1_only` mode `score_0` is fixed at 0.

### 4.3 Margin and threshold

```
margin = score_1 - score_0
prediction = 1 if margin >= positive_margin_threshold else 0
```

The threshold is **not** zero by default — it is calibrated per stage on the validation set (see Section 5).

In joint mode, two threshold files are written (`decision_config_en.json` and `decision_config_de.json`) so inference can use the language-appropriate threshold; the default `decision_config.json` mirrors the language chosen via `--joint_eval_lang_for_best`.

---

## 5. Threshold Calibration

After training, [`tune_positive_margin_threshold`](src/decision.py) runs on the validation tweets:

1. Compute the margin for every validation tweet using the trained model.
2. Sweep `threshold_steps=401` candidate thresholds in `[threshold_search_min, threshold_search_max]` (defaults `-0.2` to `0.8`).
3. For each candidate, compute `(precision, recall, f1)` and rank by `--threshold_objective`:
   - `f1` → `(f1, recall, precision)`
   - `recall` → `(recall, f1, precision)`
   - `recall_at_precision` → `(constraint, recall, f1, precision)` where `constraint = (precision >= threshold_min_precision)`
4. Save the winning threshold + metrics to `decision_config.json` in the checkpoint dir.

If no threshold satisfies the precision constraint, the calibrator returns the recall-optimal fallback and marks `constraint_satisfied = false` — check the log if you see odd behavior.

Defaults (deliberate, to avoid threshold collapse to extreme negative values):
- `threshold_objective = "f1"` (no constraint pressure to over-trigger)
- `threshold_min_precision = 0.5`
- `threshold_search_min = -0.2`, `threshold_search_max = 0.8`

---

## 6. Class Imbalance Handling

Defaults assume EN ~17 % positives and DE ~4 % positives.

| Param | EN default | DE default | Effect |
|---|---|---|---|
| `*_pos_oversample_factor` | 1.5 | 4.0 | Replicates positive rows in train data |
| `*_positive_sample_weight` | 1.5 | 2.5 | Loss multiplier for positive-tweet rows |
| `class_0_weight` | 0.7 | 0.7 | Decision-time dampening for `score_0` |
| `class_0_loss_weight` | 0.7 | 0.7 | CE loss dampening for class-0 hyp pairs (matches decision side) |
| `de_lang_loss_weight` (joint) | — | 2.0 | Loss multiplier for DE rows in joint mode |

Stacking too aggressive oversampling AND positive weights collapses precision; the current defaults are deliberately moderate.

---

## 7. CLI Reference

### Required
- `--de_train PATH` — always required
- `--en_train PATH` — required unless `--de_only` is set

### Mode selection
- `--de_only` / `--de_only_output_dir DIR`
- `--joint_train` / `--joint_output_dir DIR` / `--joint_eval_lang_for_best {en|de}`

### Splits
- `--val_size 0.15`, `--test_size 0.15`
- `--en_val PATH`, `--de_val PATH` (only used if `--val_size` is omitted)

### Model & training
- `--model_name xlm-roberta-base`
- `--epochs 4`, `--lr 2e-5`, `--batch_size 16`

### Imbalance
- `--en_pos_oversample_factor`, `--de_pos_oversample_factor`
- `--en_positive_sample_weight`, `--de_positive_sample_weight`
- `--class_0_weight 0.7`, `--class_0_loss_weight 0.7`
- `--de_lang_loss_weight 2.0` (joint mode)
- `--tweet_level_loss_weight 0.2`, `--tweet_level_margin 0.0`

### NLI training mode
- `--nli_train_mode {both_classes_contradiction | both_classes_asymmetric_neutral | class1_only}`

### Threshold calibration
- `--threshold_objective {f1|recall|recall_at_precision}`
- `--threshold_min_precision`, `--threshold_search_min`, `--threshold_search_max`

### Checkpoint selection (post-train sweep, two-stage / de_only)
- `--model_selection_objective`, `--model_selection_min_precision`
- `--model_selection_last_k_checkpoints`

### Decision scoring
- `--score_mode {entailment_only|entailment_minus_contradiction}`
- `--contradiction_weight`

### Misc
- `--skip_evaluation`, `--show_examples 10`, `--log_file PATH`

---

## 8. Recommended Commands

### Joint EN+DE (recommended)

```powershell
python -m src.train `
  --joint_train `
  --en_train data/en_cleaned.csv `
  --de_train data/de_cleaned.csv `
  --nli_train_mode both_classes_asymmetric_neutral `
  --epochs 4
```

### Two-stage EN → DE (legacy)

```powershell
python -m src.train `
  --en_train data/en_cleaned.csv `
  --de_train data/de_cleaned.csv `
  --nli_train_mode both_classes_asymmetric_neutral `
  --epochs 4
```

### German-only

```powershell
python -m src.train `
  --de_only `
  --de_train data/de_cleaned.csv `
  --nli_train_mode both_classes_asymmetric_neutral `
  --epochs 4
```

### Manual evaluation of an existing checkpoint

```powershell
python -m src.evaluate `
  --checkpoint checkpoints/xlmr-nli/joint `
  --test_data data/de_cleaned_test_holdout.csv `
  --lang de `
  --show_examples 10
```

---

## 9. Outputs

Per training run the output dir contains:

| File | Purpose |
|---|---|
| `model.safetensors`, `config.json`, tokenizer files | Final selected checkpoint |
| `checkpoint-*/` | Per-step intermediate checkpoints |
| `decision_config.json` | Threshold + score config used by inference |
| `decision_config_en.json`, `decision_config_de.json` (joint only) | Per-language threshold |
| `tweet_level_checkpoint_metrics.jsonl` | One record per (checkpoint, lang) with tweet-level p/r/f1 |
| `train.log` | Full stdout/stderr of the run |

Inference loads `decision_config.json` (or a language-specific variant if available) and applies the decision rule from Section 4.
