import os
import numpy as np
from dataclasses import dataclass
from typing import Dict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
)
import evaluate
from sklearn.utils.class_weight import compute_class_weight
from datasets import DatasetDict
from .dataset import load_nli_dataset
from .hypotheses import NLI_LABEL2ID

@dataclass
class Config:
    model_name: str = "xlm-roberta-base"
    output_dir: str = "checkpoints/xlmr-nli"
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 4  # Reduziert auf 4 für MacBook MPS
    gradient_accumulation_steps: int = 2  # Simuliert effektiv batch size 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42

def tokenize_fn(examples, tokenizer):
    encoded = tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
    # Behalte die nli_label-Spalte für das Training
    encoded["labels"] = examples["nli_label"]
    return encoded

def compute_metrics(eval_pred):
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Konvertiere zu Python-Listen für robuste Verarbeitung
    preds_list = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
    labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
    
    acc_res = metric_acc.compute(predictions=preds_list, references=labels_list)
    f1_res = metric_f1.compute(predictions=preds_list, references=labels_list, average="macro")
    
    # Cast alle Werte zu Python-float um Serialisierungsfehler zu vermeiden
    out = {k: float(v) for k, v in acc_res.items()}
    out.update({k: float(v) for k, v in f1_res.items()})
    return out

def class_weights_from_dataset(ds, num_labels: int) -> np.ndarray:
    y = np.array(ds["nli_label"])
    # Klassen 0,1,2 → weights
    weights = compute_class_weight(class_weight="balanced",
                                   classes=np.arange(num_labels),
                                   y=y)
    return weights.astype(np.float32)

def train_stage(train_csv: str, val_csv: str, lang: str, cfg: Config, resume_from: str | None):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = load_nli_dataset(train_csv, lang)
    val_ds   = load_nli_dataset(val_csv, lang) if val_csv else None

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds}) if val_ds else DatasetDict({"train": train_ds})
    tokenized = ds_dict.map(lambda ex: tokenize_fn(ex, tokenizer), batched=True, remove_columns=ds_dict["train"].column_names)
    num_labels = len(NLI_LABEL2ID)

    model_path = resume_from if resume_from else cfg.model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Klassengewichte für Imbalance
    if val_ds:
        weights = class_weights_from_dataset(ds_dict["train"], num_labels)
    else:
        weights = np.ones(num_labels, dtype=np.float32)

    # Hook für gewichte im loss
    def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        if labels is None:
            raise ValueError("'labels' not found in batch inputs. Available keys: " + str(list(inputs.keys())))
        outputs = model(**inputs)
        logits = outputs.logits
        import torch
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=logits.device))
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
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
        eval_steps=500,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="f1",
        seed=cfg.seed,
        report_to=[],
        #use_cpu=True,  # Nutze CPU statt MPS/GPU für MacBook Speicher
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
    # Monkey‑patch compute_loss für Gewichte
    trainer.compute_loss = compute_loss

    callbacks = []
    if val_ds:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    trainer.add_callback(callbacks[0]) if callbacks else None

    trainer.train()
    best_dir = args.output_dir
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    return best_dir

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_train", required=True)
    parser.add_argument("--en_val", required=False, default="")
    parser.add_argument("--de_train", required=True)
    parser.add_argument("--de_val", required=False, default="")
    parser.add_argument("--model_name", default="xlm-roberta-base")
    args_ = parser.parse_args()

    cfg = Config(model_name=args_.model_name)

    # Stage 1: EN
    en_out = train_stage(args_.en_train, args_.en_val, "en", cfg, resume_from=None)

    # Stage 2: DE (weiterfeintunen vom EN‑Checkpoint)
    cfg.output_dir = os.path.join(cfg.output_dir, "de_ft")
    _ = train_stage(args_.de_train, args_.de_val, "de", cfg, resume_from=en_out)