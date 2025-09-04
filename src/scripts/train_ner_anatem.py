"""
Entrenamiento NER en AnatEM (ES/EN) con Hugging Face.

Uso con YAML:
python -m src.scripts.train_ner_anatem --config configs/ner_anatem.yaml
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

from src.utils.ner_data import (
    read_split_dir, read_flat_dir, auto_split,
    build_label_list, tokenize_and_align_labels
)

# ------------------------- YAML opcional + CLI -------------------------

def parse_args_with_yaml():
    import argparse
    import yaml

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", default=None, help="Ruta a YAML opcional")
    known, _ = base.parse_known_args()

    cfg = {}
    if known.config:
        with open(known.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Extrae bloque W&B anidado del YAML (si existe)
    wandb_block = cfg.get("wandb", {}) or {}
    # Evita pasar un dict donde se espera un bool
    flat_cfg = {k: v for k, v in cfg.items() if k != "wandb"}

    parser = argparse.ArgumentParser(
        description="Entrenamiento NER en AnatEM.",
        parents=[base]
    )
    # Datos
    parser.add_argument("--anatem-root", help="Ruta que contiene conll/, nersuite/ o nersuite-spanish/")
    parser.add_argument("--format", choices=["conll", "nersuite", "nersuite-spanish"])
    parser.add_argument("--auto_split", action="store_true", help="Divide 80/10/10 si no hay train/devel/test")

    # Modelo / tokenizer
    parser.add_argument("--model-name", help="Modelo HF")
    parser.add_argument("--max-length", type=int)

    # Entrenamiento
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")

    # Logging / W&B (CLI puede sobrescribir YAML)
    parser.add_argument("--wandb", action="store_true", help="Activa W&B si se pasa por CLI")
    parser.add_argument("--wandb-project", help="W&B project")
    parser.add_argument("--wandb-entity", help="W&B entity (usuario o equipo)")
    parser.add_argument("--wandb-mode", help="online | offline | disabled")

    # Defaults desde YAML plano
    parser.set_defaults(**flat_cfg)
    # Defaults específicos de W&B desde el bloque anidado
    parser.set_defaults(
        wandb=bool(wandb_block.get("enabled", False)),
        wandb_project=wandb_block.get("project"),
        wandb_entity=wandb_block.get("entity"),
        wandb_mode=wandb_block.get("mode"),
    )

    args = parser.parse_args()
    return args

# ------------------------------ Métricas ------------------------------

def build_compute_metrics(id2label):
    label_list = [id2label[i] for i in range(len(id2label))]

    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)

        true_tags, pred_tags = [], []
        for p_seq, l_seq in zip(preds, labels):
            p_tags, l_tags = [], []
            for p_i, l_i in zip(p_seq, l_seq):
                if l_i == -100:
                    continue
                p_tags.append(label_list[p_i])
                l_tags.append(label_list[l_i])
            pred_tags.append(p_tags)
            true_tags.append(l_tags)

        return {
            "precision": precision_score(true_tags, pred_tags),
            "recall": recall_score(true_tags, pred_tags),
            "f1": f1_score(true_tags, pred_tags)
        }

    return compute_metrics

# -------------------------------- Main --------------------------------

def main():
    args = parse_args_with_yaml()

    # W&B desde YAML/CLI → exporta env si está habilitado
    if args.wandb:
        if args.wandb_project:
            os.environ.setdefault("WANDB_PROJECT", str(args.wandb_project))
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", str(args.wandb_entity))
        if args.wandb_mode:
            os.environ.setdefault("WANDB_MODE", str(args.wandb_mode))

    # 1) Leer datos (con o sin splits)
    has_train = os.path.isdir(os.path.join(args.anatem_root, args.format, "train"))
    has_dev = os.path.isdir(os.path.join(args.anatem_root, args.format, "devel"))
    has_test = os.path.isdir(os.path.join(args.anatem_root, args.format, "test"))

    if has_train and has_dev and has_test:
        train_tokens, train_tags = read_split_dir(args.anatem_root, args.format, "train")
        dev_tokens, dev_tags = read_split_dir(args.anatem_root, args.format, "devel")
        test_tokens, test_tags = read_split_dir(args.anatem_root, args.format, "test")
    else:
        all_tokens, all_tags = read_flat_dir(args.anatem_root, args.format)
        if args.auto_split:
            (train_tokens, train_tags), (dev_tokens, dev_tags), (test_tokens, test_tags) = auto_split(
                all_tokens, all_tags, seed=args.seed or 42, ratios=(0.8, 0.1, 0.1)
            )
        else:
            n = len(all_tokens)
            n_dev = max(1, int(n * 0.1))
            n_test = max(1, int(n * 0.1))
            train_tokens = all_tokens[: max(1, n - n_dev - n_test)]
            train_tags = all_tags[: len(train_tokens)]
            dev_tokens = all_tokens[len(train_tokens): len(train_tokens)+n_dev]
            dev_tags = all_tags[len(train_tokens): len(train_tokens)+n_dev]
            test_tokens = all_tokens[len(train_tokens)+n_dev:]
            test_tags = all_tags[len(train_tokens)+n_dev:]

    print("Oraciones: train={}, dev={}, test={}".format(len(train_tokens), len(dev_tokens), len(test_tokens)))

    # 2) Etiquetas
    labels = build_label_list(train_tags, dev_tags, test_tags)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    print("Etiquetas:", labels)

    # 3) Dataset HF
    ds = DatasetDict({
        "train": Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_tags}),
        "validation": Dataset.from_dict({"tokens": dev_tokens, "ner_tags": dev_tags}),
        "test": Dataset.from_dict({"tokens": test_tokens, "ner_tags": test_tags}),
    })

    # 4) Tokenizer + alineado
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def _map_fn(batch):
        return tokenize_and_align_labels(batch, tokenizer, label2id, args.max_length)

    tokenized = ds.map(_map_fn, batched=True, remove_columns=["tokens", "ner_tags"])

    # 5) Modelo (safetensors para evitar el gate del CVE en torch.load)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True
    )

    # 6) Entrenamiento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"ner_experiments_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=(args.batch_size * 2 if args.batch_size else 16),
        eval_strategy="epoch",         # usa eval_strategy por compatibilidad
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,  # si tu versión no lo soporta, cambia por warmup_steps=0
        weight_decay=0.01,
        seed=args.seed or 42,
        report_to=("wandb" if args.wandb else "none"),
        run_name=(args.run_name or f"AnatEM_{args.format}_{timestamp}"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = build_compute_metrics(id2label)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Entrenando...")
    trainer.train()

    print("Evaluando en test...")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(test_metrics)

    # 7) Guardado de artefactos
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)

    # Reporte seqeval detallado
    preds_logits, _, _ = trainer.predict(tokenized["test"])
    preds = np.argmax(preds_logits, axis=-1)

    true_tags = []
    pred_tags = []
    for p_seq, l_seq in zip(preds, tokenized["test"]["labels"]):
        p_tags, l_tags = [], []
        for p_i, l_i in zip(p_seq, l_seq):
            if l_i == -100:
                continue
            p_tags.append(id2label[p_i])
            l_tags.append(id2label[l_i])
        pred_tags.append(p_tags)
        true_tags.append(l_tags)

    report_txt = classification_report(true_tags, pred_tags, zero_division=0)
    with open(os.path.join(out_dir, "seqeval_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    pd.DataFrame([{
        "precision": test_metrics.get("test_precision", None),
        "recall": test_metrics.get("test_recall", None),
        "f1": test_metrics.get("test_f1", None)
    }]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    print("Listo. Artefactos guardados en:", out_dir)

if __name__ == "__main__":
    main()
