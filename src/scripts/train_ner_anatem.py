"""
Entrenamiento NER en AnatEM.

Uso:
python -m src.scripts.train_ner_anatem --config configs/ner_anatem.yaml
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
from seqeval.metrics.v1 import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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

    # Bloques anidados
    wandb_block = cfg.get("wandb", {}) or {}
    models_block = cfg.get("models", {}) or {}

    flat_cfg = {k: v for k, v in cfg.items() if k not in ("wandb", "models")}

    parser = argparse.ArgumentParser(
        description="Entrenamiento NER en AnatEM con múltiples modelos.",
        parents=[base]
    )
    # Datos
    parser.add_argument("--anatem-root")
    parser.add_argument("--format", choices=["conll", "nersuite", "nersuite-spanish"])
    parser.add_argument("--auto_split", action="store_true")

    # Modelo único (si no usas 'models:' en YAML)
    parser.add_argument("--model-name")

    # Tokenizer / entrenamiento
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")

    # Logging / W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode")

    # Defaults YAML
    parser.set_defaults(**flat_cfg)
    parser.set_defaults(
        wandb=bool(wandb_block.get("enabled", False)),
        wandb_project=wandb_block.get("project"),
        wandb_entity=wandb_block.get("entity"),
        wandb_mode=wandb_block.get("mode"),
        models=models_block  # dict {alias: hf_id}
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

    # Config W&B desde YAML/CLI → exporta env si está habilitado
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

    # 3) Dataset HF (sin tokenizar aún; tokenizaremos por modelo)
    ds = DatasetDict({
        "train": Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_tags}),
        "validation": Dataset.from_dict({"tokens": dev_tokens, "ner_tags": dev_tags}),
        "test": Dataset.from_dict({"tokens": test_tokens, "ner_tags": test_tags}),
    })

    # 4) Directorio raíz de resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = os.path.join("results", f"ner_experiments_{timestamp}")
    os.makedirs(root_out, exist_ok=True)

    # 5) Modelos a correr
    models_dict = getattr(args, "models", {}) or {}
    runs = []

    if args.model_name:
        models_to_run = [("CustomModel", args.model_name)]
    elif isinstance(models_dict, dict) and len(models_dict) > 0:
        models_to_run = list(models_dict.items())
    else:
        # fallback a uno solo si no hay bloque models ni --model-name
        default_name = args.model_name or "dccuchile/bert-base-spanish-wwm-cased"
        models_to_run = [("Default", default_name)]

    # 6) Loop por modelo
    for alias, hf_id in models_to_run:
        print(f"\n=== Modelo: {alias} -> {hf_id} ===")

        # Tokenizer + alineado por modelo
        tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)

        def _map_fn(batch):
            return tokenize_and_align_labels(batch, tokenizer, label2id, args.max_length)

        tokenized = ds.map(_map_fn, batched=True, remove_columns=["tokens", "ner_tags"])

        # Modelo (safetensors para evitar el gate del CVE)
        model = AutoModelForTokenClassification.from_pretrained(
            hf_id,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            use_safetensors=True
        )

        out_dir = os.path.join(root_out, alias)
        os.makedirs(out_dir, exist_ok=True)

        train_args = TrainingArguments(
            output_dir=out_dir,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=(args.batch_size * 2 if args.batch_size else 16),
            eval_strategy="epoch",    # usa eval_strategy por compatibilidad
            save_strategy="epoch",
            logging_steps=50,
            save_total_limit=2,
            lr_scheduler_type="linear",
            warmup_ratio=args.warmup_ratio,  # si tu versión no lo soporta, cambia por warmup_steps=0
            weight_decay=0.01,
            seed=args.seed or 42,
            report_to=("wandb" if args.wandb else "none"),
            run_name=(args.run_name or f"AnatEM_{alias}_{timestamp}"),
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

        # Guardado de artefactos por modelo
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

        report_txt = classification_report(true_tags, pred_tags)
        with open(os.path.join(out_dir, "seqeval_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_txt)

        row = {
            "alias": alias,
            "model_name": hf_id,
            "test_precision": test_metrics.get("test_precision"),
            "test_recall": test_metrics.get("test_recall"),
            "test_f1": test_metrics.get("test_f1")
        }
        runs.append(row)

    # 7) Resumen multi-modelo
    if runs:
        df = pd.DataFrame(runs)
        df.sort_values("test_f1", ascending=False, inplace=True)
        df.to_csv(os.path.join(root_out, "summary.csv"), index=False)
        print("\nResumen guardado en:", os.path.join(root_out, "summary.csv"))
    print("\nListo. Carpeta de resultados:", root_out)

if __name__ == "__main__":
    main()
