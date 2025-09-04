"""
Barridos de hiperparámetros con Weights & Biases, reusando utilidades del repo.

Uso:
  python -m src.scripts.tune --config configs/tuning_config.yaml --runs 20
"""

import os
import argparse
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import torch
import wandb

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)

from src.utils.data_preprocessing import (
    MedicalTextDataset,
    filter_geographicals_data,
    get_filtered_category_info,
    build_texts,
    compute_metrics,
    get_data_paths
)


# ------------------------- funciones helper  -------------------------

def setup_env_and_device():
    print(f"PyTorch version: {torch.__version__}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_SILENT", "true")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_data_once(cfg):
    """
    Carga y prepara (una vez) los datasets siguiendo las mismas reglas que train.py:
    - Filtra Geographicals
    - Usa columnas procesadas con fallback
    - Deriva columnas de categorías
    """
    paths = get_data_paths(cfg["data"]["base_dir"], cfg["data"]["cleaning_type"], cfg["data"]["files"])
    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {name} file: {p}")

    print("\nLoading CSVs:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")

    train_df = pd.read_csv(paths["train"])
    val_df   = pd.read_csv(paths["val"])
    test_df  = pd.read_csv(paths["test"])

    print(f"Original sizes — train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_df = filter_geographicals_data(train_df, verbose=True)
    val_df   = filter_geographicals_data(val_df, verbose=False)
    test_df  = filter_geographicals_data(test_df, verbose=False)

    print(f"After filter — train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    category_cols, label_names, num_labels = get_filtered_category_info(train_df)
    print(f"Detected {num_labels} category columns.")

    tcfg = cfg["text"]
    train_texts = build_texts(train_df, tcfg["mode"], tcfg["title_column"], tcfg["abstract_column"],
                              tcfg["title_fallback"], tcfg["abstract_fallback"])
    val_texts   = build_texts(val_df,   tcfg["mode"], tcfg["title_column"], tcfg["abstract_column"],
                              tcfg["title_fallback"], tcfg["abstract_fallback"])
    test_texts  = build_texts(test_df,  tcfg["mode"], tcfg["title_column"], tcfg["abstract_column"],
                              tcfg["title_fallback"], tcfg["abstract_fallback"])

    train_labels = train_df[category_cols].values.tolist()
    val_labels   = val_df[category_cols].values.tolist()
    test_labels  = test_df[category_cols].values.tolist()

    lengths = [len(t.split()) for t in train_texts]
    print("Text lengths — avg: %.1f words, max: %d" % (np.mean(lengths), np.max(lengths)))

    data_bundle = {
        "category_cols": category_cols,
        "label_names": label_names,
        "num_labels": num_labels,
        "train_texts": train_texts, "val_texts": val_texts, "test_texts": test_texts,
        "train_labels": train_labels, "val_labels": val_labels, "test_labels": test_labels
    }
    return data_bundle


def build_model(model_name, num_labels, config):
    """
    Crea el modelo con safetensors y aplica tasas de dropout del sweep.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        hidden_dropout_prob=float(config.get("dropout_rate", 0.1)),
        attention_probs_dropout_prob=float(config.get("attention_dropout", 0.1)),
        use_safetensors=True,
        trust_remote_code=False
    )
    return model


# ----------------------------- ejecución W&B -----------------------------

def train_one_run(cfg, data_bundle, output_root):
    """
    Función que ejecuta UNA corrida con la config que entrega W&B (wandb.config).
    """
    with wandb.init(project=cfg["wandb"]["project"],
                    entity=cfg["wandb"].get("entity"),
                    config={}, reinit=True) as run:
        # Config hiperparámetros (desde el sweep)
        c = wandb.config

        # Tokenizer una vez por ejecución
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_safetensors=True)

        # Datasets
        max_length = int(cfg["train"]["max_length"])
        train_ds = MedicalTextDataset(data_bundle["train_texts"], data_bundle["train_labels"], tokenizer, max_length)
        val_ds   = MedicalTextDataset(data_bundle["val_texts"],   data_bundle["val_labels"],   tokenizer, max_length)
        test_ds  = MedicalTextDataset(data_bundle["test_texts"],  data_bundle["test_labels"],  tokenizer, max_length)

        # Modelo
        model = build_model(cfg["model_name"], data_bundle["num_labels"], c)

        # Warmup
        steps_per_epoch = max(1, len(train_ds) // max(1, int(c.batch_size)))
        total_steps = steps_per_epoch * int(c.num_epochs)
        warmup_steps = int(total_steps * float(c.get("warmup_ratio", 0.1)))

        # Path de salida
        run_dir = os.path.join(output_root, f"run-{run.name}")
        os.makedirs(run_dir, exist_ok=True)

        # Trainer
        args = TrainingArguments(
            output_dir=run_dir,
            num_train_epochs=int(c.num_epochs),
            per_device_train_batch_size=int(c.batch_size),
            per_device_eval_batch_size=int(c.batch_size) * 2,
            gradient_accumulation_steps=int(c.get("gradient_accumulation_steps", 1)),
            learning_rate=float(c.learning_rate),
            weight_decay=float(c.get("weight_decay", 0.01)),
            warmup_steps=warmup_steps,
            lr_scheduler_type=str(c.get("lr_scheduler_type", "linear")),
            max_grad_norm=float(c.get("max_grad_norm", 1.0)),
            eval_strategy="steps", eval_steps=200,
            save_strategy="steps", save_steps=200,
            save_total_limit=2,
            save_safetensors=True,
            logging_steps=50,
            report_to=["wandb"],
            run_name=f"sweep_{run.name}",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_weighted", greater_is_better=True,
            dataloader_num_workers=0,
            fp16=(torch.cuda.is_available()),
            gradient_checkpointing=True,
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=int(cfg["train"]["early_stopping_patience"]),
                early_stopping_threshold=float(cfg["train"]["early_stopping_delta"])
            )]
        )

        trainer.train()
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        wandb.log({
            "final_test_f1_weighted": test_metrics.get("test_f1_weighted"),
            "final_test_f1_micro": test_metrics.get("test_f1_micro"),
            "final_test_accuracy": test_metrics.get("test_accuracy"),
        })

        # Limpieza para liberar memoria
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="W&B sweep runner (usa utilidades del repo).")
    parser.add_argument("--config", required=True, help="Ruta a configs/tuning_config.yaml")
    parser.add_argument("--runs", type=int, default=20, help="Cantidad de corridas del sweep")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Modo W&B
    mode = str(cfg["wandb"].get("mode", "online")).lower()
    if mode in ("offline", "disabled"):
        os.environ["WANDB_MODE"] = mode

    device = setup_env_and_device()

    # Carga de datos
    data_bundle = load_data_once(cfg)

    # Carpeta raíz para los outputs de este sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join("results", f"sweeps_{timestamp}")
    os.makedirs(output_root, exist_ok=True)

    # Definición del sweep
    sweep_spec = cfg["sweep"]
    sweep_id = wandb.sweep(sweep_spec, project=cfg["wandb"]["project"], entity=cfg["wandb"].get("entity"))
    print(f"Sweep created: {sweep_id}")

    # Agente que ejecuta N corridas
    wandb.agent(sweep_id, function=lambda: train_one_run(cfg, data_bundle, output_root), count=int(args.runs))


if __name__ == "__main__":
    main()
