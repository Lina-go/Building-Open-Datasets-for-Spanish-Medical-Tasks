# Modelo y entrenamiento por experimento, con W&B opcional (simple).

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

from src.utils.data_preprocessing import (
    MedicalTextDataset,
    filter_geographicals_data,
    get_filtered_category_info,
    build_texts,
    compute_metrics,
)
from src.utils import visualizations as viz


class UniversalMedicalClassifier:
    """
    Un experimento = (modelo HF) × (modo de texto) × (tipo de limpieza).
    Opción de loguear en W&B si 'wandb_cfg.enabled' es true.
    """
    def __init__(self, model_name, text_mode, cleaning_type, data_paths,
                 title_col, abstract_col, title_fallback, abstract_fallback,
                 out_dir, train_params, experiment_id=None, wandb_cfg=None):
        self.model_name = model_name
        self.text_mode = text_mode
        self.cleaning_type = cleaning_type
        self.data_paths = data_paths
        self.title_col = title_col
        self.abstract_col = abstract_col
        self.title_fallback = title_fallback
        self.abstract_fallback = abstract_fallback
        self.out_dir = out_dir
        self.params = train_params

        self.experiment_id = experiment_id or os.path.basename(out_dir)
        self.wandb_cfg = wandb_cfg or {"enabled": False}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

        self.category_columns = []
        self.label_names = []
        self.num_labels = 0

    def _load_csvs(self):
        import pandas as pd

        train_df = pd.read_csv(self.data_paths["train"])
        val_df = pd.read_csv(self.data_paths["val"])
        test_df = pd.read_csv(self.data_paths["test"])

        print(f"Tamaños originales — train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        train_df = filter_geographicals_data(train_df, verbose=True)
        val_df = filter_geographicals_data(val_df, verbose=False)
        test_df = filter_geographicals_data(test_df, verbose=False)

        print(f"Tras filtrar — train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        self.category_columns, self.label_names, self.num_labels = get_filtered_category_info(
            train_df, excluded=["category_Geographicals"]
        )
        print(f"Categorías detectadas: {self.num_labels}")

        train_texts = build_texts(train_df, self.text_mode, self.title_col, self.abstract_col,
                                  self.title_fallback, self.abstract_fallback)
        val_texts = build_texts(val_df, self.text_mode, self.title_col, self.abstract_col,
                                self.title_fallback, self.abstract_fallback)
        test_texts = build_texts(test_df, self.text_mode, self.title_col, self.abstract_col,
                                 self.title_fallback, self.abstract_fallback)

        train_labels = train_df[self.category_columns].values.tolist()
        val_labels = val_df[self.category_columns].values.tolist()
        test_labels = test_df[self.category_columns].values.tolist()

        lengths = [len(t.split()) for t in train_texts]
        print("Longitud textos — media: %.1f palabras, máx: %d" % (np.mean(lengths), np.max(lengths)))

        return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

    def _create_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_safetensors=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification",
            use_safetensors=True
        )
        self.model.to(self.device)

    def _maybe_init_wandb(self):
        if not self.wandb_cfg.get("enabled", False):
            return False
        import wandb
        wandb.init(
            project=self.wandb_cfg.get("project", "medical-multilabel"),
            entity=self.wandb_cfg.get("entity"),
            group=self.wandb_cfg.get("group"),
            tags=self.wandb_cfg.get("tags") or [],
            name=self.experiment_id,
            reinit=True
        )
        cfg = {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "text_mode": self.text_mode,
            "cleaning_type": self.cleaning_type,
            "train_params": self.params
        }
        wandb.config.update(cfg)
        return True

    def train_and_evaluate(self):
        os.makedirs(self.out_dir, exist_ok=True)

        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = self._load_csvs()
        self._create_model_and_tokenizer()

        wandb_on = self._maybe_init_wandb()
        report_to_value = ["wandb"] if wandb_on else "none"

        train_ds = MedicalTextDataset(train_texts, train_labels, self.tokenizer, self.params["max_length"])
        val_ds = MedicalTextDataset(val_texts, val_labels, self.tokenizer, self.params["max_length"])
        test_ds = MedicalTextDataset(test_texts, test_labels, self.tokenizer, self.params["max_length"])

        args = TrainingArguments(
            output_dir=self.out_dir,
            num_train_epochs=self.params["num_epochs"],
            per_device_train_batch_size=self.params["batch_size"],
            per_device_eval_batch_size=self.params["batch_size"],
            learning_rate=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
            warmup_steps=self.params["warmup_steps"],
            eval_strategy="steps",
            eval_steps=self.params["eval_steps"],
            save_strategy="steps",
            save_steps=self.params["save_steps"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_weighted",
            greater_is_better=True,
            save_total_limit=2,
            dataloader_num_workers=0,
            fp16=(torch.cuda.is_available() and self.params.get("use_fp16_if_possible", True)),
            gradient_checkpointing=self.params.get("gradient_checkpointing", True),
            report_to=report_to_value,
            save_safetensors=True
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.params["early_stopping_patience"])]
        )

        print(f"Entrenando en {self.out_dir} ...")
        trainer.train()
        print("Evaluando en test ...")
        test_metrics = trainer.evaluate(test_ds)

        trainer.save_model()
        self.tokenizer.save_pretrained(self.out_dir)

        preds = trainer.predict(test_ds)
        logits = preds.predictions
        labels = preds.label_ids
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        pred_bin = (probs > 0.5).astype(int)

        plots_dir = os.path.join(self.out_dir, "plots")
        base = os.path.basename(self.out_dir)
        viz.plot_metrics_heatmap(labels, pred_bin, self.label_names, plots_dir, f"Per-label metrics — {base}")
        viz.plot_label_distribution(labels, pred_bin, self.label_names, plots_dir, f"Label distribution — {base}")
        viz.plot_confidence_distribution(probs, plots_dir, f"Confidence — {base}")
        viz.dump_classification_report(labels, pred_bin, self.label_names, self.out_dir)

        if wandb_on:
            import wandb
            wandb.log(test_metrics)
            wandb.finish()

        return {
            "test_results": test_metrics,
            "num_labels": self.num_labels,
            "label_names": self.label_names
        }