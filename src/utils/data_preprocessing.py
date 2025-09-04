##################################################
# Utils para preprocesamiento y dataset multilabel.
##################################################

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class MedicalTextDataset(Dataset):
    """
    Dataset para clasificación multilabel.
    - texts: lista de strings ya construidos (título/abstract).
    - labels: lista de listas con 0/1 por categoría.
    - tokenizer: tokenizer de HF.
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.float)
        }


def get_data_paths(base_dir, cleaning_type, data_files):
    """
    Devuelve dict con rutas absolutas a train/val/test en la carpeta de limpieza.
    """
    folder = os.path.join(base_dir, cleaning_type)
    return {split: os.path.join(folder, fname) for split, fname in data_files.items()}


def filter_geographicals_data(df, verbose=True):
    """
    Elimina filas con category_Geographicals == 1 y, si existe, quita la columna.
    """
    initial = len(df)
    if "category_Geographicals" not in df.columns:
        if verbose:
            print("Column 'category_Geographicals' no encontrada. No se filtra.")
        return df

    out = df[df["category_Geographicals"] == 0].copy()
    if "category_Geographicals" in out.columns:
        out = out.drop("category_Geographicals", axis=1)

    if verbose:
        removed = initial - len(out)
        pct = (removed / max(initial, 1)) * 100
        print(f"Filtrado Geographicals: removidas {removed} filas ({pct:.2f}%).")
    return out


def get_filtered_category_info(df, excluded=None):
    """
    Retorna (lista_columnas_categoria, lista_nombres_legibles, num_labels).
    """
    excluded = set(excluded or [])
    cols = [c for c in df.columns if c.startswith("category_") and c not in excluded]
    if not cols:
        raise ValueError("No se encontraron columnas category_* tras el filtrado.")
    names = [c.replace("category_", "").replace("_", " ") for c in cols]
    return cols, names, len(cols)


def build_texts(df, mode, title_col, abstract_col, title_fallback, abstract_fallback):
    """
    Construye el texto de entrada a partir de columnas procesadas con fallback.
    mode: 'title_abstract' | 'abstract_only' | 'title_only'
    """
    t_col = title_col if title_col in df.columns else title_fallback
    a_col = abstract_col if abstract_col in df.columns else abstract_fallback

    def make_row(row):
        title = str(row.get(t_col, ""))
        abstract = str(row.get(a_col, ""))
        if mode == "abstract_only":
            return abstract.strip()
        if mode == "title_only":
            return title.strip()
        return f"{title} {abstract}".strip()

    return df.apply(make_row, axis=1).tolist()


def verify_cleaning_data(base_dir, cleaning_types, data_files):
    """
    Verifica que existan los CSVs para cada tipo de limpieza.
    Retorna dict: cleaning_type -> estado y rutas.
    """
    results = {}
    for ct in cleaning_types:
        paths = get_data_paths(base_dir, ct, data_files)
        missing = [p for p in paths.values() if not os.path.exists(p)]
        status = "complete" if len(missing) == 0 else "incomplete"
        results[ct] = {"status": status, "data_paths": paths, "missing": missing}
    return results


def compute_metrics(eval_pred):
    """
    Métricas multilabel para Trainer de HF.
    """
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }
