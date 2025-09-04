# Visualizaciones simples separadas de la lógica de entrenamiento.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_metrics_heatmap(labels, preds, label_names, out_dir, title):
    """
    Heatmap con Precision/Recall/F1 por etiqueta.
    """
    ensure_dir(out_dir)
    rows = []
    for i, name in enumerate(label_names):
        rows.append({
            "Label": name[:30],
            "Precision": precision_score(labels[:, i], preds[:, i], zero_division=0),
            "Recall": recall_score(labels[:, i], preds[:, i], zero_division=0),
            "F1": f1_score(labels[:, i], preds[:, i], zero_division=0),
        })
    df = pd.DataFrame(rows).set_index("Label")

    plt.figure(figsize=(12, max(6, 0.35 * len(df))))
    sns.heatmap(df[["Precision", "Recall", "F1"]], annot=True, fmt=".3f", cmap="RdYlBu_r")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()

    df.reset_index().to_csv(os.path.join(out_dir, "per_label_metrics.csv"), index=False)


def plot_label_distribution(labels, preds, label_names, out_dir, title):
    """
    Barras comparando frecuencia verdadera vs predicha de las etiquetas más comunes.
    """
    ensure_dir(out_dir)
    true_counts = labels.sum(axis=0)
    pred_counts = preds.sum(axis=0)

    n = min(15, len(label_names))
    idx = np.argsort(true_counts)[-n:]

    x = np.arange(n)
    w = 0.4
    plt.figure(figsize=(14, 7))
    plt.bar(x - w/2, true_counts[idx], width=w, label="True")
    plt.bar(x + w/2, pred_counts[idx], width=w, label="Pred")
    plt.xticks(x, [label_names[i][:20] for i in idx], rotation=40, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "label_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_confidence_distribution(probs, out_dir, title):
    """
    Histograma simple de las probabilidades de salida.
    """
    ensure_dir(out_dir)
    plt.figure(figsize=(10, 5))
    plt.hist(probs.flatten(), bins=50, edgecolor="black", alpha=0.75)
    plt.axvline(0.5, color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def dump_classification_report(labels, preds, label_names, out_dir):
    """
    Guarda classification_report como JSON.
    """
    ensure_dir(out_dir)
    report = classification_report(labels, preds, target_names=label_names, output_dict=True, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)