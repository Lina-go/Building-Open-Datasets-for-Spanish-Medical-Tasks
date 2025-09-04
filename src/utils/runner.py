##################################################################
# Runner para experimentos con varios modelos
##################################################

import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt

from src.models.nlu_model import UniversalMedicalClassifier
from src.utils.data_preprocessing import get_data_paths, verify_cleaning_data


class ExperimentRunner:
    """
    Ejecuta la rejilla: modelos × text_configs × cleaning_types.
    Guarda resultados por experimento y un CSV resumen.
    """
    def __init__(self, output_root):
        self.output_root = output_root
        self.results = []

    def run_all(self, models_map, text_configs, cleaning_types,
                data_base_dir, data_files, text_cols, train_params,
                wandb_cfg=None):
        os.makedirs(self.output_root, exist_ok=True)
        start = time.time()

        verification = verify_cleaning_data(data_base_dir, cleaning_types, data_files)
        cleaning_types = [ct for ct, info in verification.items() if info["status"] == "complete"]
        if not cleaning_types:
            print("No hay tipos de limpieza válidos con datos completos.")
            return

        total = len(models_map) * len(text_configs) * len(cleaning_types)
        print("Ejecutando experimentos:")
        print(f"  modelos={len(models_map)} textos={len(text_configs)} limpiezas={len(cleaning_types)} total={total}")

        for ct in cleaning_types:
            paths = get_data_paths(data_base_dir, ct, data_files)
            for short_name, hf_name in models_map.items():
                for text_mode in text_configs:
                    exp_id = f"{short_name}_{text_mode}_{ct}"
                    out_dir = os.path.join(self.output_root, exp_id)
                    print("\n" + "="*70)
                    print(f"Experimento {exp_id}")
                    print("="*70)

                    clf = UniversalMedicalClassifier(
                        model_name=hf_name,
                        text_mode=text_mode,
                        cleaning_type=ct,
                        data_paths=paths,
                        title_col=text_cols["title_column"],
                        abstract_col=text_cols["abstract_column"],
                        title_fallback=text_cols["title_fallback"],
                        abstract_fallback=text_cols["abstract_fallback"],
                        out_dir=out_dir,
                        train_params=train_params,
                        experiment_id=exp_id,
                        wandb_cfg=(wandb_cfg or {"enabled": False})
                    )

                    out = clf.train_and_evaluate()
                    self.results.append({
                        "experiment_id": exp_id,
                        "model_name": hf_name,
                        "text_config": text_mode,
                        "cleaning_type": ct,
                        "test_results": out["test_results"],
                        "num_labels": out["num_labels"]
                    })

        self._summarize()
        elapsed = time.time() - start
        print(f"\nExperimentos finalizados en {elapsed/60:.1f} min. Resultados en: {self.output_root}")

    def _summarize(self):
        summary_dir = os.path.join(self.output_root, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        rows = []
        for r in self.results:
            m = r["test_results"]
            rows.append({
                "Experiment_ID": r["experiment_id"],
                "Model": r["model_name"],
                "Text_Config": r["text_config"],
                "Cleaning_Type": r["cleaning_type"],
                "F1_Weighted": m.get("eval_f1_weighted"),
                "F1_Micro": m.get("eval_f1_micro"),
                "F1_Macro": m.get("eval_f1_macro"),
                "Accuracy": m.get("eval_accuracy"),
                "Precision_Micro": m.get("eval_precision_micro"),
                "Recall_Micro": m.get("eval_recall_micro"),
                "Num_Labels": r["num_labels"]
            })

        df = pd.DataFrame(rows)
        csv_path = os.path.join(summary_dir, "experiment_results_multi_cleaning.csv")
        df.to_csv(csv_path, index=False)

        if len(df) > 0 and df["F1_Weighted"].notna().any():
            import numpy as np
            plt.figure(figsize=(12, 6))
            df_sorted = df.sort_values("F1_Weighted", ascending=False).head(20)
            plt.barh(df_sorted["Experiment_ID"], df_sorted["F1_Weighted"])
            plt.xlabel("F1-Weighted")
            plt.title("Top Experimentos")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, "top_experiments.png"), dpi=300, bbox_inches="tight")
            plt.close()

        with open(os.path.join(summary_dir, "all_results.json"), "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"Resumen guardado en: {summary_dir}")