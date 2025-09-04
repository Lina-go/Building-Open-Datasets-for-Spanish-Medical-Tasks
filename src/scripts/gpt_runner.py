"""
GPT runner
Uso:
export AZURE_OPENAI_ENDPOINT="https://<endpoint>.openai.azure.com"
export AZURE_OPENAI_API_KEY="xxxxxxxx"

python -m src.scripts.gpt_runner \
    --config configs/experiment_config.yaml \
    --gpt-config configs/gpt_small_sample.yaml
"""

import os
import argparse
import json
from datetime import datetime

import pandas as pd
import yaml
import wandb

from src.gpt.gpt_classifier import GPTMedicalClassifier, GPT_LABELS
from src.utils.data_preprocessing import (
    get_data_paths,
    filter_geographicals_data,
    verify_cleaning_data
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def subset_to_gpt_space(df):
    keep = [c for c in df.columns if c.startswith("category_")]
    valid = []
    for c in keep:
        nice = c.replace("category_", "").replace("_", " ")
        if nice in GPT_LABELS:
            valid.append(c)
    non_cat = [c for c in df.columns if not c.startswith("category_")]
    return df[non_cat + valid]


def run_small_sample(cfg_path, gpt_cfg_path,
                     strategies, sample_size, force_batch, batch_threshold,
                     wandb_project, wandb_entity, wandb_mode, deployment_override):
    # YAML principal (rutas de datos / cleaning_types)
    cfg = load_yaml(cfg_path)
    base_dir = cfg["data"]["base_dir"]
    files = cfg["data"]["files"]

    # YAML opcional específico de GPT (si se pasa, sobreescribe defaults/CLI)
    if gpt_cfg_path:
        g = load_yaml(gpt_cfg_path)
        strategies = g.get("strategies", strategies)
        sample_size = g.get("sample_size", sample_size)
        force_batch = g.get("force_batch", force_batch)
        batch_threshold = g.get("batch_threshold", batch_threshold)
        wb = g.get("wandb", {}) or {}
        wandb_project = wb.get("project", wandb_project)
        wandb_entity = wb.get("entity", wandb_entity)
        wandb_mode = wb.get("mode", wandb_mode)
        az = g.get("azure", {}) or {}
        deployment_override = az.get("deployment", deployment_override)

    if batch_threshold is None:
        batch_threshold = 500

    if wandb_mode in ("offline", "disabled"):
        os.environ["WANDB_MODE"] = wandb_mode

    variants = verify_cleaning_data(base_dir, cfg["experiments"]["cleaning_types"], files)
    cleaning_types = [ct for ct, res in variants.items() if res["status"] == "complete"]
    if not cleaning_types:
        print("No hay cleaning_types válidos.")
        return

    out_root = os.path.join("results", f"gpt_small_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_root, exist_ok=True)

    print("GPT small-sample")
    print("Cleaning types:", cleaning_types)
    print("Strategies:", strategies)
    print("Sample size:", sample_size if sample_size else "FULL")
    print("Batch:", force_batch, "Threshold:", batch_threshold)
    print("W&B:", wandb_project, wandb_entity, wandb_mode)
    print("Deployment:", deployment_override or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt"))

    all_results = []

    for ct in cleaning_types:
        paths = get_data_paths(base_dir, ct, files)
        test_df = pd.read_csv(paths["test"])
        test_df = filter_geographicals_data(test_df, verbose=True)
        test_df = subset_to_gpt_space(test_df)

        if sample_size and len(test_df) > sample_size:
            df_eval = test_df.head(sample_size).copy()
        else:
            df_eval = test_df.copy()

        for strat in strategies:
            exp_id = f"GPT_small_{strat}_{ct}"
            print("\n==", exp_id, "==")
            out_dir = os.path.join(out_root, exp_id)

            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=exp_id,
                group="gpt_small_sample",
                config={
                    "strategy": strat,
                    "cleaning_type": ct,
                    "sample_size": len(df_eval),
                    "batch_threshold": batch_threshold,
                    "force_batch": force_batch
                }
            )

            deployment = deployment_override or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt")
            clf = GPTMedicalClassifier(
                deployment_name=deployment,
                strategy=strat,
                temperature=0.0
            )

            res = clf.evaluate_on_dataframe(
                df_eval,
                output_dir=out_dir,
                use_batch=force_batch,
                batch_threshold=batch_threshold
            )

            res["cleaning_type"] = ct
            res["sample_size"] = len(df_eval)
            res["total_dataset_size"] = len(test_df)
            all_results.append(res)

            m = res["metrics"]
            wandb.log({
                "f1_weighted": m.get("f1_weighted"),
                "f1_micro": m.get("f1_micro"),
                "f1_macro": m.get("f1_macro"),
                "accuracy": m.get("accuracy"),
                "samples": res.get("samples", 0),
            })
            wandb.finish()

    summary_dir = os.path.join(out_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "gpt_small_sample_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    rows = []
    for r in all_results:
        m = r.get("metrics", {})
        rows.append({
            "Experiment_ID": r.get("experiment_id"),
            "Model": r.get("model_name"),
            "Strategy": r.get("strategy"),
            "Cleaning_Type": r.get("cleaning_type"),
            "Sample_Size": r.get("sample_size"),
            "Total_Dataset_Size": r.get("total_dataset_size"),
            "F1_Weighted": m.get("f1_weighted"),
            "F1_Micro": m.get("f1_micro"),
            "F1_Macro": m.get("f1_macro"),
            "Accuracy": m.get("accuracy"),
        })
    pd.DataFrame(rows).to_csv(os.path.join(summary_dir, "gpt_small_sample_results.csv"), index=False)

    print("\nResultados guardados en:", out_root)


def main():
    p = argparse.ArgumentParser(description="GPT small-sample (simple, con W&B y YAML opcional).")
    p.add_argument("--config", required=True, help="Ruta a configs/experiment_config.yaml")
    p.add_argument("--gpt-config", default=None, help="YAML opcional con parámetros de GPT/W&B")
    p.add_argument("--strategies", nargs="+", default=["zero_shot", "few_shot"])
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--force-batch", action="store_true")
    p.add_argument("--batch-threshold", type=int, default=None)
    p.add_argument("--wandb-project", default="medical-multilabel")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--deployment", default=None, help="Override del deployment Azure (si no usas el YAML)")
    args = p.parse_args()

    run_small_sample(
        cfg_path=args.config,
        gpt_cfg_path=args.gpt_config,
        strategies=args.strategies,
        sample_size=args.sample_size,
        force_batch=args.force_batch,
        batch_threshold=args.batch_threshold,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        deployment_override=args.deployment
    )


if __name__ == "__main__":
    main()
