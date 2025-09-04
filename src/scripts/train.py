"""
Script para lanzar experimentos definidos en desde el archivo YAML.
Uso:
  python -m src.scripts.train --config configs/experiment_config.yaml
"""

import argparse
import os
from datetime import datetime
import yaml

from src.utils.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Lanza experimentos desde un YAML.")
    parser.add_argument("--config", required=True, help="Ruta a experiment_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    text_cfg = cfg["text"]
    models_map = cfg["models"]
    exp_cfg = cfg["experiments"]
    train_cfg = cfg["train"]
    out_cfg = cfg["output"]
    wandb_cfg = cfg.get("wandb", {"enabled": False})

    if str(wandb_cfg.get("mode", "online")).lower() in ("offline", "disabled"):
        os.environ["WANDB_MODE"] = str(wandb_cfg["mode"]).lower()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(out_cfg["base_dir"] + f"_{timestamp}")

    runner = ExperimentRunner(output_root)
    runner.run_all(
        models_map=models_map,
        text_configs=exp_cfg["text_configs"],
        cleaning_types=exp_cfg["cleaning_types"],
        data_base_dir=data_cfg["base_dir"],
        data_files=data_cfg["files"],
        text_cols=text_cfg,
        train_params=train_cfg,
        wandb_cfg=wandb_cfg
    )


if __name__ == "__main__":
    main()
