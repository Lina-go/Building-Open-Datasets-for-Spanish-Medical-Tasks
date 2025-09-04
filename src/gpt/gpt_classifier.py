##############################
# gpt_classifier.py
##############################

import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from openai import AzureOpenAI

from src.gpt.prompting import GPT_LABELS, build_prompt
from src.gpt.azure_batch_tools import (
    generate_jsonl, create_file, create_batch_job,
    poll_batch_until_done, download_bytes, parse_batch_output
)


class GPTMedicalClassifier:
    """
    Cliente mínimo para Azure OpenAI:
    - chat.completions online
    - Global-Batch para lotes grandes
    """

    def __init__(self, deployment_name, strategy="zero_shot", temperature=0.0,
                 azure_endpoint=None, api_key=None, api_version=None):
        self.deployment_name = deployment_name
        self.strategy = strategy
        self.temperature = float(temperature)

        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=version
        )

    # ---------------------- ONLINE ----------------------

    def classify_text(self, text):
        prompt = build_prompt(text, self.strategy)
        resp = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a precise medical multilabel classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=350,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        preds = [x for x in data if x in GPT_LABELS]
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) if usage else 0
        return preds, {"raw_response": content, "tokens_used": tokens}

    # ---------------------- BATCH -----------------------

    def classify_texts_batch(self, texts, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        input_jsonl = os.path.join(work_dir, "input.jsonl")

        generate_jsonl(
            texts=texts,
            strategy=self.strategy,
            system_prompt="You are a precise medical multilabel classifier.",
            deployment_name=self.deployment_name,
            output_path=input_jsonl,
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=350
        )

        input_file_id = create_file(self.client, input_jsonl)
        batch_id = create_batch_job(self.client, input_file_id)

        status, output_file_id, error_file_id = poll_batch_until_done(self.client, batch_id)
        if status != "completed" or not output_file_id:
            return [], []

        raw_bytes = download_bytes(self.client, output_file_id)
        by_id = parse_batch_output(raw_bytes)

        preds_list = []
        metas_list = []
        for i in range(len(texts)):
            item = by_id.get(f"task-{i}", {})
            content = item.get("content") or "[]"
            usage = item.get("usage") or {}
            data = json.loads(content)
            preds = [x for x in data if x in GPT_LABELS]
            tokens = int(usage.get("total_tokens", 0))
            preds_list.append(preds)
            metas_list.append({"raw_response": content, "tokens_used": tokens})

        return preds_list, metas_list

    # --------------------- EVALUACIÓN --------------------

    def _true_labels_from_df(self, df):
        cols = [c for c in df.columns if c.startswith("category_")]
        mapping = {c: c.replace("category_", "").replace("_", " ") for c in cols}
        out = []
        for _, row in df.iterrows():
            labels = []
            for c, nice in mapping.items():
                if nice in GPT_LABELS and int(row[c]) == 1:
                    labels.append(nice)
            out.append(labels)
        return out

    def _metrics_from_lists(self, true_lists, pred_lists):
        y_true = np.zeros((len(true_lists), len(GPT_LABELS)))
        y_pred = np.zeros((len(pred_lists), len(GPT_LABELS)))
        for i, (t, p) in enumerate(zip(true_lists, pred_lists)):
            for j, lab in enumerate(GPT_LABELS):
                if lab in t:
                    y_true[i, j] = 1
                if lab in p:
                    y_pred[i, j] = 1
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        }

    def evaluate_on_dataframe(self, df, output_dir=None, use_batch=False, batch_threshold=200):
        start = time.time()

        texts = []
        for _, row in df.iterrows():
            title = str(row.get("title_processed", row.get("title", ""))).strip()
            abstract = str(row.get("spanish_abstract_processed", row.get("spanish_abstract", ""))).strip()
            texts.append(f"{title} {abstract}".strip())

        true_lists = self._true_labels_from_df(df)

        if use_batch and len(texts) >= batch_threshold:
            preds_list, metas_list = self.classify_texts_batch(texts, work_dir=output_dir or ".")
        else:
            preds_list = []
            metas_list = []
            for t in texts:
                preds, meta = self.classify_text(t)
                preds_list.append(preds)
                metas_list.append(meta)
                time.sleep(0.03)

        metrics = self._metrics_from_lists(true_lists, preds_list)
        elapsed = time.time() - start

        results = {
            "experiment_id": f"GPT_{self.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_name": self.deployment_name,
            "strategy": self.strategy,
            "samples": len(df),
            "metrics": metrics,
            "texts_preview": texts[:50],
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "gpt_results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            pd.DataFrame({
                "text": texts,
                "true_categories": ["; ".join(x) for x in true_lists],
                "predicted_categories": ["; ".join(x) for x in preds_list],
            }).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

        print("Done in %.1fs | F1-weighted=%.4f" % (elapsed, metrics["f1_weighted"]))
        return results
