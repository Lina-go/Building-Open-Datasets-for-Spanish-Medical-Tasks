"""
Improved GPT-NER evaluation runner with k-NN retrieval and self-verification
Expected improvements based on GPT-NER paper:
- k-NN few-shot: +10-15% F1 over random few-shot
- Self-verification: +2-5% F1

Usage:
export AZURE_OPENAI_ENDPOINT="https://<endpoint>.openai.azure.com"
export AZURE_OPENAI_API_KEY="xxxxxxxx"

python -m src.scripts.gpt_ner_runner_improved \
    --config configs/gpt_ner_improved.yaml \
    --anatem-root data/AnatEM \
    --format nersuite-spanish \
    --knn-index models/knn_index
"""

import os
import argparse
import json
import time
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import yaml
import wandb
from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from src.gpt.gpt_ner_improved import ImprovedGPTNER
from src.gpt.knn_retrieval import KNNRetriever
from src.utils.ner_data import read_split_dir, read_flat_dir


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_data(anatem_root, format_type):
    """Load test data"""
    has_test = os.path.isdir(os.path.join(anatem_root, format_type, "test"))
    
    if has_test:
        test_tokens, test_tags = read_split_dir(anatem_root, format_type, "test")
    else:
        all_tokens, all_tags = read_flat_dir(anatem_root, format_type)
        n_test = int(len(all_tokens) * 0.2)
        test_tokens = all_tokens[-n_test:]
        test_tags = all_tags[-n_test:]
    
    return test_tokens, test_tags


def extract_entity_types_from_tags(tags_list):
    """Extract unique entity types from BIO tags"""
    entity_types = set()
    for tags in tags_list:
        for tag in tags:
            if tag.startswith('B-') or tag.startswith('I-'):
                entity_types.add(tag[2:])
    return sorted(entity_types)


def entities_to_bio_tags(tokens: List[str], 
                        entities: List[Tuple[str, str]], 
                        true_entity_types: List[str]) -> List[str]:
    """
    Convert predicted entities back to BIO tags
    Uses improved matching from previous diagnostic work
    """
    import re
    
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def find_entity_span(entity_text, tokens):
        """Find entity span in tokens"""
        entity_tokens = entity_text.lower().split()
        
        # Try exact match first
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if [t.lower() for t in tokens[i:i+len(entity_tokens)]] == entity_tokens:
                return (i, i + len(entity_tokens))
        
        # Try normalized match
        entity_norm = normalize_text(entity_text)
        for i in range(len(tokens)):
            for j in range(i+1, min(i+len(entity_tokens)+3, len(tokens)+1)):
                span_norm = normalize_text(" ".join(tokens[i:j]))
                if entity_norm == span_norm:
                    return (i, j)
        
        return None
    
    def map_type(pred_type, true_types):
        """Map predicted type to true type"""
        pred_lower = pred_type.lower()
        
        # Direct match
        for true_type in true_types:
            if pred_lower == true_type.lower():
                return true_type
        
        # Partial match
        for true_type in true_types:
            if pred_lower in true_type.lower() or true_type.lower() in pred_lower:
                return true_type
        
        # Default
        if true_types:
            return true_types[0]
        return pred_type
    
    # Initialize tags
    tags = ['O'] * len(tokens)
    assigned = [False] * len(tokens)
    
    for entity_text, entity_type in entities:
        span = find_entity_span(entity_text, tokens)
        if not span:
            continue
        
        start, end = span
        
        # Check conflicts
        if any(assigned[start:end]):
            continue
        
        # Map type
        mapped_type = map_type(entity_type, true_entity_types)
        
        # Assign tags
        tags[start] = f'B-{mapped_type}'
        for i in range(start + 1, end):
            if i < len(tokens):
                tags[i] = f'I-{mapped_type}'
        
        # Mark as assigned
        for i in range(start, end):
            if i < len(tokens):
                assigned[i] = True
    
    return tags


def evaluate_predictions(true_tags_list: List[List[str]], 
                        pred_tags_list: List[List[str]]) -> dict:
    """Compute evaluation metrics"""
    
    if not true_tags_list or not pred_tags_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    return {
        "precision": precision_score(true_tags_list, pred_tags_list),
        "recall": recall_score(true_tags_list, pred_tags_list),
        "f1": f1_score(true_tags_list, pred_tags_list)
    }


def run_improved_evaluation(config_path, anatem_root, format_type, 
                           knn_index_path, strategies, sample_size,
                           wandb_project, wandb_entity, wandb_mode, 
                           deployment_override):
    """Main evaluation function"""
    
    # Load config
    cfg = {}
    if config_path and os.path.exists(config_path):
        cfg = load_yaml(config_path)
        strategies = cfg.get("strategies", strategies)
        sample_size = cfg.get("sample_size", sample_size)
        
        wb = cfg.get("wandb", {}) or {}
        wandb_project = wb.get("project", wandb_project)
        wandb_entity = wb.get("entity", wandb_entity)
        wandb_mode = wb.get("mode", wandb_mode)
        
        az = cfg.get("azure", {}) or {}
        deployment_override = az.get("deployment", deployment_override)
    
    # Setup wandb
    if wandb_mode in ("offline", "disabled"):
        os.environ["WANDB_MODE"] = wandb_mode
    
    # Load test data
    print(f"Loading test data from: {anatem_root}")
    test_tokens, test_tags = load_test_data(anatem_root, format_type)
    
    if sample_size and len(test_tokens) > sample_size:
        test_tokens = test_tokens[:sample_size]
        test_tags = test_tags[:sample_size]
    
    print(f"Test samples: {len(test_tokens)}")
    
    # Get entity types
    entity_types = extract_entity_types_from_tags(test_tags)
    print(f"Entity types: {entity_types}")
    
    # Load k-NN index if provided
    retriever = None
    if knn_index_path and os.path.exists(knn_index_path):
        print(f"\nLoading k-NN index from: {knn_index_path}")
        retriever = KNNRetriever(index_path=knn_index_path)
        print(f"✓ Index loaded with {retriever.index.ntotal} examples")
    else:
        print("\nNo k-NN index found - using zero-shot only")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join("results", f"gpt_ner_improved_{timestamp}")
    os.makedirs(out_root, exist_ok=True)
    
    all_results = []
    
    for strategy in strategies:
        # Skip few-shot if no retriever
        if strategy == "knn_few_shot" and not retriever:
            print(f"\nSkipping {strategy} - no k-NN index available")
            continue
        
        exp_id = f"GPT_NER_improved_{strategy}_{timestamp}"
        print(f"\n{'='*80}")
        print(f"{exp_id}")
        print(f"{'='*80}")
        
        out_dir = os.path.join(out_root, exp_id)
        os.makedirs(out_dir, exist_ok=True)
        
        # Setup experiment config
        use_verification = strategy.endswith("_verified")
        use_knn = strategy.startswith("knn")
        
        # Initialize wandb
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=exp_id,
            group="gpt_ner_improved",
            config={
                "strategy": strategy,
                "use_knn": use_knn,
                "use_verification": use_verification,
                "test_samples": len(test_tokens),
                "format": format_type
            }
        )
        
        # Initialize classifier
        deployment = deployment_override or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        
        classifier = ImprovedGPTNER(
            deployment_name=deployment,
            retriever=retriever if use_knn else None,
            use_verification=use_verification,
            temperature=0.0,
            entity_types=entity_types
        )
        
        # Run evaluation
        start_time = time.time()
        
        print(f"\nExtracting entities...")
        pred_entities_list = []
        
        for i, tokens in enumerate(tqdm(test_tokens)):
            sentence = " ".join(tokens)
            entities = classifier.extract_entities(sentence, k=5)
            pred_entities_list.append(entities)
            
            # Rate limiting
            if (i + 1) % 10 == 0:
                time.sleep(0.5)
        
        # Convert to BIO tags
        print(f"\nConverting to BIO tags...")
        pred_tags_list = []
        for tokens, pred_entities in zip(test_tokens, pred_entities_list):
            pred_tags = entities_to_bio_tags(tokens, pred_entities, entity_types)
            pred_tags_list.append(pred_tags)
        
        # Evaluate
        metrics = evaluate_predictions(test_tags, pred_tags_list)
        elapsed = time.time() - start_time
        
        # Log to wandb
        wandb.log({
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "samples": len(test_tokens)
        })
        
        # Store results
        result = {
            "experiment_id": exp_id,
            "strategy": strategy,
            "use_knn": use_knn,
            "use_verification": use_verification,
            "samples": len(test_tokens),
            "metrics": metrics,
            "elapsed_time": elapsed
        }
        all_results.append(result)
        
        print(f"\nResults:")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Time:      {elapsed:.1f}s")
        
        # Save detailed results
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        wandb.finish()
    
    # Save summary
    if all_results:
        summary_dir = os.path.join(out_root, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # JSON
        with open(os.path.join(summary_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # CSV
        rows = []
        for r in all_results:
            m = r["metrics"]
            rows.append({
                "Experiment_ID": r["experiment_id"],
                "Strategy": r["strategy"],
                "Use_KNN": r["use_knn"],
                "Use_Verification": r["use_verification"],
                "Samples": r["samples"],
                "F1": m["f1"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "Time_seconds": r["elapsed_time"]
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(summary_dir, "summary.csv"), index=False)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {out_root}")
        print(f"{'='*80}")
        print("\nSummary:")
        print(df.to_string(index=False))
        
        # Print improvements
        if len(df) > 1:
            print(f"\n{'='*80}")
            print("IMPROVEMENTS:")
            print(f"{'='*80}")
            baseline_f1 = df[df["Strategy"].str.contains("zero_shot")]["F1"].values
            if len(baseline_f1) > 0:
                baseline = baseline_f1[0]
                for _, row in df.iterrows():
                    if row["F1"] > baseline:
                        improvement = ((row["F1"] - baseline) / baseline) * 100
                        print(f"{row['Strategy']}: +{improvement:.1f}% over baseline")


def main():
    parser = argparse.ArgumentParser(description="Improved GPT-NER evaluation")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--anatem-root", default="data/AnatEM")
    parser.add_argument("--format", default="nersuite-spanish")
    parser.add_argument("--knn-index", default="models/knn_index", 
                       help="Path to k-NN index directory")
    parser.add_argument("--strategies", nargs="+", 
                       default=["zero_shot", "knn_few_shot", "knn_few_shot_verified"])
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--wandb-project", default="gpt-ner-improved")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--deployment", default=None)
    
    args = parser.parse_args()
    
    run_improved_evaluation(
        config_path=args.config,
        anatem_root=args.anatem_root,
        format_type=args.format,
        knn_index_path=args.knn_index,
        strategies=args.strategies,
        sample_size=args.sample_size,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        deployment_override=args.deployment
    )


if __name__ == "__main__":
    main()