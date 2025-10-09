"""
GPT NER runner
Evalúa GPT-4 en tareas de Named Entity Recognition para AnatEM

Uso:
export AZURE_OPENAI_ENDPOINT="https://<endpoint>.openai.azure.com"
export AZURE_OPENAI_API_KEY="xxxxxxxx"

python -m src.scripts.gpt_ner_runner \
    --config configs/gpt_ner.yaml \
    --anatem-root data/AnatEM \
    --format nersuite-spanish
"""

import os
import argparse
import json
import time
from datetime import datetime

import pandas as pd
import yaml
import wandb
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from src.gpt.gpt_ner_classifier import GPTNERClassifier
from src.utils.ner_data import read_flat_dir, read_split_dir

# Add this after the imports in gpt_ner_runner.py, around line 25

import re

def normalize_text(text):
    """Normalize text for fuzzy matching"""
    # Remove accents, lowercase, remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fuzzy_find_entity(entity_text, tokens):
    """
    Find entity in tokens with fuzzy matching
    Returns list of (start_idx, end_idx) for all matches
    """
    entity_tokens = entity_text.lower().split()
    entity_normalized = normalize_text(entity_text)
    
    matches = []
    
    # Try exact match first
    for i in range(len(tokens) - len(entity_tokens) + 1):
        if [t.lower() for t in tokens[i:i+len(entity_tokens)]] == entity_tokens:
            matches.append((i, i + len(entity_tokens)))
    
    if matches:
        return matches
    
    # Try fuzzy match - check if entity is substring or superstring
    for i in range(len(tokens)):
        for j in range(i+1, min(i+10, len(tokens)+1)):
            span_text = " ".join(tokens[i:j])
            span_normalized = normalize_text(span_text)
            
            # Check if entity is in span or span is in entity
            if (entity_normalized in span_normalized or 
                span_normalized in entity_normalized):
                # Must have at least 50% overlap
                overlap = len(set(entity_normalized.split()) & set(span_normalized.split()))
                min_len = min(len(entity_normalized.split()), len(span_normalized.split()))
                if overlap >= min_len * 0.5:
                    matches.append((i, j))
    
    return matches


def map_entity_type(pred_type, true_entity_types):
    """
    Map GPT's generic types to ground truth specific types
    """
    pred_lower = pred_type.lower()
    
    # Direct match
    if pred_type in true_entity_types:
        return pred_type
    
    # Fuzzy matching
    type_mapping = {
        'cell': 'Cell',
        'cells': 'Cell',
        'célula': 'Cell',
        'células': 'Cell',
        'tissue': 'Tissue',
        'tejido': 'Tissue',
        'organ': 'Organ',
        'órgano': 'Organ',
        'structure': 'Cellular_component',
        'component': 'Cellular_component',
        'nucleus': 'Cellular_component',
        'núcleo': 'Cellular_component',
    }
    
    # Try mapping
    for key, value in type_mapping.items():
        if key in pred_lower and value in true_entity_types:
            return value
    
    # Default: return most common type in true_entity_types
    if true_entity_types:
        # Return Cell as default if available
        if 'Cell' in true_entity_types:
            return 'Cell'
        return sorted(true_entity_types)[0]
    
    return pred_type


def entities_to_bio_tags_improved(sentence, entities, tokens, true_entity_types):
    """Convert entities back to BIO format with fuzzy matching"""
    tags = ['O'] * len(tokens)
    
    for entity_text, entity_type in entities:
        # Map the type
        mapped_type = map_entity_type(entity_type, true_entity_types)
        
        # Find all possible matches
        matches = fuzzy_find_entity(entity_text, tokens)
        
        # Use first match
        if matches:
            start_idx, end_idx = matches[0]
            tags[start_idx] = f'B-{mapped_type}'
            for i in range(start_idx + 1, end_idx):
                if i < len(tokens):
                    tags[i] = f'I-{mapped_type}'
    
    return tags

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_test_data(anatem_root, format_type):
    """Load test data from AnatEM"""
    has_test = os.path.isdir(os.path.join(anatem_root, format_type, "test"))
    
    if has_test:
        test_tokens, test_tags = read_split_dir(anatem_root, format_type, "test")
    else:
        all_tokens, all_tags = read_flat_dir(anatem_root, format_type)
        n_test = int(len(all_tokens) * 0.2)
        test_tokens = all_tokens[-n_test:]
        test_tags = all_tags[-n_test:]
    
    return test_tokens, test_tags

def convert_to_sentences(tokens_list, tags_list):
    """Convert token/tag lists to sentence strings with true entities"""
    sentences = []
    true_entities_list = []
    
    for tokens, tags in zip(tokens_list, tags_list):
        sentence = " ".join(tokens)
        sentences.append(sentence)
        
        # Extract true entities
        entities = []
        current_entity = []
        current_type = None
        
        for token, tag in zip(tokens, tags):
            if tag.startswith('B-'):
                if current_entity:
                    entities.append((" ".join(current_entity), current_type))
                current_entity = [token]
                current_type = tag[2:]
            elif tag.startswith('I-') and current_entity:
                current_entity.append(token)
            else:
                if current_entity:
                    entities.append((" ".join(current_entity), current_type))
                current_entity = []
                current_type = None
        
        if current_entity:
            entities.append((" ".join(current_entity), current_type))
        
        true_entities_list.append(entities)
    
    return sentences, true_entities_list

def entities_to_bio_tags(sentence, entities, tokens):
    """Convert entities back to BIO format for evaluation"""
    tags = ['O'] * len(tokens)
    
    for entity_text, entity_type in entities:
        entity_tokens = entity_text.split()
        
        # Find entity in tokens
        for start_idx in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[start_idx:start_idx + len(entity_tokens)] == entity_tokens:
                tags[start_idx] = f'B-{entity_type}'
                for i in range(1, len(entity_tokens)):
                    if start_idx + i < len(tokens):
                        tags[start_idx + i] = f'I-{entity_type}'
                break
    
    return tags

def evaluate_ner_predictions(true_entities_list, pred_entities_list, tokens_list):
    """Convert predictions to BIO format and compute metrics"""
    
    # Get all true entity types for mapping
    all_true_types = set()
    for true_entities in true_entities_list:
        for _, etype in true_entities:
            all_true_types.add(etype)
    
    true_tags_list = []
    pred_tags_list = []
    
    for true_entities, pred_entities, tokens in zip(true_entities_list, pred_entities_list, tokens_list):
        sentence = " ".join(tokens)
        
        # Use old function for true tags (they're already correct)
        true_tags = entities_to_bio_tags(sentence, true_entities, tokens)
        
        # Use improved function for predicted tags
        pred_tags = entities_to_bio_tags_improved(sentence, pred_entities, tokens, all_true_types)
        
        true_tags_list.append(true_tags)
        pred_tags_list.append(pred_tags)
    
    if not true_tags_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    return {
        "precision": precision_score(true_tags_list, pred_tags_list),
        "recall": recall_score(true_tags_list, pred_tags_list),
        "f1": f1_score(true_tags_list, pred_tags_list)
    }

def run_gpt_ner_evaluation(config_path, anatem_root, format_type, strategies, 
                          sample_size, force_batch, batch_threshold,
                          wandb_project, wandb_entity, wandb_mode, deployment_override):
    """Main GPT NER evaluation function"""
    
    # Load config if provided
    cfg = {}
    if config_path and os.path.exists(config_path):
        cfg = load_yaml(config_path)
        strategies = cfg.get("strategies", strategies)
        sample_size = cfg.get("sample_size", sample_size)
        force_batch = cfg.get("force_batch", force_batch)
        batch_threshold = cfg.get("batch_threshold", batch_threshold)
        
        wb = cfg.get("wandb", {}) or {}
        wandb_project = wb.get("project", wandb_project)
        wandb_entity = wb.get("entity", wandb_entity)
        wandb_mode = wb.get("mode", wandb_mode)
        
        az = cfg.get("azure", {}) or {}
        deployment_override = az.get("deployment", deployment_override)

    # Setup wandb mode
    if wandb_mode in ("offline", "disabled"):
        os.environ["WANDB_MODE"] = wandb_mode

    # Load test data
    print(f"Loading test data from: {anatem_root}")
    test_tokens, test_tags = load_test_data(anatem_root, format_type)
    
    # Sample if requested
    if sample_size and len(test_tokens) > sample_size:
        test_tokens = test_tokens[:sample_size]
        test_tags = test_tags[:sample_size]
    
    print(f"Test samples: {len(test_tokens)}")
    
    # Convert to sentences and extract true entities
    sentences, true_entities_list = convert_to_sentences(test_tokens, test_tags)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join("results", f"gpt_ner_evaluation_{timestamp}")
    os.makedirs(out_root, exist_ok=True)
    
    all_results = []
    
    for strategy in strategies:
        exp_id = f"GPT_NER_{strategy}_{timestamp}"
        print(f"\n=== {exp_id} ===")
        out_dir = os.path.join(out_root, exp_id)
        
        # Initialize wandb run
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=exp_id,
            group="gpt_ner_evaluation",
            config={
                "strategy": strategy,
                "test_samples": len(sentences),
                "format": format_type,
                "batch_threshold": batch_threshold,
                "force_batch": force_batch
            }
        )
        
        # Initialize GPT NER classifier
        deployment = deployment_override or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        classifier = GPTNERClassifier(
            deployment_name=deployment,
            strategy=strategy,
            temperature=0.0
        )
        
        # Run evaluation
        start_time = time.time()
        try:
            if force_batch and len(sentences) >= batch_threshold:
                pred_entities_list = classifier.extract_entities_batch(sentences, work_dir=out_dir)
            else:
                pred_entities_list = []
                for sentence in sentences:
                    entities = classifier.extract_entities(sentence)
                    pred_entities_list.append(entities)
                    time.sleep(0.1)  # Rate limiting
            
            # Evaluate predictions
            metrics = evaluate_ner_predictions(true_entities_list, pred_entities_list, test_tokens)
            elapsed = time.time() - start_time
            
            # Log to wandb
            wandb.log({
                "f1": metrics.get("f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "samples": len(sentences)
            })
            
            # Store results
            result = {
                "experiment_id": exp_id,
                "strategy": strategy,
                "samples": len(sentences),
                "metrics": metrics,
                "elapsed_time": elapsed
            }
            all_results.append(result)
            
            print(f"F1: {metrics.get('f1', 0):.4f}, Time: {elapsed:.1f}s")
            
            # Save detailed results
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                
                # Save predictions
                predictions_data = []
                for i, (sentence, true_entities, pred_entities) in enumerate(
                    zip(sentences, true_entities_list, pred_entities_list)
                ):
                    predictions_data.append({
                        "sentence": sentence,
                        "true_entities": true_entities,
                        "predicted_entities": pred_entities
                    })
                
                with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
                    json.dump(predictions_data, f, indent=2, ensure_ascii=False)
                
                # Save metrics
                with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"Error in strategy {strategy}: {e}")
            
        wandb.finish()
    
    # Save summary
    if all_results:
        summary_dir = os.path.join(out_root, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # JSON results
        with open(os.path.join(summary_dir, "gpt_ner_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # CSV summary
        rows = []
        for r in all_results:
            m = r.get("metrics", {})
            rows.append({
                "Experiment_ID": r.get("experiment_id"),
                "Strategy": r.get("strategy"),
                "Samples": r.get("samples"),
                "F1": m.get("f1"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "Time_seconds": r.get("elapsed_time")
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(summary_dir, "gpt_ner_summary.csv"), index=False)
        
        print(f"\nResults saved to: {out_root}")
        print("\nSummary:")
        print(df.to_string(index=False))
    else:
        print("No successful evaluations")

def main():
    parser = argparse.ArgumentParser(description="GPT NER evaluation runner")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--anatem-root", default="data/AnatEM", help="AnatEM root directory")
    parser.add_argument("--format", default="nersuite-spanish", choices=["conll", "nersuite", "nersuite-spanish"])
    parser.add_argument("--strategies", nargs="+", default=["zero_shot", "few_shot"])
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--force-batch", action="store_true")
    parser.add_argument("--batch-threshold", type=int, default=500)
    parser.add_argument("--wandb-project", default="gpt-ner-evaluation")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--deployment", default=None, help="Azure OpenAI deployment override")
    
    args = parser.parse_args()
    
    run_gpt_ner_evaluation(
        config_path=args.config,
        anatem_root=args.anatem_root,
        format_type=args.format,
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