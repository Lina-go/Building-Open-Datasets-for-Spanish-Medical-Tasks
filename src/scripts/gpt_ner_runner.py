"""
GPT NER runner - IMPROVED VERSION
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
import re
from datetime import datetime

import pandas as pd
import yaml
import wandb
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from src.gpt.gpt_ner_classifier import GPTNERClassifier
from src.utils.ner_data import read_flat_dir, read_split_dir


# ==================== IMPROVED MATCHING FUNCTIONS ====================

def normalize_text(text):
    """Normalize text for fuzzy matching"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def strict_find_entity(entity_text, tokens):
    """
    Find entity in tokens with STRICT matching first, then fuzzy
    Returns list of (start_idx, end_idx) for all matches
    
    Priority:
    1. Exact token match
    2. Case-insensitive exact match
    3. Fuzzy match with 70% threshold (last resort)
    """
    entity_tokens = entity_text.lower().split()
    matches = []
    
    # 1. Try EXACT match first (most common case)
    for i in range(len(tokens) - len(entity_tokens) + 1):
        if [t.lower() for t in tokens[i:i+len(entity_tokens)]] == entity_tokens:
            matches.append((i, i + len(entity_tokens)))
    
    if matches:
        return matches
    
    # 2. Try case-insensitive exact match
    entity_normalized = normalize_text(entity_text)
    
    for i in range(len(tokens)):
        for j in range(i+1, min(i+len(entity_tokens)+3, len(tokens)+1)):
            span_text = " ".join(tokens[i:j])
            span_normalized = normalize_text(span_text)
            
            # Check if normalized versions match exactly
            if entity_normalized == span_normalized:
                matches.append((i, j))
    
    if matches:
        return matches
    
    # 3. Only as last resort: fuzzy matching with HIGH threshold (70%)
    entity_words = set(entity_normalized.split())
    
    for i in range(len(tokens)):
        for j in range(i+1, min(i+len(entity_tokens)+3, len(tokens)+1)):
            span_text = " ".join(tokens[i:j])
            span_words = set(normalize_text(span_text).split())
            
            # Require at least 70% word overlap (increased from 50%)
            if entity_words and span_words:
                overlap = len(entity_words & span_words)
                min_len = min(len(entity_words), len(span_words))
                if overlap >= min_len * 0.7:  # 70% threshold
                    matches.append((i, j))
    
    return matches


def smart_map_entity_type(pred_type, true_entity_types):
    """
    Intelligently map GPT's predicted types to ground truth types
    
    Handles:
    - Direct matches
    - Spanish/English variations
    - Partial matches
    - Sensible defaults
    """
    pred_lower = pred_type.lower()
    
    # 1. Direct exact match (case insensitive)
    for true_type in true_entity_types:
        if pred_type.lower() == true_type.lower():
            return true_type
    
    # 2. Common mappings (Spanish and English)
    type_mapping = {
        # Cell types
        'cell': 'Cell',
        'cells': 'Cell',
        'célula': 'Cell',
        'células': 'Cell',
        'cellular': 'Cell',
        
        # Tissue types
        'tissue': 'Tissue',
        'tissues': 'Tissue',
        'tejido': 'Tissue',
        'tejidos': 'Tissue',
        
        # Organ types
        'organ': 'Organ',
        'organs': 'Organ',
        'órgano': 'Organ',
        'órganos': 'Organ',
        
        # Cellular components
        'component': 'Cellular_component',
        'cellular_component': 'Cellular_component',
        'cell_component': 'Cellular_component',
        'structure': 'Cellular_component',
        'organelle': 'Cellular_component',
        'nucleus': 'Cellular_component',
        'núcleo': 'Cellular_component',
        'membrane': 'Cellular_component',
        'membrana': 'Cellular_component',
        'mitochondria': 'Cellular_component',
        'mitocondria': 'Cellular_component',
        
        # Organism subdivisions
        'subdivision': 'Organism_subdivision',
        'region': 'Organism_subdivision',
        'sistema': 'Organism_subdivision',
        'system': 'Organism_subdivision',
    }
    
    # 3. Try mapping
    for key, value in type_mapping.items():
        if key in pred_lower and value in true_entity_types:
            return value
    
    # 4. Partial match on true types
    for true_type in true_entity_types:
        true_lower = true_type.lower()
        # Check if pred is substring of true or vice versa
        if pred_lower in true_lower or true_lower in pred_lower:
            return true_type
    
    # 5. Default: return most common type or first alphabetically
    if true_entity_types:
        # Prefer Cell as default if available
        if 'Cell' in true_entity_types:
            return 'Cell'
        if 'Tissue' in true_entity_types:
            return 'Tissue'
        if 'Organ' in true_entity_types:
            return 'Organ'
        return sorted(true_entity_types)[0]
    
    return pred_type


def entities_to_bio_tags_improved(sentence, entities, tokens, true_entity_types):
    """
    Convert predicted entities back to BIO format with improved matching
    
    Features:
    - Conflict resolution (no double-assignment)
    - Smart type mapping
    - Strict matching priority
    """
    tags = ['O'] * len(tokens)
    
    # Track which tokens have been assigned to avoid conflicts
    assigned = [False] * len(tokens)
    
    for entity_text, entity_type in entities:
        # Map the type to ground truth types
        mapped_type = smart_map_entity_type(entity_type, true_entity_types)
        
        # Find all possible matches
        matches = strict_find_entity(entity_text, tokens)
        
        if not matches:
            continue
        
        # Use first non-conflicting match
        for start_idx, end_idx in matches:
            # Check if any tokens in this span are already assigned
            if any(assigned[start_idx:end_idx]):
                continue
            
            # Assign tags
            tags[start_idx] = f'B-{mapped_type}'
            for i in range(start_idx + 1, end_idx):
                if i < len(tokens):
                    tags[i] = f'I-{mapped_type}'
            
            # Mark as assigned
            for i in range(start_idx, end_idx):
                if i < len(tokens):
                    assigned[i] = True
            
            break  # Use first valid match only
    
    return tags


# ==================== ORIGINAL HELPER FUNCTIONS ====================

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


def evaluate_ner_predictions(true_entities_list, pred_entities_list, tokens_list):
    """
    Convert predictions to BIO format and compute metrics
    WITH DIAGNOSTIC OUTPUT to help debug
    """
    
    # Get all true entity types for mapping
    all_true_types = set()
    for true_entities in true_entities_list:
        for _, etype in true_entities:
            all_true_types.add(etype)
    
    print(f"\nFound {len(all_true_types)} unique entity types: {sorted(all_true_types)}")
    
    true_tags_list = []
    pred_tags_list = []
    
    # Track statistics for diagnostics
    total_pred = 0
    total_true = 0
    matched = 0
    
    for true_entities, pred_entities, tokens in zip(true_entities_list, pred_entities_list, tokens_list):
        sentence = " ".join(tokens)
        
        # For true tags: use simple exact matching (they should always work)
        true_tags = ['O'] * len(tokens)
        for entity_text, entity_type in true_entities:
            entity_tokens = entity_text.split()
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if [t.lower() for t in tokens[i:i+len(entity_tokens)]] == [e.lower() for e in entity_tokens]:
                    true_tags[i] = f'B-{entity_type}'
                    for j in range(1, len(entity_tokens)):
                        if i+j < len(tokens):
                            true_tags[i+j] = f'I-{entity_type}'
                    break
        
        # For predicted tags: use improved matching
        pred_tags = entities_to_bio_tags_improved(sentence, pred_entities, tokens, all_true_types)
        
        true_tags_list.append(true_tags)
        pred_tags_list.append(pred_tags)
        
        # Count for diagnostics
        total_true += len(true_entities)
        total_pred += len(pred_entities)
        # Simple overlap check for diagnostics
        pred_entities_text = {e[0].lower() for e in pred_entities}
        true_entities_text = {e[0].lower() for e in true_entities}
        matched += len(pred_entities_text & true_entities_text)
    
    # Print diagnostics
    print(f"\nDiagnostics:")
    print(f"  Total true entities: {total_true}")
    print(f"  Total predicted entities: {total_pred}")
    print(f"  Exact text matches: {matched}")
    if total_true > 0:
        print(f"  Rough recall: {matched/total_true:.2%}")
    if total_pred > 0:
        print(f"  Rough precision: {matched/total_pred:.2%}")
    
    if not true_tags_list:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    return {
        "precision": precision_score(true_tags_list, pred_tags_list),
        "recall": recall_score(true_tags_list, pred_tags_list),
        "f1": f1_score(true_tags_list, pred_tags_list)
    }


# ==================== MAIN EVALUATION FUNCTION ====================

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
    
    # Get entity types from data
    from src.gpt.gpt_ner_classifier import extract_entity_types_from_data
    entity_types = extract_entity_types_from_data(test_tags)
    print(f"Entity types in dataset: {entity_types}")
    
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
        
        # Initialize GPT NER classifier with entity types
        deployment = deployment_override or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        classifier = GPTNERClassifier(
            deployment_name=deployment,
            strategy=strategy,
            temperature=0.0,
            entity_types=entity_types  # Pass entity types
        )
        
        # Run evaluation
        start_time = time.time()
        try:
            if force_batch and len(sentences) >= batch_threshold:
                print(f"Using batch processing for {len(sentences)} samples...")
                pred_entities_list = classifier.extract_entities_batch(sentences, work_dir=out_dir)
            else:
                print(f"Using online processing for {len(sentences)} samples...")
                pred_entities_list = []
                for i, sentence in enumerate(sentences):
                    if i % 10 == 0:
                        print(f"  Processing {i+1}/{len(sentences)}...")
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
            
            print(f"\nResults: F1={metrics.get('f1', 0):.4f}, Precision={metrics.get('precision', 0):.4f}, Recall={metrics.get('recall', 0):.4f}")
            print(f"Time: {elapsed:.1f}s")
            
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
            import traceback
            traceback.print_exc()
            
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
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {out_root}")
        print(f"{'='*80}")
        print("\nSummary:")
        print(df.to_string(index=False))
    else:
        print("No successful evaluations")


# ==================== MAIN ENTRY POINT ====================

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