"""
Build k-NN index for improved GPT-NER
Creates FAISS index from training data for k-NN retrieval

Usage:
python -m src.scripts.build_ner_knn_index \
    --anatem-root data/AnatEM \
    --format nersuite-spanish \
    --output-dir models/knn_index \
    --model-name sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import os
from src.utils.ner_data import read_split_dir, read_flat_dir
from src.gpt.knn_retrieval import KNNRetriever


def main():
    parser = argparse.ArgumentParser(description="Build k-NN index for GPT-NER")
    parser.add_argument("--anatem-root", default="data/AnatEM", help="AnatEM root directory")
    parser.add_argument("--format", default="nersuite-spanish", 
                       choices=["conll", "nersuite", "nersuite-spanish"])
    parser.add_argument("--output-dir", default="models/knn_index", 
                       help="Directory to save index")
    parser.add_argument("--model-name", 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Sentence transformer model for encoding")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for encoding")
    parser.add_argument("--use-train-only", action="store_true",
                       help="Use only training split (if available)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Building k-NN Index for GPT-NER")
    print("="*80)
    
    # Load training data
    print(f"\nLoading data from: {args.anatem_root}")
    print(f"Format: {args.format}")
    
    has_train = os.path.isdir(os.path.join(args.anatem_root, args.format, "train"))
    
    if has_train and args.use_train_only:
        print("Using training split only")
        train_tokens, train_tags = read_split_dir(args.anatem_root, args.format, "train")
    elif has_train:
        print("Using train + dev splits")
        train_tokens, train_tags = read_split_dir(args.anatem_root, args.format, "train")
        dev_tokens, dev_tags = read_split_dir(args.anatem_root, args.format, "devel")
        train_tokens.extend(dev_tokens)
        train_tags.extend(dev_tags)
    else:
        print("No splits found, using 80% of data")
        all_tokens, all_tags = read_flat_dir(args.anatem_root, args.format)
        n_train = int(len(all_tokens) * 0.8)
        train_tokens = all_tokens[:n_train]
        train_tags = all_tags[:n_train]
    
    print(f"Training examples: {len(train_tokens)}")
    
    # Count entity types
    entity_types = set()
    for tags in train_tags:
        for tag in tags:
            if tag.startswith('B-') or tag.startswith('I-'):
                entity_types.add(tag[2:])
    
    print(f"Entity types: {len(entity_types)}")
    print(f"Types: {sorted(entity_types)}")
    
    # Build index
    print(f"\nBuilding k-NN index...")
    print(f"Model: {args.model_name}")
    
    retriever = KNNRetriever()
    retriever.build_index(
        train_tokens=train_tokens,
        train_tags=train_tags,
        model_name=args.model_name,
        batch_size=args.batch_size
    )
    
    # Save index
    os.makedirs(args.output_dir, exist_ok=True)
    retriever.save_index(args.output_dir)
    
    # Save entity types
    import json
    with open(os.path.join(args.output_dir, "entity_types.json"), "w") as f:
        json.dump(sorted(entity_types), f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Index saved to: {args.output_dir}")
    print(f"{'='*80}")
    
    # Test retrieval
    print("\nTesting retrieval...")
    test_sentence = " ".join(train_tokens[0])
    print(f"Query: {test_sentence[:100]}...")
    
    examples = retriever.retrieve_examples(test_sentence, k=3, model_name=args.model_name)
    print(f"\nTop 3 similar examples:")
    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. Score: {ex['score']:.4f}")
        print(f"   Sentence: {ex['sentence'][:80]}...")
        print(f"   Entities: {ex['entities'][:3]}")
    
    print("\n✓ Index ready for use!")


if __name__ == "__main__":
    main()