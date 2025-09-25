#!/usr/bin/env python3
"""
Simple test for NER data loading

Usage:
  python -m src.scripts.test_ner_data --anatem-root data/AnatEM --format nersuite-spanish
"""

import os
import argparse
from collections import Counter

from src.utils.ner_data import read_flat_dir, read_split_dir

def check_data_structure(anatem_root, format_type):
    """Check if data has split structure or flat structure"""
    splits = ["train", "devel", "test"]
    has_splits = all(os.path.isdir(os.path.join(anatem_root, format_type, split)) for split in splits)
    return has_splits

def load_data(anatem_root, format_type):
    """Load data using appropriate method"""
    if check_data_structure(anatem_root, format_type):
        print("Found split directories")
        train_tokens, train_tags = read_split_dir(anatem_root, format_type, "train")
        dev_tokens, dev_tags = read_split_dir(anatem_root, format_type, "devel") 
        test_tokens, test_tags = read_split_dir(anatem_root, format_type, "test")
        
        print(f"Train: {len(train_tokens)}, Dev: {len(dev_tokens)}, Test: {len(test_tokens)}")
        return train_tokens, train_tags
    else:
        print("Using flat directory")
        all_tokens, all_tags = read_flat_dir(anatem_root, format_type)
        print(f"Total: {len(all_tokens)} sentences")
        return all_tokens, all_tags

def analyze_data(tokens, tags):
    """Print basic statistics"""
    lengths = [len(sent) for sent in tokens]
    print(f"\nSentence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    
    all_tags = [tag for sent in tags for tag in sent]
    tag_counts = Counter(all_tags)
    print(f"\nTop 10 tags:")
    for tag, count in tag_counts.most_common(10):
        print(f"  {tag}: {count}")
    
    entities = [tag for tag in all_tags if tag != 'O' and '-' in tag]
    entity_types = set(tag[2:] for tag in entities)
    print(f"\nEntity types found: {len(entity_types)}")
    for etype in sorted(entity_types):
        print(f"  {etype}")

def show_sample(tokens, tags):
    """Show sample sentence"""
    if tokens:
        print(f"\nSample sentence:")
        sample_tokens = tokens[0][:10]
        sample_tags = tags[0][:10]
        for token, tag in zip(sample_tokens, sample_tags):
            print(f"  {token:15} -> {tag}")

def test_data_loading(anatem_root="data/AnatEM", format_type="nersuite-spanish"):
    """Main test function"""
    print(f"Testing data from: {anatem_root}")
    print(f"Format: {format_type}")
    print("-" * 40)
    
    try:
        tokens, tags = load_data(anatem_root, format_type)
        analyze_data(tokens, tags)
        show_sample(tokens, tags)
        print("\nData loading test completed successfully")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NER data loading")
    parser.add_argument("--anatem-root", default="data/AnatEM", help="Path to AnatEM root directory")
    parser.add_argument("--format", default="nersuite-spanish", choices=["conll", "nersuite", "nersuite-spanish"])
    args = parser.parse_args()
    
    success = test_data_loading(args.anatem_root, args.format)
    exit(0 if success else 1)