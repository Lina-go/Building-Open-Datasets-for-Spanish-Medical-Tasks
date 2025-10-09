"""
Diagnostic script to test GPT NER improvements
Run this BEFORE the full evaluation to verify:
1. GPT can extract entities
2. Matching works correctly
3. Type mapping is correct

Usage:
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_API_KEY="..."
python -m src.scripts.test_improvements
"""

import os
import sys

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.ner_data import read_flat_dir
from src.gpt.gpt_ner_classifier import GPTNERClassifier, extract_entity_types_from_data


def test_extraction_quality():
    """Test if GPT is actually extracting entities"""
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    if not endpoint or not api_key:
        print("ERROR: Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
        return False
    
    print("="*80)
    print("TEST 1: GPT Entity Extraction Quality")
    print("="*80)
    
    # Load real data
    try:
        all_tokens, all_tags = read_flat_dir("data/AnatEM", "nersuite-spanish")
        entity_types = extract_entity_types_from_data(all_tags)
        print(f"✓ Loaded data: {len(all_tokens)} sentences")
        print(f"✓ Entity types in data: {entity_types}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    # Find good test sentences (with 2-5 entities)
    test_cases = []
    for tokens, tags in zip(all_tokens[:100], all_tags[:100]):
        n_entities = sum(1 for tag in tags if tag.startswith('B-'))
        if 2 <= n_entities <= 5:
            test_cases.append((tokens, tags))
            if len(test_cases) >= 5:
                break
    
    if not test_cases:
        print("✗ Could not find suitable test cases")
        return False
    
    print(f"✓ Selected {len(test_cases)} test sentences\n")
    
    # Test both strategies
    for strategy in ["zero_shot", "few_shot"]:
        print(f"\n{'='*80}")
        print(f"Testing {strategy.upper()}")
        print(f"{'='*80}")
        
        classifier = GPTNERClassifier(
            deployment_name=deployment,
            strategy=strategy,
            temperature=0.0,
            entity_types=entity_types
        )
        
        total_true = 0
        total_pred = 0
        exact_matches = 0
        
        for i, (tokens, tags) in enumerate(test_cases, 1):
            sentence = " ".join(tokens)
            
            # Get true entities
            true_entities = []
            current_entity = []
            current_type = None
            
            for token, tag in zip(tokens, tags):
                if tag.startswith('B-'):
                    if current_entity:
                        true_entities.append((" ".join(current_entity), current_type))
                    current_entity = [token]
                    current_type = tag[2:]
                elif tag.startswith('I-') and current_entity:
                    current_entity.append(token)
                else:
                    if current_entity:
                        true_entities.append((" ".join(current_entity), current_type))
                    current_entity = []
                    current_type = None
            
            if current_entity:
                true_entities.append((" ".join(current_entity), current_type))
            
            # Get predicted entities
            pred_entities = classifier.extract_entities(sentence)
            
            # Calculate matches
            true_texts = {e[0].lower() for e in true_entities}
            pred_texts = {e[0].lower() for e in pred_entities}
            matches = true_texts & pred_texts
            
            total_true += len(true_entities)
            total_pred += len(pred_entities)
            exact_matches += len(matches)
            
            print(f"\nTest {i}:")
            print(f"  Sentence: {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
            print(f"  True: {len(true_entities)} entities")
            for e, t in true_entities[:3]:
                print(f"    - '{e}' ({t})")
            print(f"  Predicted: {len(pred_entities)} entities")
            for e, t in pred_entities[:3]:
                print(f"    - '{e}' ({t})")
            print(f"  Exact matches: {len(matches)}")
        
        print(f"\n{strategy.upper()} SUMMARY:")
        print(f"  Total true entities: {total_true}")
        print(f"  Total predicted: {total_pred}")
        print(f"  Exact matches: {exact_matches}")
        if total_true > 0:
            recall = exact_matches / total_true
            print(f"  Rough recall: {recall:.1%}")
        if total_pred > 0:
            precision = exact_matches / total_pred
            print(f"  Rough precision: {precision:.1%}")
        
        # Criteria for success
        if strategy == "few_shot":
            if exact_matches < total_true * 0.3:  # Should get at least 30% recall
                print(f"\n✗ {strategy} FAILING - Not extracting enough entities")
                print("  Problem: GPT is not identifying entities correctly")
                print("  Solution: Improve prompts with more/better examples")
                return False
        else:
            if exact_matches < total_true * 0.1:  # Zero-shot should get at least 10%
                print(f"\n✗ {strategy} FAILING - Almost no extraction")
                return False
    
    print("\n" + "="*80)
    print("✓ EXTRACTION TEST PASSED")
    print("="*80)
    return True


def test_matching_logic():
    """Test if entity matching is working correctly"""
    print("\n" + "="*80)
    print("TEST 2: Entity Matching Logic")
    print("="*80)
    
    # Define matching functions locally for testing
    import re
    
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def strict_find_entity(entity_text, tokens):
        entity_tokens = entity_text.lower().split()
        matches = []
        
        # 1. Exact match
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if [t.lower() for t in tokens[i:i+len(entity_tokens)]] == entity_tokens:
                matches.append((i, i + len(entity_tokens)))
        
        if matches:
            return matches
        
        # 2. Normalized match
        entity_normalized = normalize_text(entity_text)
        for i in range(len(tokens)):
            for j in range(i+1, min(i+len(entity_tokens)+3, len(tokens)+1)):
                span_text = " ".join(tokens[i:j])
                span_normalized = normalize_text(span_text)
                if entity_normalized == span_normalized:
                    matches.append((i, j))
        
        return matches
    
    def smart_map_entity_type(pred_type, true_entity_types):
        pred_lower = pred_type.lower()
        
        # Direct match
        for true_type in true_entity_types:
            if pred_type.lower() == true_type.lower():
                return true_type
        
        # Mapping
        type_mapping = {
            'cell': 'Cell', 'cells': 'Cell', 'célula': 'Cell', 'células': 'Cell',
            'tissue': 'Tissue', 'tejido': 'Tissue',
            'organ': 'Organ', 'órgano': 'Organ',
            'component': 'Cellular_component', 'nucleus': 'Cellular_component',
        }
        
        for key, value in type_mapping.items():
            if key in pred_lower and value in true_entity_types:
                return value
        
        # Partial match
        for true_type in true_entity_types:
            true_lower = true_type.lower()
            if pred_lower in true_lower or true_lower in pred_lower:
                return true_type
        
        # Default
        if 'Cell' in true_entity_types:
            return 'Cell'
        return sorted(true_entity_types)[0] if true_entity_types else pred_type
    
    # Test cases
    test_cases = [
        {
            "name": "Exact match",
            "tokens": ["las", "células", "del", "hígado"],
            "entity": "células",
            "expected": [(1, 2)]
        },
        {
            "name": "Multi-word exact match",
            "tokens": ["el", "tejido", "muscular", "está", "dañado"],
            "entity": "tejido muscular",
            "expected": [(1, 3)]
        },
        {
            "name": "Case insensitive",
            "tokens": ["El", "Hígado", "está", "inflamado"],
            "entity": "hígado",
            "expected": [(1, 2)]
        },
        {
            "name": "No match",
            "tokens": ["las", "células", "son", "pequeñas"],
            "entity": "hígado",
            "expected": []
        }
    ]
    
    all_passed = True
    for test in test_cases:
        result = strict_find_entity(test["entity"], test["tokens"])
        passed = result == test["expected"]
        status = "✓" if passed else "✗"
        print(f"{status} {test['name']}: {result}")
        if not passed:
            print(f"  Expected: {test['expected']}")
            all_passed = False
    
    # Test type mapping
    print("\nType Mapping Tests:")
    true_types = ["Cell", "Tissue", "Organ", "Cellular_component"]
    
    type_tests = [
        ("Cell", "Cell"),
        ("célula", "Cell"),
        ("cells", "Cell"),
        ("Tissue", "Tissue"),
        ("tejido", "Tissue"),
        ("Organ", "Organ"),
        ("órgano", "Organ"),
        ("nucleus", "Cellular_component"),
    ]
    
    for pred, expected in type_tests:
        result = smart_map_entity_type(pred, true_types)
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"{status} '{pred}' -> '{result}' (expected '{expected}')")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ MATCHING TEST PASSED")
    else:
        print("\n✗ MATCHING TEST FAILED")
    
    return all_passed


def main():
    """Run all diagnostic tests"""
    print("\n" + "="*80)
    print("GPT NER DIAGNOSTIC TESTS")
    print("="*80)
    
    tests_passed = 0
    tests_total = 2
    
    # Test 1: Extraction
    if test_extraction_quality():
        tests_passed += 1
    else:
        print("\n⚠️  Extraction test failed - fix prompts before proceeding")
    
    # Test 2: Matching
    if test_matching_logic():
        tests_passed += 1
    else:
        print("\n⚠️  Matching test failed - fix matching logic")
    
    # Summary
    print("\n" + "="*80)
    print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
    print("="*80)
    
    if tests_passed == tests_total:
        print("\n✅ ALL TESTS PASSED")
        print("\nNext steps:")
        print("1. Update your code with the improved versions")
        print("2. Run full evaluation:")
        print("   python -m src.scripts.gpt_ner_runner --config configs/gpt_ner_fixed.yaml")
        print("3. Expected results:")
        print("   - Zero-shot F1: 0.20-0.40")
        print("   - Few-shot F1: 0.40-0.60")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nFix the issues above before running full evaluation")
    
    return tests_passed == tests_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)