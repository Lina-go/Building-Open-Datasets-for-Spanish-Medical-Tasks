"""
Simple test for GPT NER
Prueba rápida del clasificador GPT para NER

Uso:
export AZURE_OPENAI_ENDPOINT="https://<endpoint>.openai.azure.com"
export AZURE_OPENAI_API_KEY="xxxxxxxx"

python -m src.scripts.test_gpt_ner --text "The patient showed inflammation in the cardiac muscle and left ventricle."
"""

import argparse
import json
from src.gpt.gpt_ner_classifier import GPTNERClassifier


def test_gpt_ner(text, strategy="zero_shot", deployment="gpt-4o-mini"):
    """Test GPT NER with a single sentence"""
    
    print(f"Testing GPT NER with strategy: {strategy}")
    print(f"Deployment: {deployment}")
    print(f"Input text: {text}")
    print("-" * 50)
    
    try:
        # Initialize classifier
        classifier = GPTNERClassifier(
            deployment_name=deployment,
            strategy=strategy,
            temperature=0.0
        )
        
        # Extract entities
        entities = classifier.extract_entities(text)
        
        print("Extracted entities:")
        if entities:
            for entity_text, entity_type in entities:
                print(f"  - {entity_text} ({entity_type})")
        else:
            print("  No entities found")
        
        print(f"\nTotal entities: {len(entities)}")
        return entities
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Test GPT NER classifier")
    parser.add_argument("--text", required=True, help="Text to analyze")
    parser.add_argument("--strategy", default="zero_shot", choices=["zero_shot", "few_shot"])
    parser.add_argument("--deployment", default="gpt-4o-mini", help="Azure OpenAI deployment")
    
    args = parser.parse_args()
    
    entities = test_gpt_ner(args.text, args.strategy, args.deployment)
    
    # Save results
    result = {
        "input_text": args.text,
        "strategy": args.strategy,
        "entities": [{"entity": e[0], "type": e[1]} for e in entities]
    }
    
    with open("test_gpt_ner_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResult saved to: test_gpt_ner_result.json")


if __name__ == "__main__":
    main()