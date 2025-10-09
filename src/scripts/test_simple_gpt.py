"""
Simple test with EASY sentences to verify GPT can extract at all

Usage:
export AZURE_OPENAI_ENDPOINT="https://<endpoint>.openai.azure.com"
export AZURE_OPENAI_API_KEY="xxxxxxxx"
python -m src.scripts.test_simple_gpt
"""

import os
import json
from openai import AzureOpenAI


def test_simple_extraction():
    """Test GPT with very simple, clear sentences"""
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    if not endpoint or not api_key:
        print("ERROR: Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
        return
    
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-10-21"
    )
    
    # Very simple test sentences
    test_cases = [
        {
            "text": "El paciente tiene problemas en el corazón y el hígado.",
            "expected": ["corazón", "hígado"]
        },
        {
            "text": "Se encontraron células cancerosas en el tejido pulmonar.",
            "expected": ["células cancerosas", "tejido pulmonar"]
        },
        {
            "text": "El cerebro y la médula espinal están afectados.",
            "expected": ["cerebro", "médula espinal"]
        }
    ]
    
    print("="*80)
    print("TESTING GPT WITH SIMPLE SENTENCES")
    print("="*80)
    
    successes = 0
    
    for i, test in enumerate(test_cases, 1):
        text = test["text"]
        expected = test["expected"]
        
        print(f"\n{'='*80}")
        print(f"TEST {i}")
        print(f"{'='*80}")
        print(f"Text: {text}")
        print(f"Expected: {expected}")
        
        # Simple prompt
        prompt = f"""Extrae órganos, células y tejidos del texto:

"{text}"

JSON [{{"entity": "...", "type": "..."}}]:"""
        
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "Extraes entidades anatómicas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean markdown
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            print(f"\nGPT Response: {content}")
            
            # Parse
            try:
                data = json.loads(content)
                
                if isinstance(data, list) and len(data) > 0:
                    print(f"\n✓ SUCCESS: Extracted {len(data)} entities")
                    for item in data:
                        if isinstance(item, dict):
                            print(f"    - {item.get('entity', '?')} ({item.get('type', '?')})")
                    successes += 1
                elif isinstance(data, list) and len(data) == 0:
                    print(f"\n✗ RETURNED EMPTY ARRAY")
                else:
                    print(f"\n✗ WRONG FORMAT: {type(data)}")
            
            except json.JSONDecodeError as e:
                print(f"\n✗ JSON PARSE ERROR: {e}")
        
        except Exception as e:
            print(f"\n✗ API ERROR: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"RESULTS: {successes}/{len(test_cases)} successful")
    print(f"{'='*80}")
    
    if successes == 0:
        print("\n❌ GPT CANNOT EXTRACT ENTITIES AT ALL!")
        print("\nPossible causes:")
        print("  1. Wrong deployment - check AZURE_OPENAI_DEPLOYMENT")
        print("  2. API configuration issue")
        print("  3. Model doesn't support Spanish")
        print("  4. Temperature/parameters wrong")
        print("\nTry:")
        print(f"  echo $AZURE_OPENAI_ENDPOINT")
        print(f"  echo $AZURE_OPENAI_DEPLOYMENT")
        print(f"  Current deployment: {deployment}")
    
    elif successes < len(test_cases):
        print(f"\n⚠️  PARTIAL SUCCESS ({successes}/{len(test_cases)})")
        print("  GPT can extract but not reliably")
        print("  May need prompt tuning")
    
    else:
        print("\n✅ SUCCESS! GPT can extract entities")
        print("  The problem is likely with:")
        print("  1. Complex prompts on real data")
        print("  2. Noisy/technical text in dataset")
        print("  3. Entity matching logic")


if __name__ == "__main__":
    test_simple_extraction()