"""
Debug script for GPT NER - diagnose why GPT returns zero entities

Usage:
export AZURE_OPENAI_ENDPOINT="https://<endpoint>.openai.azure.com"
export AZURE_OPENAI_API_KEY="xxxxxxxx"
python -m src.scripts.debug_gpt_ner
"""

import os
import json
from openai import AzureOpenAI
from src.utils.ner_data import read_flat_dir


def extract_entity_types(tags_list):
    """Extract unique entity types from BIO tags"""
    entity_types = set()
    for tags in tags_list:
        for tag in tags:
            if tag.startswith('B-') or tag.startswith('I-'):
                entity_type = tag[2:]
                entity_types.add(entity_type)
    return sorted(entity_types)


def build_prompt_v1(text, entity_types):
    """Original complex prompt"""
    entity_types_str = ", ".join(entity_types)
    return f"""Eres un experto en reconocimiento de entidades anatómicas en textos médicos.

TAREA: Extrae TODAS las menciones de partes del cuerpo, órganos, tejidos, células y estructuras anatómicas del siguiente texto en español.

TIPOS VÁLIDOS: {entity_types_str}

INCLUYE:
- Órganos (corazón, hígado, cerebro, etc.)
- Tejidos (tejido muscular, tejido nervioso, etc.)
- Células (células madre, células cancerosas, neuronas, etc.)
- Estructuras (ventrículo, núcleo, membrana, etc.)

NO INCLUYAS:
- Enfermedades, procedimientos, medicamentos, síntomas

FORMATO DE SALIDA:
- SOLO un array JSON
- Cada objeto: {{"entity": "texto exacto", "type": "tipo"}}
- Si no hay entidades: []

TEXTO: "{text}"

RESPUESTA:"""


def build_prompt_v2_simple(text):
    """Ultra simple prompt"""
    return f"""Extrae órganos, células, tejidos del texto en español.

Texto: "{text}"

JSON [{{"entity": "...", "type": "..."}}]:"""


def build_prompt_v3_examples(text):
    """Simple with examples"""
    return f"""Extrae entidades anatómicas del texto médico.

Ejemplos:
"células del hígado" → [{{"entity": "células", "type": "Cell"}}, {{"entity": "hígado", "type": "Organ"}}]

Texto: "{text}"

JSON:"""


def test_prompts():
    """Test different prompt styles on real data"""
    
    # Setup
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
    
    # Load data
    print("Loading data from data/AnatEM/nersuite-spanish...")
    try:
        all_tokens, all_tags = read_flat_dir("data/AnatEM", "nersuite-spanish")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    entity_types = extract_entity_types(all_tags)
    print(f"Entity types: {entity_types}")
    
    # Find a sentence with entities
    test_idx = None
    for i, (tokens, tags) in enumerate(zip(all_tokens[:50], all_tags[:50])):
        # Count entities
        n_entities = sum(1 for tag in tags if tag.startswith('B-'))
        if 3 <= n_entities <= 8:  # Good test case
            test_idx = i
            break
    
    if test_idx is None:
        test_idx = 10
    
    tokens = all_tokens[test_idx]
    tags = all_tags[test_idx]
    sentence = " ".join(tokens)
    
    # Extract true entities
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
    
    print("\n" + "="*80)
    print("TEST SENTENCE")
    print("="*80)
    print(f"Text: {sentence[:200]}...")
    print(f"\nTrue entities ({len(true_entities)}):")
    for entity, etype in true_entities:
        print(f"  - '{entity}' ({etype})")
    
    # Test different prompts
    prompts = {
        "v1_complex": build_prompt_v1(sentence, entity_types),
        "v2_ultra_simple": build_prompt_v2_simple(sentence),
        "v3_with_examples": build_prompt_v3_examples(sentence),
    }
    
    results = {}
    
    for name, prompt in prompts.items():
        print(f"\n{'='*80}")
        print(f"TESTING: {name}")
        print(f"{'='*80}")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"First 150 chars: {prompt[:150]}...")
        
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "Extraes entidades anatómicas de textos médicos en español."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Clean
            content = raw_content
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            print(f"\nGPT Response: {raw_content[:200]}...")
            
            # Parse
            try:
                entities_data = json.loads(content)
                
                if isinstance(entities_data, list):
                    pred_entities = []
                    for item in entities_data:
                        if isinstance(item, dict) and "entity" in item:
                            pred_entities.append((item["entity"], item.get("type", "?")))
                    
                    results[name] = {
                        "success": True,
                        "count": len(pred_entities),
                        "entities": pred_entities
                    }
                    
                    print(f"\n✓ Extracted {len(pred_entities)} entities:")
                    for entity, etype in pred_entities[:5]:
                        print(f"    '{entity}' ({etype})")
                    if len(pred_entities) > 5:
                        print(f"    ... and {len(pred_entities) - 5} more")
                    
                    if len(pred_entities) == 0:
                        print("\n❌ PROBLEM: Returned empty array!")
                
                else:
                    results[name] = {"success": False, "error": "Not a list"}
                    print(f"❌ Not a list: {type(entities_data)}")
            
            except json.JSONDecodeError as e:
                results[name] = {"success": False, "error": f"JSON error: {e}"}
                print(f"❌ JSON parse error: {e}")
        
        except Exception as e:
            results[name] = {"success": False, "error": str(e)}
            print(f"❌ API error: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"True entities: {len(true_entities)}")
    print(f"\nPrompt results:")
    
    best_name = None
    best_count = 0
    
    for name, result in results.items():
        if result.get("success"):
            count = result["count"]
            print(f"  {name}: {count} entities")
            if count > best_count:
                best_count = count
                best_name = name
        else:
            print(f"  {name}: FAILED - {result.get('error')}")
    
    if best_name and best_count > 0:
        print(f"\n✅ BEST: {best_name} extracted {best_count} entities")
        print(f"\nRecommendation: Use prompt style '{best_name}'")
    else:
        print("\n❌ ALL PROMPTS FAILED!")
        print("\nPossible issues:")
        print("  1. GPT deployment doesn't support Spanish well")
        print("  2. Model temperature/parameters wrong")
        print("  3. Azure OpenAI API configuration issue")
        print("  4. Text is too noisy/technical")
        print("\nTry:")
        print("  - Test with English: 'cells in the liver'")
        print("  - Check deployment supports chat completions")
        print("  - Verify API key and endpoint")


if __name__ == "__main__":
    test_prompts()