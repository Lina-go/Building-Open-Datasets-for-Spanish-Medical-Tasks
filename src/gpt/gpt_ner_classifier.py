"""
GPT NER Classifier - FINAL VERSION with v3 prompt that actually works
"""

import os
import json
import time
from openai import AzureOpenAI

from src.gpt.azure_batch_tools import (
    generate_jsonl, create_file, create_batch_job,
    poll_batch_until_done, download_bytes, parse_batch_output
)


def build_ner_prompt(text, strategy="zero_shot", entity_types=None):
    """Build prompt using v3 style that worked best in debug"""
    
    if strategy == "few_shot":
        # Use the EXACT examples that worked in debug test
        examples = """Ejemplos:
"células del hígado" → [{"entity": "células", "type": "Cell"}, {"entity": "hígado", "type": "Organ"}]

"""
    else:
        examples = ""
    
    # v3 style: Simple with examples
    return f"""Extrae entidades anatómicas del texto médico.

{examples}Texto: "{text}"

JSON:"""


class GPTNERClassifier:
    """GPT-based NER classifier using v3 prompt style"""
    
    def __init__(self, deployment_name, strategy="zero_shot", temperature=0.0,
                 azure_endpoint=None, api_key=None, api_version=None,
                 entity_types=None):
        self.deployment_name = deployment_name
        self.strategy = strategy
        self.temperature = float(temperature)
        self.entity_types = entity_types

        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=version
        )
    
    def clean_json_content(self, content):
        """Clean markdown code blocks"""
        content = content.strip()
        
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        return content.strip()

    def extract_entities(self, text):
        """Extract entities from a single text"""
        prompt = build_ner_prompt(text, self.strategy, self.entity_types)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "Extraes entidades anatómicas de textos médicos en español."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=1.0,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            content = self.clean_json_content(content)
            
            try:
                entities_data = json.loads(content)
                if isinstance(entities_data, list):
                    entities = []
                    for item in entities_data:
                        if isinstance(item, dict) and "entity" in item:
                            entity_text = item["entity"]
                            entity_type = item.get("type", "ANATOMY")
                            entities.append((entity_text, entity_type))
                    return entities
                else:
                    return []
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            print(f"Error in GPT request: {e}")
            return []

    def extract_entities_batch(self, texts, work_dir):
        """Extract entities from multiple texts using batch API"""
        os.makedirs(work_dir, exist_ok=True)
        input_jsonl = os.path.join(work_dir, "input.jsonl")

        lines = []
        for i, text in enumerate(texts):
            prompt = build_ner_prompt(text, self.strategy, self.entity_types)
            body = {
                "model": self.deployment_name,
                "messages": [
                    {"role": "system", "content": "Extraes entidades anatómicas de textos médicos en español."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "top_p": 1.0,
                "max_tokens": 500
            }
            lines.append({
                "custom_id": f"task-{i}",
                "method": "POST",
                "url": "/chat/completions",
                "body": body
            })

        with open(input_jsonl, "w", encoding="utf-8", newline="\n") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")

        try:
            input_file_id = create_file(self.client, input_jsonl)
            batch_id = create_batch_job(self.client, input_file_id)
            
            status, output_file_id, error_file_id = poll_batch_until_done(self.client, batch_id)
            
            if status != "completed" or not output_file_id:
                print(f"Batch failed with status: {status}")
                return [[] for _ in texts]
            
            raw_bytes = download_bytes(self.client, output_file_id)
            results_by_id = parse_batch_output(raw_bytes)
            
            entities_list = []
            for i in range(len(texts)):
                task_result = results_by_id.get(f"task-{i}", {})
                content = task_result.get("content", "[]")
                content = self.clean_json_content(content)
                
                try:
                    entities_data = json.loads(content)
                    if isinstance(entities_data, list):
                        entities = []
                        for item in entities_data:
                            if isinstance(item, dict) and "entity" in item:
                                entity_text = item["entity"]
                                entity_type = item.get("type", "ANATOMY")
                                entities.append((entity_text, entity_type))
                        entities_list.append(entities)
                    else:
                        entities_list.append([])
                except json.JSONDecodeError:
                    entities_list.append([])
            
            return entities_list
            
        except Exception as e:
            print(f"Batch processing error: {e}")
            return [[] for _ in texts]


def extract_entity_types_from_data(tags_list):
    """Extract all unique entity types from BIO tags"""
    entity_types = set()
    for tags in tags_list:
        for tag in tags:
            if tag.startswith('B-') or tag.startswith('I-'):
                entity_type = tag[2:]
                entity_types.add(entity_type)
    return sorted(entity_types)