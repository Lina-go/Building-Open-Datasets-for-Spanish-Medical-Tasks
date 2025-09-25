"""
GPT NER Classifier for anatomical entity recognition
"""

import os
import json
import time
from openai import AzureOpenAI

from src.gpt.azure_batch_tools import (
    generate_jsonl, create_file, create_batch_job,
    poll_batch_until_done, download_bytes, parse_batch_output
)


def build_ner_prompt(text, strategy="zero_shot"):
    """Build prompt for NER task"""
    
    if strategy == "few_shot":
        examples = """
Examples:

Input: "The patient showed inflammation in the left ventricle and damage to the cardiac muscle."
Output: [{"entity": "left ventricle", "type": "ANATOMY"}, {"entity": "cardiac muscle", "type": "ANATOMY"}]

Input: "Examination revealed abnormal liver function and enlarged spleen."
Output: [{"entity": "liver", "type": "ANATOMY"}, {"entity": "spleen", "type": "ANATOMY"}]

Input: "MRI scan shows lesions in the frontal cortex and hippocampus region."
Output: [{"entity": "frontal cortex", "type": "ANATOMY"}, {"entity": "hippocampus", "type": "ANATOMY"}]

"""
    else:
        examples = ""
    
    return f"""You are an expert in anatomical entity recognition for medical texts.

Task: Extract all anatomical entities (body parts, organs, tissues, anatomical structures) from the input text.

Rules:
- Output ONLY a JSON array of objects
- Each object has: {{"entity": "text span", "type": "ANATOMY"}}
- Include only anatomical structures, organs, body parts, tissues
- Use exact text spans from the input
- If no anatomical entities found, output []
- Do not include diseases, symptoms, or procedures

{examples}Input: "{text}"
Output:"""


class GPTNERClassifier:
    """GPT-based NER classifier for anatomical entities"""
    
    def __init__(self, deployment_name, strategy="zero_shot", temperature=0.0,
                 azure_endpoint=None, api_key=None, api_version=None):
        self.deployment_name = deployment_name
        self.strategy = strategy
        self.temperature = float(temperature)

        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=version
        )
    
    def clean_json_content(self, content):
        """Helper method to clean markdown code blocks from JSON content"""
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        return content.strip()

    def extract_entities(self, text):
        """Extract entities from a single text"""
        prompt = build_ner_prompt(text, self.strategy)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert anatomical entity extractor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=1.0,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            content = self.clean_json_content(content)
            
            # Parse JSON response
            try:
                entities_data = json.loads(content)
                if isinstance(entities_data, list):
                    # Convert to (entity_text, entity_type) tuples
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
                # Try to extract entities from text if JSON parsing fails
                print(f"JSON parse error. Raw response: {content}")
                return []
                
        except Exception as e:
            print(f"Error in GPT request: {e}")
            return []

    def extract_entities_batch(self, texts, work_dir):
        """Extract entities from multiple texts using batch API"""
        os.makedirs(work_dir, exist_ok=True)
        input_jsonl = os.path.join(work_dir, "input.jsonl")

        # Generate batch file
        lines = []
        for i, text in enumerate(texts):
            prompt = build_ner_prompt(text, self.strategy)
            body = {
                "model": self.deployment_name,
                "messages": [
                    {"role": "system", "content": "You are an expert anatomical entity extractor."},
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

        # Write JSONL
        with open(input_jsonl, "w", encoding="utf-8", newline="\n") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")

        # Submit batch
        try:
            input_file_id = create_file(self.client, input_jsonl)
            batch_id = create_batch_job(self.client, input_file_id)
            
            status, output_file_id, error_file_id = poll_batch_until_done(self.client, batch_id)
            
            if status != "completed" or not output_file_id:
                print(f"Batch failed with status: {status}")
                return [[] for _ in texts]  # Return empty lists
            
            # Parse results
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