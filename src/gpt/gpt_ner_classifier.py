"""
GPT NER Classifier - IMPROVED VERSION
Key improvements:
- Better prompts with multiple diverse examples
- Increased max_tokens for longer responses
- Better system message
- Pass entity types to prompt for better guidance
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
    """
    Build improved prompt with better examples and instructions
    
    Args:
        text: The text to extract entities from
        strategy: "zero_shot" or "few_shot"
        entity_types: Optional list of valid entity types from the dataset
    """
    
    # Get actual entity types from the dataset if provided
    if entity_types:
        types_str = ", ".join(sorted(entity_types))
        types_instruction = f"\nTIPOS VÁLIDOS: {types_str}"
    else:
        types_instruction = "\nTIPOS: Cell, Tissue, Organ, Cellular_component, Organism_subdivision"
    
    if strategy == "few_shot":
        # Use MULTIPLE diverse examples covering different entity types
        examples = """
EJEMPLOS:

Texto: "Las células madre del hígado se dividen rápidamente."
Respuesta: [
  {"entity": "células madre", "type": "Cell"},
  {"entity": "hígado", "type": "Organ"}
]

Texto: "El tejido muscular del ventrículo izquierdo estaba inflamado."
Respuesta: [
  {"entity": "tejido muscular", "type": "Tissue"},
  {"entity": "ventrículo izquierdo", "type": "Organ"}
]

Texto: "Observamos el núcleo celular y la membrana plasmática."
Respuesta: [
  {"entity": "núcleo celular", "type": "Cellular_component"},
  {"entity": "membrana plasmática", "type": "Cellular_component"}
]

Texto: "La corteza cerebral presenta neuronas dañadas."
Respuesta: [
  {"entity": "corteza cerebral", "type": "Tissue"},
  {"entity": "neuronas", "type": "Cell"}
]

Texto: "El sistema nervioso central incluye el cerebro y la médula espinal."
Respuesta: [
  {"entity": "sistema nervioso central", "type": "Organism_subdivision"},
  {"entity": "cerebro", "type": "Organ"},
  {"entity": "médula espinal", "type": "Organ"}
]

"""
    else:
        examples = ""
    
    # Comprehensive instructions
    prompt = f"""Eres un experto en reconocimiento de entidades anatómicas en textos médicos en español.

TAREA: Extrae TODAS las menciones de estructuras anatómicas del texto.
{types_instruction}

INCLUYE:
- Órganos: corazón, hígado, pulmón, cerebro
- Células: neuronas, células madre, linfocitos
- Tejidos: tejido muscular, tejido nervioso, epitelio
- Componentes celulares: núcleo, mitocondria, membrana
- Subdivisiones: lóbulo, ventrículo, corteza, sistema nervioso

NO INCLUYAS:
- Enfermedades, procedimientos, medicamentos, síntomas

FORMATO DE SALIDA:
- Array JSON con objetos {{"entity": "texto_exacto", "type": "tipo"}}
- Usa el texto EXACTO como aparece en el texto original
- Si no hay entidades anatómicas: []
{examples}
TEXTO A ANALIZAR:
"{text}"

RESPUESTA JSON:"""
    
    return prompt


class GPTNERClassifier:
    """
    GPT-based NER classifier with improved prompts and error handling
    """
    
    def __init__(self, deployment_name, strategy="zero_shot", temperature=0.0,
                 azure_endpoint=None, api_key=None, api_version=None,
                 entity_types=None):
        """
        Initialize GPT NER classifier
        
        Args:
            deployment_name: Azure OpenAI deployment name
            strategy: "zero_shot" or "few_shot"
            temperature: Temperature for generation (default 0.0 for deterministic)
            azure_endpoint: Azure endpoint URL (or from env)
            api_key: Azure API key (or from env)
            api_version: API version (default 2024-10-21)
            entity_types: Optional list of valid entity types
        """
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
        """
        Clean markdown code blocks from GPT response
        
        Args:
            content: Raw content from GPT
            
        Returns:
            Cleaned JSON string
        """
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        return content.strip()

    def extract_entities(self, text):
        """
        Extract entities from a single text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of tuples (entity_text, entity_type)
        """
        prompt = build_ner_prompt(text, self.strategy, self.entity_types)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un experto en extracción de entidades anatómicas de textos médicos en español. Respondes SOLO con JSON válido."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                top_p=1.0,
                max_tokens=800  # Increased from 500 to allow longer responses
            )
            
            content = response.choices[0].message.content.strip()
            content = self.clean_json_content(content)
            
            try:
                entities_data = json.loads(content)
                if isinstance(entities_data, list):
                    entities = []
                    for item in entities_data:
                        if isinstance(item, dict) and "entity" in item:
                            entity_text = item["entity"].strip()
                            entity_type = item.get("type", "ANATOMY").strip()
                            entities.append((entity_text, entity_type))
                    return entities
                else:
                    print(f"Warning: Response is not a list: {type(entities_data)}")
                    return []
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                print(f"Content: {content[:200]}")
                return []
                
        except Exception as e:
            print(f"Error in GPT request: {e}")
            return []

    def extract_entities_batch(self, texts, work_dir):
        """
        Extract entities from multiple texts using batch API
        
        Args:
            texts: List of texts to process
            work_dir: Working directory for batch files
            
        Returns:
            List of lists of tuples (entity_text, entity_type)
        """
        os.makedirs(work_dir, exist_ok=True)
        input_jsonl = os.path.join(work_dir, "input.jsonl")

        # Generate batch input file
        lines = []
        for i, text in enumerate(texts):
            prompt = build_ner_prompt(text, self.strategy, self.entity_types)
            body = {
                "model": self.deployment_name,
                "messages": [
                    {
                        "role": "system", 
                        "content": "Eres un experto en extracción de entidades anatómicas de textos médicos en español. Respondes SOLO con JSON válido."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "top_p": 1.0,
                "max_tokens": 800
            }
            lines.append({
                "custom_id": f"task-{i}",
                "method": "POST",
                "url": "/chat/completions",
                "body": body
            })

        # Write input file
        with open(input_jsonl, "w", encoding="utf-8", newline="\n") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write("\n")

        try:
            # Upload and create batch job
            input_file_id = create_file(self.client, input_jsonl)
            batch_id = create_batch_job(self.client, input_file_id)
            
            print(f"Batch job created: {batch_id}")
            print("Waiting for completion...")
            
            # Wait for completion
            status, output_file_id, error_file_id = poll_batch_until_done(self.client, batch_id)
            
            if status != "completed" or not output_file_id:
                print(f"Batch failed with status: {status}")
                if error_file_id:
                    print(f"Error file ID: {error_file_id}")
                return [[] for _ in texts]
            
            # Download and parse results
            raw_bytes = download_bytes(self.client, output_file_id)
            results_by_id = parse_batch_output(raw_bytes)
            
            # Extract entities from results
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
                                entity_text = item["entity"].strip()
                                entity_type = item.get("type", "ANATOMY").strip()
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
    """
    Extract all unique entity types from BIO tags
    
    Args:
        tags_list: List of lists of BIO tags
        
    Returns:
        Sorted list of unique entity types
    """
    entity_types = set()
    for tags in tags_list:
        for tag in tags:
            if tag.startswith('B-') or tag.startswith('I-'):
                entity_type = tag[2:]
                entity_types.add(entity_type)
    return sorted(entity_types)