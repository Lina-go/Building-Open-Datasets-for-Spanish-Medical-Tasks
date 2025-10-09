"""
GPT NER Classifier - FIXED VERSION with BROADER scope
Key fix: Prompt now covers ALL entity types in AnatEM (anatomy, diseases, chemicals, pathways)
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
    Build improved prompt that covers ALL entity types in AnatEM dataset
    Including: anatomy, diseases (Cancer), chemicals, pathways, substances
    
    Args:
        text: The text to extract entities from
        strategy: "zero_shot" or "few_shot"
        entity_types: Optional list of valid entity types from the dataset
    """
    
    # Show actual entity types from the dataset
    if entity_types:
        types_str = ", ".join(sorted(entity_types))
        types_instruction = f"\nTIPOS VÁLIDOS: {types_str}"
    else:
        types_instruction = "\nEJEMPLOS DE TIPOS: Cell, Tissue, Organ, Cancer, Pathological_formation, Simple_chemical, Pathway"
    
    if strategy == "few_shot":
        # Use examples covering the ACTUAL entity types in AnatEM
        examples = """
EJEMPLOS:

Texto: "Las células HeLa son células cancerosas del cuello uterino."
Respuesta: [
  {"entity": "células HeLa", "type": "Cell"},
  {"entity": "células cancerosas", "type": "Cell"},
  {"entity": "cuello uterino", "type": "Organ"}
]

Texto: "El tumor primario es un astrocitoma en el lóbulo temporal."
Respuesta: [
  {"entity": "tumor", "type": "Cancer"},
  {"entity": "astrocitoma", "type": "Cancer"},
  {"entity": "lóbulo temporal", "type": "Multi-tissue_structure"}
]

Texto: "Las neuronas del cerebro producen dopamina."
Respuesta: [
  {"entity": "neuronas", "type": "Cell"},
  {"entity": "cerebro", "type": "Organ"},
  {"entity": "dopamina", "type": "Simple_chemical"}
]

Texto: "El tejido muscular contiene mitocondrias que generan ATP."
Respuesta: [
  {"entity": "tejido muscular", "type": "Tissue"},
  {"entity": "mitocondrias", "type": "Cellular_component"},
  {"entity": "ATP", "type": "Simple_chemical"}
]

Texto: "La vía de señalización Wnt regula células madre del sistema nervioso."
Respuesta: [
  {"entity": "vía de señalización Wnt", "type": "Pathway"},
  {"entity": "células madre", "type": "Cell"},
  {"entity": "sistema nervioso", "type": "Anatomical_system"}
]

Texto: "La sangre circula por el sistema cardiovascular llevando oxígeno."
Respuesta: [
  {"entity": "sangre", "type": "Organism_substance"},
  {"entity": "sistema cardiovascular", "type": "Anatomical_system"},
  {"entity": "oxígeno", "type": "Simple_chemical"}
]

"""
    else:
        examples = ""
    
    # MUCH BROADER instructions to match AnatEM dataset
    prompt = f"""Eres un experto en extracción de entidades biomédicas de textos científicos en español.

TAREA: Extrae TODAS las entidades biomédicas del texto.
{types_instruction}

CATEGORÍAS A INCLUIR:
1. ANATOMÍA:
   - Células: neuronas, células madre, linfocitos, células HeLa
   - Tejidos: tejido muscular, tejido nervioso, epitelio
   - Órganos: corazón, hígado, cerebro, pulmón
   - Sistemas: sistema nervioso, sistema cardiovascular
   - Componentes celulares: núcleo, mitocondria, membrana
   - Estructuras: lóbulo, ventrículo, corteza

2. ENFERMEDADES:
   - Cáncer: tumor, astrocitoma, carcinoma
   - Formaciones patológicas: lesión, quiste

3. SUSTANCIAS:
   - Químicos: dopamina, glucosa, ATP, oxígeno
   - Fluidos: sangre, linfa, líquido cefalorraquídeo

4. PROCESOS:
   - Vías: vía de señalización, pathway, ruta metabólica

FORMATO DE SALIDA:
- Array JSON con objetos {{"entity": "texto_exacto", "type": "tipo"}}
- Usa el texto EXACTO como aparece
- Incluye TODAS las entidades biomédicas relevantes
- Si no hay entidades: []
{examples}
TEXTO A ANALIZAR:
"{text}"

RESPUESTA JSON:"""
    
    return prompt


class GPTNERClassifier:
    """
    GPT-based NER classifier for biomedical entities
    Handles: anatomy, diseases, chemicals, pathways, substances
    """
    
    def __init__(self, deployment_name, strategy="zero_shot", temperature=0.0,
                 azure_endpoint=None, api_key=None, api_version=None,
                 entity_types=None):
        """
        Initialize GPT NER classifier
        
        Args:
            deployment_name: Azure OpenAI deployment name
            strategy: "zero_shot" or "few_shot"
            temperature: Temperature for generation (default 0.0)
            azure_endpoint: Azure endpoint URL
            api_key: Azure API key
            api_version: API version
            entity_types: List of valid entity types from dataset
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
        """Clean markdown code blocks from GPT response"""
        content = content.strip()
        
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        return content.strip()

    def extract_entities(self, text):
        """Extract biomedical entities from a single text"""
        prompt = build_ner_prompt(text, self.strategy, self.entity_types)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "Eres un experto en extracción de entidades biomédicas (anatomía, enfermedades, químicos, vías) de textos científicos en español. Respondes SOLO con JSON válido."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                top_p=1.0,
                max_tokens=1000  # Increased to 1000 for more entities
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
                            entity_type = item.get("type", "BIOMEDICAL").strip()
                            entities.append((entity_text, entity_type))
                    return entities
                else:
                    print(f"Warning: Response is not a list: {type(entities_data)}")
                    return []
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                print(f"Content: {content[:300]}")
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
                    {
                        "role": "system", 
                        "content": "Eres un experto en extracción de entidades biomédicas (anatomía, enfermedades, químicos, vías) de textos científicos en español. Respondes SOLO con JSON válido."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "top_p": 1.0,
                "max_tokens": 1000
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
            
            print(f"Batch job created: {batch_id}")
            print("Waiting for completion...")
            
            status, output_file_id, error_file_id = poll_batch_until_done(self.client, batch_id)
            
            if status != "completed" or not output_file_id:
                print(f"Batch failed with status: {status}")
                if error_file_id:
                    print(f"Error file ID: {error_file_id}")
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
                                entity_text = item["entity"].strip()
                                entity_type = item.get("type", "BIOMEDICAL").strip()
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