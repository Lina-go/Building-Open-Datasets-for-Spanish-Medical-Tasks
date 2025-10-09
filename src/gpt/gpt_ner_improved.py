"""
Improved GPT-NER with k-NN retrieval and self-verification
Based on: GPT-NER: Named Entity Recognition via Large Language Models
https://arxiv.org/pdf/2304.10428.pdf
"""

import os
import json
import time
from typing import List, Tuple, Dict, Optional
from openai import AzureOpenAI
from src.gpt.knn_retrieval import KNNRetriever


def build_knn_few_shot_prompt(text: str, 
                              examples: List[Dict],
                              entity_types: List[str],
                              k: int = 5) -> str:
    """
    Build improved few-shot prompt with k-NN retrieved examples
    
    Args:
        text: Test sentence
        examples: k-NN retrieved examples from training set
        entity_types: Valid entity types
        k: Number of examples to include
        
    Returns:
        Formatted prompt
    """
    
    types_str = ", ".join(sorted(entity_types))
    
    prompt = f"""Eres un experto en extracción de entidades biomédicas de textos científicos en español.

TAREA: Extrae TODAS las entidades biomédicas del texto.
TIPOS VÁLIDOS: {types_str}

FORMATO DE SALIDA:
- Array JSON con objetos {{"entity": "texto_exacto", "type": "tipo"}}
- Usa el texto EXACTO como aparece
- Si no hay entidades: []

EJEMPLOS DE ENTRENAMIENTO (ordenados por similitud):
"""
    
    # Add k-NN examples
    for i, example in enumerate(examples[:k], 1):
        sentence = example['sentence']
        entities = example['entities']
        
        # Format entities as JSON
        entities_json = [{"entity": e[0], "type": e[1]} for e in entities]
        
        prompt += f"\nEjemplo {i}:\n"
        prompt += f'Texto: "{sentence}"\n'
        prompt += f'Respuesta: {json.dumps(entities_json, ensure_ascii=False)}\n'
    
    prompt += f'\n\nAHORA ANALIZA EL SIGUIENTE TEXTO:\n'
    prompt += f'Texto: "{text}"\n'
    prompt += f'Respuesta JSON:'
    
    return prompt


def build_verification_prompt(text: str,
                              entity: str,
                              entity_type: str,
                              entity_types: List[str],
                              examples: Optional[List[Dict]] = None) -> str:
    """
    Build self-verification prompt (zero-shot or few-shot)
    
    Args:
        text: Original sentence
        entity: Entity to verify
        entity_type: Predicted entity type
        entity_types: Valid entity types
        examples: Optional verification examples
        
    Returns:
        Verification prompt
    """
    
    types_str = ", ".join(sorted(entity_types))
    
    prompt = f"""Eres un experto verificador de entidades biomédicas.

TAREA: Verificar si la entidad extraída es correcta.

TIPOS VÁLIDOS: {types_str}

"""
    
    # Add few-shot examples if provided
    if examples:
        prompt += "EJEMPLOS:\n\n"
        for ex in examples[:3]:
            ex_sentence = ex.get('sentence', '')
            ex_entity = ex.get('entity', '')
            ex_type = ex.get('type', '')
            ex_correct = ex.get('correct', True)
            answer = "SÍ" if ex_correct else "NO"
            
            prompt += f'Texto: "{ex_sentence}"\n'
            prompt += f'Entidad: "{ex_entity}" (tipo: {ex_type})\n'
            prompt += f'¿Es correcta? {answer}\n\n'
    
    prompt += f'VERIFICA LA SIGUIENTE EXTRACCIÓN:\n'
    prompt += f'Texto: "{text}"\n'
    prompt += f'Entidad extraída: "{entity}" (tipo: {entity_type})\n\n'
    prompt += f'¿Es "{entity}" realmente una entidad de tipo {entity_type} en este texto?\n'
    prompt += f'Responde SOLO: SÍ o NO\n'
    prompt += f'Respuesta:'
    
    return prompt


class ImprovedGPTNER:
    """
    Improved GPT-NER with k-NN retrieval and self-verification
    """
    
    def __init__(self,
                 deployment_name: str,
                 retriever: Optional[KNNRetriever] = None,
                 use_verification: bool = True,
                 temperature: float = 0.0,
                 azure_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_version: Optional[str] = None,
                 entity_types: Optional[List[str]] = None):
        """
        Initialize improved GPT-NER
        
        Args:
            deployment_name: Azure OpenAI deployment
            retriever: KNNRetriever instance (for few-shot with k-NN)
            use_verification: Whether to use self-verification
            temperature: Temperature for generation
            azure_endpoint: Azure endpoint URL
            api_key: Azure API key
            api_version: API version
            entity_types: List of valid entity types
        """
        self.deployment_name = deployment_name
        self.retriever = retriever
        self.use_verification = use_verification
        self.temperature = float(temperature)
        self.entity_types = entity_types or []
        
        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=version
        )
    
    def extract_entities(self, text: str, k: int = 5) -> List[Tuple[str, str]]:
        """
        Extract entities with k-NN few-shot prompting
        
        Args:
            text: Input sentence
            k: Number of k-NN examples to use
            
        Returns:
            List of (entity_text, entity_type) tuples
        """
        # Build prompt
        if self.retriever:
            # k-NN few-shot
            examples = self.retriever.retrieve_examples(text, k=k)
            prompt = build_knn_few_shot_prompt(text, examples, self.entity_types, k=k)
        else:
            # Zero-shot fallback
            prompt = self._build_zero_shot_prompt(text)
        
        # Call GPT
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un experto en extracción de entidades biomédicas. Respondes SOLO con JSON válido."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                top_p=1.0,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            content = self._clean_json_content(content)
            
            # Parse entities
            entities_data = json.loads(content)
            if not isinstance(entities_data, list):
                return []
            
            entities = []
            for item in entities_data:
                if isinstance(item, dict) and "entity" in item:
                    entity_text = item["entity"].strip()
                    entity_type = item.get("type", "BIOMEDICAL").strip()
                    entities.append((entity_text, entity_type))
            
            # Self-verification
            if self.use_verification and entities:
                entities = self._verify_entities(text, entities)
            
            return entities
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []
    
    def _verify_entities(self, text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Self-verification step to filter false positives
        
        Args:
            text: Original sentence
            entities: Predicted entities
            
        Returns:
            Filtered entities
        """
        verified = []
        
        for entity_text, entity_type in entities:
            prompt = build_verification_prompt(
                text, entity_text, entity_type, self.entity_types
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un experto verificador de entidades biomédicas."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                
                answer = response.choices[0].message.content.strip().upper()
                
                # Keep entity if verified
                if "SÍ" in answer or "SI" in answer or "YES" in answer:
                    verified.append((entity_text, entity_type))
                
                # Small delay to avoid rate limiting
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Verification error: {e}")
                # Keep entity on error (conservative)
                verified.append((entity_text, entity_type))
        
        return verified
    
    def _build_zero_shot_prompt(self, text: str) -> str:
        """Build simple zero-shot prompt as fallback"""
        types_str = ", ".join(sorted(self.entity_types))
        
        return f"""Extrae entidades biomédicas del texto.

TIPOS: {types_str}

FORMATO: Array JSON con {{"entity": "...", "type": "..."}}

Texto: "{text}"

JSON:"""
    
    def _clean_json_content(self, content: str) -> str:
        """Clean markdown code blocks from response"""
        content = content.strip()
        
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        return content.strip()