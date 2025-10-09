"""
k-NN retrieval for GPT-NER using sentence embeddings (SimCSE approach)
Inspired by: https://arxiv.org/pdf/2304.10428.pdf
"""

import os
import json
import numpy as np
import faiss
from typing import List, Tuple, Dict
from tqdm import tqdm


class KNNRetriever:
    """
    Retrieve similar training examples using sentence embeddings
    Following GPT-NER paper's approach with SimCSE
    """
    
    def __init__(self, index_path: str = None):
        """
        Initialize retriever
        
        Args:
            index_path: Path to saved FAISS index (optional)
        """
        self.index = None
        self.train_sentences = []
        self.train_entities = []
        self.embeddings = None
        
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
    
    def build_index(self, 
                   train_tokens: List[List[str]], 
                   train_tags: List[List[str]],
                   model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                   batch_size: int = 32):
        """
        Build FAISS index from training data
        
        Args:
            train_tokens: List of token lists
            train_tags: List of BIO tag lists
            model_name: Sentence transformer model
            batch_size: Batch size for encoding
        """
        from sentence_transformers import SentenceTransformer
        
        print(f"Building k-NN index with {len(train_tokens)} examples...")
        
        # Store training data
        self.train_sentences = [" ".join(tokens) for tokens in train_tokens]
        
        # Extract entities for each sentence
        self.train_entities = []
        for tokens, tags in zip(train_tokens, train_tags):
            entities = self._extract_entities_from_bio(tokens, tags)
            self.train_entities.append(entities)
        
        # Encode sentences
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        
        print("Encoding sentences...")
        self.embeddings = model.encode(
            self.train_sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
        self.index.add(self.embeddings.astype(np.float32))
        
        print(f"✓ Index built with {self.index.ntotal} examples")
    
    def retrieve_examples(self, 
                         query_sentence: str,
                         k: int = 5,
                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[Dict]:
        """
        Retrieve k most similar training examples
        
        Args:
            query_sentence: Test sentence to find examples for
            k: Number of examples to retrieve
            model_name: Model to encode query
            
        Returns:
            List of dicts with 'sentence', 'entities', 'score'
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        from sentence_transformers import SentenceTransformer
        
        # Encode query
        model = SentenceTransformer(model_name)
        query_embedding = model.encode(
            [query_sentence],
            normalize_embeddings=True
        )
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'sentence': self.train_sentences[idx],
                'entities': self.train_entities[idx],
                'score': float(score)
            })
        
        return results
    
    def _extract_entities_from_bio(self, tokens: List[str], tags: List[str]) -> List[Tuple[str, str]]:
        """Extract entities from BIO tags"""
        entities = []
        current_entity = []
        current_type = None
        
        for token, tag in zip(tokens, tags):
            if tag.startswith('B-'):
                if current_entity:
                    entities.append((" ".join(current_entity), current_type))
                current_entity = [token]
                current_type = tag[2:]
            elif tag.startswith('I-') and current_entity:
                current_entity.append(token)
            else:
                if current_entity:
                    entities.append((" ".join(current_entity), current_type))
                current_entity = []
                current_type = None
        
        if current_entity:
            entities.append((" ".join(current_entity), current_type))
        
        return entities
    
    def save_index(self, save_dir: str):
        """Save index and metadata"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
        
        # Save metadata
        metadata = {
            'train_sentences': self.train_sentences,
            'train_entities': self.train_entities,
        }
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Save embeddings
        np.save(os.path.join(save_dir, "embeddings.npy"), self.embeddings)
        
        print(f"✓ Index saved to {save_dir}")
    
    def load_index(self, index_dir: str):
        """Load index and metadata"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        
        # Load metadata
        with open(os.path.join(index_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.train_sentences = metadata['train_sentences']
        self.train_entities = metadata['train_entities']
        
        # Load embeddings
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        
        print(f"✓ Index loaded from {index_dir} ({self.index.ntotal} examples)")


def build_entity_level_index(train_tokens: List[List[str]], 
                             train_tags: List[List[str]],
                             model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict:
    """
    Build entity-level k-NN index (more advanced approach from GPT-NER paper)
    Groups examples by entity type for better retrieval
    
    Returns:
        Dict mapping entity types to KNNRetriever instances
    """
    from collections import defaultdict
    
    # Group by entity type
    type_to_examples = defaultdict(list)
    
    for tokens, tags in zip(train_tokens, train_tags):
        # Get entity types in this sentence
        entity_types = set()
        for tag in tags:
            if tag.startswith('B-') or tag.startswith('I-'):
                entity_types.add(tag[2:])
        
        # Add to each entity type
        for etype in entity_types:
            type_to_examples[etype].append((tokens, tags))
    
    # Build index per type
    retrievers = {}
    for etype, examples in type_to_examples.items():
        if len(examples) < 5:  # Skip types with too few examples
            continue
        
        print(f"\nBuilding index for entity type: {etype} ({len(examples)} examples)")
        
        tokens_list = [ex[0] for ex in examples]
        tags_list = [ex[1] for ex in examples]
        
        retriever = KNNRetriever()
        retriever.build_index(tokens_list, tags_list, model_name=model_name)
        retrievers[etype] = retriever
    
    return retrievers