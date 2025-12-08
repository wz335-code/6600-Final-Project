from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class CosineEvaluator:
    """Calculate cosine similarity between texts using pre-trained models"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading Sentence Transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        # Generate text embeddings
        embeddings = self.model.encode([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    def compute_similarities_batch(self, 
                                   original_texts: List[str], 
                                   rewritten_texts: List[str],
                                   show_progress: bool = True):
        assert len(original_texts) == len(rewritten_texts), \
            "Number of original texts and rewritten texts must be equal"
        
        similarities = []
        
        iterator = zip(original_texts, rewritten_texts)
        if show_progress:
            iterator = tqdm(iterator, total=len(original_texts), 
                          desc="Computing cosine similarity")
        
        for orig, rewritten in iterator:
            # Handle empty text cases
            if not orig or not rewritten or pd.isna(orig) or pd.isna(rewritten):
                similarities.append(np.nan)
            else:
                sim = self.compute_similarity(str(orig), str(rewritten))
                similarities.append(sim)
        
        return similarities