"""
Stage 1: Multi-Modal Retrieval with Temporal Weighting

Implements the FUNNEL broad search strategy combining:
- Vector similarity (Gemini embedding-001)
- Keyword search (BM25)
- Temporal weighting (document-type-specific decay)
- Document priority multipliers

Reference: Manuscript Section 2.2
"""

import math
from datetime import datetime
from typing import List, Dict
import numpy as np
import google.generativeai as genai
from .config import (
    EMBEDDING_MODEL,
    WEIGHT_SEMANTIC, WEIGHT_KEYWORD, WEIGHT_PRIORITY,
    PRIORITY_MULTIPLIERS, TEMPORAL_LAMBDA, TEMPORAL_CATEGORIES,
    TOP_K_STAGE1
)


class Stage1Retrieval:
    """
    Implements broad retrieval using the FUNNEL scoring formula:
    final_score = (w1*semantic + w2*keyword + w3*priority) * temporal_score
    
    Where:
        w1 = 0.5 (WEIGHT_SEMANTIC)
        w2 = 0.3 (WEIGHT_KEYWORD)
        w3 = 0.2 (WEIGHT_PRIORITY)
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model_name = f"models/{EMBEDDING_MODEL}"

    def get_embedding(self, text: str) -> List[float]:
        """Generate 768-dimensional embedding using Gemini embedding-001."""
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]

    def compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Standard cosine similarity.
        Manuscript Section 2.2.1: similarity(q, d) = (q · d) / (||q|| ||d||)
        """
        a, b = np.array(vec1), np.array(vec2)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute_bm25(self, query: str, document: str, corpus: List[str]) -> float:
        """
        BM25 scoring function.
        Manuscript Section 2.2.2 with k1=1.5, b=0.75
        
        score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1+1)) / (f(qi,D) + k1·(1-b+b·|D|/avgdl))
        """
        q_terms = query.lower().split()
        d_terms = document.lower().split()
        
        # Parameters (manuscript values)
        k1, b = 1.5, 0.75
        avg_dl = sum(len(d.split()) for d in corpus) / len(corpus) if corpus else 1
        
        score = 0.0
        doc_len = len(d_terms)
        
        for term in q_terms:
            # Term Frequency in document
            tf = d_terms.count(term)
            if tf == 0:
                continue
                
            # Inverse Document Frequency
            doc_count = sum(1 for d in corpus if term in d.lower())
            idf = math.log((len(corpus) - doc_count + 0.5) / (doc_count + 0.5) + 1.0)
            
            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
            score += idf * (numerator / denominator)
            
        return score

    def get_temporal_score(self, timestamp: datetime, doc_type: str) -> float:
        """
        Calculate temporal decay score based on document type.
        Manuscript Section 2.2.3:
        
        - No decay (1.0) for history/allergies
        - exp(-λ * age_days) for vitals/labs  
        - 0.5 + 0.5*exp(-λ * age_days) for notes/reports
        
        Where λ = 0.002 (calibrated for ~50% weight at 1 year)
        """
        days_old = (datetime.now() - timestamp).days
        if days_old < 0:
            days_old = 0
        
        # Determine decay category
        if doc_type in TEMPORAL_CATEGORIES["no_decay"]:
            return 1.0
        elif doc_type in TEMPORAL_CATEGORIES["full_decay"]:
            return math.exp(-TEMPORAL_LAMBDA * days_old)
        else:
            # Moderate decay with 0.5 floor (default for notes/reports)
            return 0.5 + 0.5 * math.exp(-TEMPORAL_LAMBDA * days_old)

    def get_priority_multiplier(self, doc_type: str) -> float:
        """
        Priority multipliers by clinical importance.
        Manuscript Table 1:
            Discharge Summary: 1.5
            Procedure Notes: 1.3
            Imaging Reports: 1.2
            Progress Notes: 1.0
            Nursing Notes: 0.8
        """
        return PRIORITY_MULTIPLIERS.get(doc_type, 1.0)

    def retrieve(self, query: str, documents: List[Dict], top_k: int = TOP_K_STAGE1) -> List[Dict]:
        """
        Main Stage 1 entry point.
        
        Args:
            query: User's clinical question
            documents: List of dicts with keys: 'id', 'text', 'type', 'timestamp', 'embedding'
            top_k: Number of candidates to return (default: 50)
            
        Returns:
            Top-k documents sorted by final_score (descending)
        """
        query_vec = self.get_embedding(query)
        corpus_texts = [d['text'] for d in documents]
        
        scored_results = []
        for doc in documents:
            # 1. Semantic Score (Vector Similarity)
            semantic_score = self.compute_cosine_similarity(query_vec, doc['embedding'])
            
            # 2. Keyword Score (BM25), normalized to [0,1]
            keyword_score = self.compute_bm25(query, doc['text'], corpus_texts)
            keyword_score = min(1.0, keyword_score / 10.0)
            
            # 3. Priority Score (Document Type Multiplier)
            priority_score = self.get_priority_multiplier(doc['type'])
            # Normalize to [0,1] range (max multiplier is 1.5)
            priority_score_normalized = priority_score / 1.5
            
            # 4. Temporal Weight
            temporal_score = self.get_temporal_score(doc['timestamp'], doc['type'])
            
            # Unified Ranking Formula (Manuscript Section 2.2.5)
            base_relevance = (
                WEIGHT_SEMANTIC * semantic_score +
                WEIGHT_KEYWORD * keyword_score +
                WEIGHT_PRIORITY * priority_score_normalized
            )
            final_score = base_relevance * temporal_score
            
            doc_copy = doc.copy()
            doc_copy['final_score'] = final_score
            doc_copy['_debug'] = {
                'semantic': semantic_score,
                'keyword': keyword_score,
                'priority': priority_score,
                'temporal': temporal_score
            }
            scored_results.append(doc_copy)
        
        # Sort by final score (descending)
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)
        return scored_results[:top_k]
