"""
Stage 2: LLM-Based Reranking

Uses Gemini Flash 1.5 to score candidates on clinical relevance,
filtering from 50 candidates to top 10.

Reference: Manuscript Section 2.3
"""

from typing import List, Dict, Tuple
import google.generativeai as genai
from .config import (
    RERANK_MODEL,
    FLASH_INPUT_COST, FLASH_OUTPUT_COST,
    TOP_K_RERANK
)
from .prompts import RERANK_PROMPT


class Stage2Reranking:
    """
    Refines Stage 1 candidates using Gemini Flash for semantic relevance assessment.
    
    Manuscript Section 2.3.2:
    - Scores document relevance on scale of 0-10
    - 10 = directly answers query with high-quality, recent clinical data
    - 5 = relevant but indirect or outdated information  
    - 0 = irrelevant
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(RERANK_MODEL)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = TOP_K_RERANK) -> Tuple[List[Dict], float]:
        """
        Rerank candidates using LLM-based relevance scoring.
        
        Args:
            query: User's clinical question
            candidates: Stage 1 output (up to 50 documents)
            top_k: Number of documents to return (default: 10)
            
        Returns:
            Tuple of (reranked_documents, cost_usd)
        """
        if not candidates:
            return [], 0.0

        # Prepare textual representation for the prompt
        events_text = ""
        for i, c in enumerate(candidates):
            date_str = c['timestamp'].strftime("%Y-%m-%d") if hasattr(c['timestamp'], 'strftime') else str(c['timestamp'])[:10]
            # Truncate text to avoid token limits
            text_preview = c['text'][:300] + "..." if len(c['text']) > 300 else c['text']
            events_text += f"[{i+1}] {date_str} | {c['type']}: {text_preview}\n"

        prompt = RERANK_PROMPT.format(
            query=query,
            top_k=top_k,
            events_text=events_text,
            num_candidates=len(candidates)
        )

        response = self.model.generate_content(prompt)

        # Parse indices from response
        valid_indices = self._parse_indices(response.text, len(candidates), top_k)

        # Calculate cost
        input_tokens = len(prompt.split()) * 1.3  # Rough token estimate
        output_tokens = len(response.text.split()) * 1.3
        cost = (input_tokens / 1000 * FLASH_INPUT_COST) + (output_tokens / 1000 * FLASH_OUTPUT_COST)

        # Build reranked list
        reranked = [candidates[i] for i in valid_indices]
        return reranked, cost

    def _parse_indices(self, response_text: str, num_candidates: int, top_k: int) -> List[int]:
        """
        Parse document indices from LLM response.
        Expects format: "5,2,8,1,12,..."
        """
        try:
            indices_str = response_text.strip()
            
            # Handle potential markdown or preamble
            if "Format:" in indices_str:
                indices_str = indices_str.split("Format:")[-1]
            
            # Clean up common formatting issues
            indices_str = indices_str.replace("[", "").replace("]", "")
            indices_str = indices_str.replace("\n", ",")
            
            # Parse comma-separated integers (1-indexed in prompt, convert to 0-indexed)
            indices = []
            for x in indices_str.split(","):
                x = x.strip()
                if x.isdigit():
                    idx = int(x) - 1  # Convert to 0-indexed
                    if 0 <= idx < num_candidates:
                        indices.append(idx)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            
            return unique_indices[:top_k]
            
        except Exception as e:
            print(f"Reranking parse error: {e}")
            # Fallback to Stage 1 order
            return list(range(min(top_k, num_candidates)))
