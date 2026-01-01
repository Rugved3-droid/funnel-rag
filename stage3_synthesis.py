"""
Stage 3: Evidence-Grounded Synthesis and Conflict Detection

Uses Gemini Pro 1.5 to generate clinical answers with:
- Mandatory inline citations
- Citation validation
- Conflict detection (cosine > 0.85 with negation signals)

Reference: Manuscript Section 2.4
"""

import re
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import google.generativeai as genai
from .config import (
    SYNTHESIS_MODEL,
    PRO_INPUT_COST, PRO_OUTPUT_COST,
    CONFLICT_SIMILARITY_THRESHOLD
)
from .prompts import SYNTHESIS_PROMPT


class Stage3Synthesis:
    """
    Final stage of the FUNNEL pipeline.
    
    Manuscript Section 2.4.1 - Synthesis requirements:
    1. Every factual claim must be cited with document ID
    2. If information is not in evidence, respond "Not found in chart"
    3. If documents conflict, state "Conflicting information found" and cite both
    4. Do not infer, extrapolate, or generate information not explicitly stated
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(SYNTHESIS_MODEL)

    def synthesize(self, query: str, events: List[Dict]) -> Tuple[str, List[Dict], float]:
        """
        Generate answer with mandatory citations.
        
        Args:
            query: User's clinical question
            events: Top 10 documents from Stage 2
            
        Returns:
            Tuple of (answer_text, validated_citations, cost_usd)
        """
        # Format context for prompt
        context_parts = []
        for i, event in enumerate(events):
            date_str = event['timestamp'].strftime('%Y-%m-%d') if hasattr(event['timestamp'], 'strftime') else str(event['timestamp'])[:10]
            context_parts.append(
                f"[{i+1}] Event ID: {event['id']}\n"
                f"    Date: {date_str}\n"
                f"    Type: {event['type']}\n"
                f"    Content: {event['text']}\n"
            )

        context = "\n".join(context_parts)
        prompt = SYNTHESIS_PROMPT.format(context=context, query=query)

        response = self.model.generate_content(prompt)
        answer_text = response.text

        # Extract and validate citations
        citations = self._extract_citations(answer_text, events)

        # Clean answer text (remove JSON block if present)
        clean_answer = re.sub(r'```json\s*\{.*?\}\s*```', '', answer_text, flags=re.DOTALL).strip()

        # Calculate cost
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(answer_text.split()) * 1.3
        cost = (input_tokens / 1000 * PRO_INPUT_COST) + (output_tokens / 1000 * PRO_OUTPUT_COST)

        return clean_answer, citations, cost

    def _extract_citations(self, text: str, events: List[Dict]) -> List[Dict]:
        """
        Extract and validate citations from generated text.
        Manuscript Section 2.4.2: Citation Validation
        """
        # Try JSON extraction first (structured output)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                raw_citations = data.get("citations", [])

                event_map = {e['id']: e for e in events}
                validated = []
                for rc in raw_citations:
                    eid = rc.get("event_id")
                    if eid in event_map:
                        validated.append({
                            "id": eid,
                            "type": event_map[eid]['type'],
                            "snippet": rc.get("snippet", event_map[eid]['text'][:100])
                        })
                return validated
            except json.JSONDecodeError:
                pass

        # Fallback: regex for [n] notation
        indices = set(re.findall(r'\[(\d+)\]', text))
        validated = []
        for idx_str in indices:
            idx = int(idx_str) - 1  # Convert to 0-indexed
            if 0 <= idx < len(events):
                validated.append({
                    "id": events[idx]['id'],
                    "type": events[idx]['type'],
                    "snippet": events[idx]['text'][:100]
                })
        return validated

    def detect_conflicts(self, events: List[Dict]) -> List[Dict]:
        """
        Detect contradictory information in retrieved documents.
        
        Manuscript Section 2.4.3:
        - Cosine similarity > 0.85 between claims
        - Contradictory sentiment (negation detection)
        
        Args:
            events: Documents to check for conflicts
            
        Returns:
            List of detected conflicts with source IDs and reason
        """
        conflicts = []
        negation_keywords = {
            "no ", "not ", "absent", "negative", "denies", "denied",
            "none", "without", "never", "neither", "nor", "unremarkable",
            "no evidence", "ruled out"
        }

        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                # Only compare events of same or related types
                if events[i]['type'] != events[j]['type']:
                    continue

                # Check semantic similarity using embeddings
                sim = self._cosine_similarity(
                    events[i].get('embedding'),
                    events[j].get('embedding')
                )

                # Threshold from manuscript: 0.85
                if sim > CONFLICT_SIMILARITY_THRESHOLD:
                    # Check for contradictory negation signals
                    text_i = events[i]['text'].lower()
                    text_j = events[j]['text'].lower()

                    neg_i = any(k in text_i for k in negation_keywords)
                    neg_j = any(k in text_j for k in negation_keywords)

                    # Conflict if one is positive, one is negative
                    if neg_i != neg_j:
                        conflicts.append({
                            "event_1": events[i]['id'],
                            "event_2": events[j]['id'],
                            "type": "potential_contradiction",
                            "similarity": round(sim, 3),
                            "reason": "High semantic similarity with conflicting negation signals.",
                            "text_1_preview": events[i]['text'][:100],
                            "text_2_preview": events[j]['text'][:100]
                        })

        return conflicts

    def _cosine_similarity(self, v1: Optional[List[float]], v2: Optional[List[float]]) -> float:
        """Compute cosine similarity between two vectors."""
        if v1 is None or v2 is None:
            return 0.0
        a, b = np.array(v1), np.array(v2)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
