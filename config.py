"""
Configuration parameters for the FUNNEL pipeline.
All values match the MethodsX manuscript specifications.
"""

# =============================================================================
# Model Configuration
# =============================================================================
EMBEDDING_MODEL = "embedding-001"          # Google Gemini embedding-001 (768-dim)
RERANK_MODEL = "gemini-1.5-flash"          # Stage 2: Fast reranking
SYNTHESIS_MODEL = "gemini-1.5-pro"         # Stage 3: High-quality synthesis

# =============================================================================
# Stage 1: Retrieval Weights (Section 2.2.5 of manuscript)
# final_score = (w1*semantic + w2*keyword + w3*priority) * temporal_score
# =============================================================================
WEIGHT_SEMANTIC = 0.5   # w1: Vector similarity weight
WEIGHT_KEYWORD = 0.3    # w2: BM25 keyword weight
WEIGHT_PRIORITY = 0.2   # w3: Document type priority weight

# =============================================================================
# Stage 1: Document Type Priority Multipliers (Table 1 of manuscript)
# =============================================================================
PRIORITY_MULTIPLIERS = {
    "Discharge Summary": 1.5,
    "Procedure Notes": 1.3,
    "Imaging Reports": 1.2,
    "Progress Notes": 1.0,
    "Nursing Notes": 0.8,
    # Additional types with sensible defaults
    "SOAP Notes": 1.0,       # Equivalent to Progress Notes
    "Lab Results": 0.9,
    "Operation Note": 1.3,   # Equivalent to Procedure Notes
}

# =============================================================================
# Stage 1: Temporal Decay Settings (Section 2.2.3 of manuscript)
# decay_factor varies by document category:
# - 1.0 for history/allergies (no decay)
# - exp(-λ * age_days) for vitals/labs
# - 0.5 + 0.5*exp(-λ * age_days) for notes/reports
# where λ = 0.002 (calibrated to give ~50% weight at 1 year)
# =============================================================================
TEMPORAL_LAMBDA = 0.002  # Decay constant (manuscript: λ = 0.002)

# Document categories for temporal decay behavior
TEMPORAL_CATEGORIES = {
    # No decay (static history)
    "no_decay": ["History", "Allergies", "Surgical History", "Family History"],
    # Full exponential decay (volatile data)
    "full_decay": ["Lab Results", "Vitals", "Medications"],
    # Moderate decay with 0.5 floor (notes and reports)
    "moderate_decay": ["SOAP Notes", "Progress Notes", "Nursing Notes", 
                       "Imaging Reports", "Discharge Summary", "Procedure Notes"]
}

# =============================================================================
# Stage 1: Retrieval Counts
# =============================================================================
TOP_K_VECTOR = 30        # Top documents from vector similarity
TOP_K_BM25 = 20          # Top documents from BM25
TOP_K_STAGE1 = 50        # Final candidates passed to Stage 2

# =============================================================================
# Stage 2: Reranking
# =============================================================================
TOP_K_RERANK = 10        # Top documents passed to Stage 3

# =============================================================================
# Stage 3: Conflict Detection (Section 2.4.3 of manuscript)
# Threshold: cosine > 0.85 with contradictory negation signals
# =============================================================================
CONFLICT_SIMILARITY_THRESHOLD = 0.85

# =============================================================================
# Cost Estimation (Approximate USD per 1K tokens, as of 2024)
# =============================================================================
FLASH_INPUT_COST = 0.000075
FLASH_OUTPUT_COST = 0.0003
PRO_INPUT_COST = 0.00125
PRO_OUTPUT_COST = 0.005
