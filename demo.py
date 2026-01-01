"""
FUNNEL End-to-End Demonstration Script

Reproduction code for MethodsX submission.
Demonstrates the complete three-stage pipeline on mock clinical data.

Usage:
    python -m repro.demo
    
Requires:
    GOOGLE_API_KEY environment variable set
"""

import os
from datetime import datetime
from .stage1_retrieval import Stage1Retrieval
from .stage2_reranking import Stage2Reranking
from .stage3_synthesis import Stage3Synthesis


def create_mock_data():
    """
    Create mock clinical events for demonstration.
    In practice, these would be loaded from your EHR export (FHIR, CSV, etc.)
    
    Note: Embeddings here are placeholders. In production, generate real
    768-dimensional embeddings using Stage1Retrieval.get_embedding()
    """
    return [
        {
            "id": "lab_001",
            "type": "Lab Results",
            "timestamp": datetime(2023, 10, 15),
            "text": "Creatinine: 1.1 mg/dL. BUN: 18 mg/dL. eGFR: 72 mL/min/1.73m2. Baseline renal function stable.",
            "embedding": [0.1] * 768  # Placeholder - use real embeddings
        },
        {
            "id": "soap_001",
            "type": "SOAP Notes",
            "timestamp": datetime(2023, 11, 20),
            "text": "Patient presents with mild fatigue. Denies chest pain, shortness of breath. Renal function stable per recent labs. Continue current medications.",
            "embedding": [0.12] * 768
        },
        {
            "id": "lab_002",
            "type": "Lab Results",
            "timestamp": datetime(2023, 12, 5),
            "text": "Creatinine: 1.2 mg/dL (slight increase from baseline). Potassium: 4.5 mEq/L. Normal range.",
            "embedding": [0.11] * 768
        },
        {
            "id": "imaging_001",
            "type": "Imaging Reports",
            "timestamp": datetime(2023, 12, 10),
            "text": "Renal ultrasound: Bilateral kidneys normal in size. No hydronephrosis. No masses or stones identified.",
            "embedding": [0.08] * 768
        },
        {
            "id": "discharge_001",
            "type": "Discharge Summary",
            "timestamp": datetime(2024, 1, 5),
            "text": "Admitted for CHF exacerbation. Creatinine peaked at 1.8 mg/dL during admission (AKI on CKD). Resolved to 1.3 mg/dL at discharge. Baseline creatinine ~1.1 mg/dL.",
            "embedding": [0.15] * 768
        }
    ]


def run_demo():
    """Execute the full FUNNEL pipeline demonstration."""
    
    # 1. Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("=" * 60)
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        print("Please set it with: export GOOGLE_API_KEY='your-key-here'")
        print("=" * 60)
        return

    # 2. Load mock data
    print("=" * 60)
    print("FUNNEL Pipeline Demonstration")
    print("=" * 60)
    
    mock_events = create_mock_data()
    print(f"\nLoaded {len(mock_events)} clinical events")
    
    # 3. Initialize pipeline stages
    s1 = Stage1Retrieval(api_key)
    s2 = Stage2Reranking(api_key)
    s3 = Stage3Synthesis(api_key)

    # 4. Define query
    query = "What is the patient's baseline creatinine and current renal function status?"
    print(f"\nQUERY: {query}")
    print("-" * 60)

    # 5. Stage 1: Broad Retrieval
    print("\n[Stage 1] Executing Multi-Modal Retrieval...")
    s1_results = s1.retrieve(query, mock_events, top_k=5)
    print(f"  → Retrieved {len(s1_results)} candidates")
    for i, doc in enumerate(s1_results[:3]):
        print(f"     {i+1}. [{doc['type']}] score={doc['final_score']:.3f}")

    # 6. Stage 2: LLM Reranking
    print("\n[Stage 2] Executing Gemini Flash Reranking...")
    s2_results, s2_cost = s2.rerank(query, s1_results, top_k=3)
    print(f"  → Reranked to top {len(s2_results)} documents")
    print(f"  → Stage 2 cost: ${s2_cost:.4f}")

    # 7. Stage 3: Conflict Detection
    print("\n[Stage 3] Detecting Conflicts...")
    conflicts = s3.detect_conflicts(s2_results)
    if conflicts:
        print(f"  → Found {len(conflicts)} potential conflict(s):")
        for c in conflicts:
            print(f"     ⚠ {c['event_1']} vs {c['event_2']}: {c['reason']}")
    else:
        print("  → No conflicts detected")

    # 8. Stage 3: Cited Synthesis
    print("\n[Stage 3] Generating Cited Synthesis...")
    answer, citations, s3_cost = s3.synthesize(query, s2_results)
    print(f"  → Stage 3 cost: ${s3_cost:.4f}")

    # 9. Output Results
    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(f"\nAnswer:\n{answer}")
    print(f"\nCitations Validated: {len(citations)}")
    for cit in citations:
        print(f"  - [{cit['id']}] {cit['type']}: {cit['snippet'][:50]}...")
    
    print(f"\nConflicts Found: {len(conflicts)}")
    print(f"Total Pipeline Cost: ${s2_cost + s3_cost:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
