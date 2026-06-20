import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from AMG_with_KG import AMGKGSystem, load_jsonl


def serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def result_to_row(result: Dict[str, Any], q_idx: int) -> Dict[str, Any]:
    return {
        "q_idx": q_idx,
        "question": result.get("question", ""),
        "options": serialize(result.get("options", {})),
        "expected_answer": result.get("answer_idx", result.get("expected_answer", "")),
        "expected_answer_text": result.get("expected_answer", ""),
        "model_answer": result.get("answer", "NAN"),
        "confidence": result.get("confidence", 0.0),
        "explanation": result.get("explanation", ""),
        "reasoning": result.get("reasoning", ""),
        "final_response": result.get("final_response", ""),
        "reasoning_traces": serialize(result.get("reasoning_traces", [])),
        "entities": serialize(result.get("entities", [])),
        "retrieved_entities": serialize(result.get("retrieved_entities", [])),
        "relations": serialize(result.get("relations", [])),
        "graph_context": serialize(result.get("graph_context", {})),
        "search_context": serialize(result.get("search_context", [])),
        "retrieved_papers": serialize(result.get("retrieved_papers", [])),
        "medical_terms": serialize(result.get("medical_terms", [])),
        "search_phrases": serialize(result.get("search_phrases", [])),
        "graph_stats": serialize(result.get("graph_stats", {})),
        "documents": result.get("documents", ""),
    }


def load_processed_ids(output_path: str) -> set:
    if not Path(output_path).exists():
        return set()
    df = pd.read_csv(output_path)
    if "q_idx" not in df.columns:
        return set()
    return set(int(value) for value in df["q_idx"].dropna().tolist())


def append_rows(rows: List[Dict[str, Any]], output_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    if Path(output_path).exists():
        existing = pd.read_csv(output_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-inspired AMG KG-RAG on MEDQA-style JSONL data.")
    parser.add_argument("--input", default="data_clean/data_clean/questions/US/test.jsonl")
    parser.add_argument("--output", default="results/medqa_kg.csv")
    parser.add_argument("--provider", choices=["gemini", "openai", "openai-compatible", "openrouter", "ollama"], default="ollama")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--mkg-path", default="artifacts/global_mkg.json")
    parser.add_argument("--limit", type=int, default=5, help="Use -1 for all questions.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--max-entities", type=int, default=6)
    parser.add_argument("--max-retrieved-entities", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--verify-evidence", action="store_true")
    parser.add_argument("--evidence-relevance-threshold", type=float, default=0.8)
    parser.add_argument("--no-pubmed", action="store_true")
    parser.add_argument("--no-wikipedia", action="store_true")
    parser.add_argument("--no-vector-db", action="store_true")
    parser.add_argument("--no-update-store", action="store_true")
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input}")

    questions = load_jsonl(args.input)
    end = len(questions) if args.limit == -1 else min(len(questions), args.start + args.limit)
    selected = list(enumerate(questions[args.start:end], start=args.start))
    processed = load_processed_ids(args.output)

    system = AMGKGSystem(
        provider=args.provider,
        model=args.model,
        mkg_path=args.mkg_path,
        use_pubmed=not args.no_pubmed,
        use_wikipedia=not args.no_wikipedia,
        use_vector_db=not args.no_vector_db,
        max_entities=args.max_entities,
        max_retrieved_entities=args.max_retrieved_entities,
        confidence_threshold=args.confidence_threshold,
        use_evidence_verifier=args.verify_evidence,
        evidence_relevance_threshold=args.evidence_relevance_threshold,
    )

    rows = []
    for q_idx, question_data in selected:
        if q_idx in processed:
            print(f"Skipping Q#{q_idx}: already in {args.output}")
            continue

        print(f"Processing Q#{q_idx}")
        try:
            result = system.answer_question(question_data, update_store=not args.no_update_store)
            rows.append(result_to_row(result, q_idx))
        except Exception as exc:
            rows.append({
                "q_idx": q_idx,
                "question": question_data.get("question", ""),
                "options": serialize(question_data.get("options", {})),
                "expected_answer": question_data.get("answer_idx", ""),
                "expected_answer_text": question_data.get("answer", ""),
                "model_answer": "NAN",
                "confidence": 0.0,
                "explanation": f"Pipeline failed: {exc}",
                "reasoning": "",
                "final_response": "",
                "reasoning_traces": "[]",
                "entities": "[]",
                "retrieved_entities": "[]",
                "relations": "[]",
                "graph_context": "{}",
                "search_context": "[]",
                "retrieved_papers": "[]",
                "medical_terms": "[]",
                "search_phrases": "[]",
                "graph_stats": "{}",
                "documents": "",
            })

        if len(rows) >= args.flush_every:
            append_rows(rows, args.output)
            rows = []

    append_rows(rows, args.output)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
