import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from AMG_with_KG import AMGKGSystem, MedicalEntity, MedicalRelation, normalize_name


def iter_text_chunks(directory: str, chunk_words: int, overlap_words: int) -> Iterable[Dict[str, Any]]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Textbook directory not found: {directory}")

    for file_path in sorted(root.glob("*.txt")):
        words = file_path.read_text(encoding="utf-8", errors="ignore").split()
        if not words:
            continue

        step = max(1, chunk_words - overlap_words)
        for chunk_id, start in enumerate(range(0, len(words), step)):
            chunk = " ".join(words[start:start + chunk_words]).strip()
            if chunk:
                yield {
                    "filename": file_path.name,
                    "chunk_id": chunk_id,
                    "text": chunk,
                }


def extract_chunk_knowledge(system: AMGKGSystem, chunk: Dict[str, Any], max_entities: int) -> Dict[str, Any]:
    prompt = f"""
Return only valid JSON.
You are building a reusable medical knowledge graph from a textbook passage.
Extract important medical entities and clinically meaningful relationships grounded in the passage.
Keep only information supported by the passage.

Schema:
{{
  "entities": [
    {{"name":"...", "entity_type":"disease|drug|symptom|mechanism|treatment|finding|anatomy|medical_concept", "description":"...", "confidence":0.0}}
  ],
  "relations": [
    {{"source":"entity name", "target":"entity name", "relation_type":"treats|causes|adverse_effect_of|mechanism_of_action|risk_factor_for|symptom_of|indicates|contraindicated_with|part_of|related_to", "confidence":0.0, "evidence":"short quote/paraphrase from passage"}}
  ]
}}

Passage source:
{chunk["filename"]} chunk {chunk["chunk_id"]}

Passage:
{chunk["text"][:5000]}
"""
    data = system.invoke_json(prompt, {"entities": [], "relations": []})
    entities = data.get("entities", [])[:max_entities] if isinstance(data, dict) else []
    entity_names = {normalize_name(item.get("name", "")) for item in entities if isinstance(item, dict)}
    relations = []
    for relation in data.get("relations", []) if isinstance(data, dict) else []:
        if not isinstance(relation, dict):
            continue
        if normalize_name(relation.get("source", "")) in entity_names and normalize_name(relation.get("target", "")) in entity_names:
            relations.append(relation)
    return {"entities": entities, "relations": relations}


def add_knowledge_to_store(system: AMGKGSystem, knowledge: Dict[str, Any], source_label: str) -> None:
    entities: List[MedicalEntity] = []
    for item in knowledge.get("entities", []):
        if not isinstance(item, dict) or not item.get("name"):
            continue
        entities.append(MedicalEntity(
            name=item["name"],
            description=item.get("description", ""),
            entity_type=item.get("entity_type", "medical_concept"),
            confidence=float(item.get("confidence", 0.6) or 0.6),
            sources=[source_label],
        ))

    valid_entity_names = {normalize_name(entity.name): entity.name for entity in entities}
    relations: List[MedicalRelation] = []
    for item in knowledge.get("relations", []):
        if not isinstance(item, dict):
            continue
        source_key = normalize_name(item.get("source", ""))
        target_key = normalize_name(item.get("target", ""))
        if source_key not in valid_entity_names or target_key not in valid_entity_names:
            continue
        relations.append(MedicalRelation(
            source=valid_entity_names[source_key],
            target=valid_entity_names[target_key],
            relation_type=item.get("relation_type", "related_to"),
            confidence=float(item.get("confidence", 0.5) or 0.5),
            evidence=item.get("evidence", ""),
            sources=[source_label],
        ))

    system.update_global_mkg(entities, relations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline/background builder for the global Medical Knowledge Graph.")
    parser.add_argument("--textbook-dir", default="data_clean/data_clean/textbooks/en")
    parser.add_argument("--provider", choices=["openai", "openai-compatible", "ollama"], default="ollama")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--mkg-path", default="artifacts/global_mkg.json")
    parser.add_argument("--chunk-words", type=int, default=512)
    parser.add_argument("--overlap-words", type=int, default=100)
    parser.add_argument("--max-chunks", type=int, default=10, help="Use -1 for all chunks.")
    parser.add_argument("--max-entities-per-chunk", type=int, default=8)
    parser.add_argument("--no-vector-db", action="store_true", help="Do not initialize Chroma retrieval while building.")
    args = parser.parse_args()

    os.makedirs(Path(args.mkg_path).parent, exist_ok=True)
    system = AMGKGSystem(
        provider=args.provider,
        model=args.model,
        mkg_path=args.mkg_path,
        use_pubmed=False,
        use_wikipedia=False,
        use_vector_db=not args.no_vector_db,
    )

    processed = 0
    for chunk in iter_text_chunks(args.textbook_dir, args.chunk_words, args.overlap_words):
        if args.max_chunks != -1 and processed >= args.max_chunks:
            break

        source_label = f"textbook:{chunk['filename']}:chunk:{chunk['chunk_id']}"
        print(f"Building MKG from {source_label}")
        try:
            knowledge = extract_chunk_knowledge(system, chunk, args.max_entities_per_chunk)
            add_knowledge_to_store(system, knowledge, source_label)
            processed += 1
        except Exception as exc:
            print(f"Failed {source_label}: {exc}")

    print(f"Processed chunks: {processed}")
    print(f"Global MKG saved to: {args.mkg_path}")
    print(f"Nodes: {len(system.store.entities)}")
    print(f"Edges: {len(system.store.relations)}")


if __name__ == "__main__":
    main()
