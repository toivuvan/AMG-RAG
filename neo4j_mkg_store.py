import argparse
import json
from pathlib import Path

from decouple import config
from neo4j import GraphDatabase


def relation_type_to_cypher(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).upper())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "RELATED_TO"


class Neo4jMKGImporter:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def create_constraints(self) -> None:
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT medical_entity_name IF NOT EXISTS "
                "FOR (e:MedicalEntity) REQUIRE e.name IS UNIQUE"
            )

    def clear(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n:MedicalEntity) DETACH DELETE n")

    def import_json(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        with self.driver.session() as session:
            for node in nodes:
                session.run(
                    """
                    MERGE (e:MedicalEntity {name: $name})
                    SET e.description = $description,
                        e.entity_type = $entity_type,
                        e.confidence = $confidence,
                        e.sources = $sources
                    """,
                    name=node.get("name", ""),
                    description=node.get("description", ""),
                    entity_type=node.get("entity_type", "medical_concept"),
                    confidence=float(node.get("confidence", 0.0) or 0.0),
                    sources=node.get("sources", []),
                )

            for edge in edges:
                rel_type = relation_type_to_cypher(edge.get("relation_type", "related_to"))
                query = f"""
                    MATCH (s:MedicalEntity {{name: $source}})
                    MATCH (t:MedicalEntity {{name: $target}})
                    MERGE (s)-[r:{rel_type}]->(t)
                    SET r.relation_type = $relation_type,
                        r.confidence = $confidence,
                        r.evidence = $evidence,
                        r.summary = $summary,
                        r.sources = $sources
                """
                session.run(
                    query,
                    source=edge.get("source", ""),
                    target=edge.get("target", ""),
                    relation_type=edge.get("relation_type", "related_to"),
                    confidence=float(edge.get("confidence", 0.0) or 0.0),
                    evidence=edge.get("evidence", ""),
                    summary=edge.get("summary", edge.get("evidence", "")),
                    sources=edge.get("sources", []),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import artifacts/global_mkg.json into Neo4j.")
    parser.add_argument("--mkg-path", default="artifacts/global_mkg.json")
    parser.add_argument("--uri", default=config("NEO4J_URI", default="bolt://localhost:7687"))
    parser.add_argument("--user", default=config("NEO4J_USER", default="neo4j"))
    parser.add_argument("--password", default=config("NEO4J_PASSWORD", default="password"))
    parser.add_argument("--clear", action="store_true", help="Delete existing MedicalEntity graph before import.")
    args = parser.parse_args()

    if not Path(args.mkg_path).exists():
        raise FileNotFoundError(f"MKG JSON not found: {args.mkg_path}")

    importer = Neo4jMKGImporter(args.uri, args.user, args.password)
    try:
        importer.create_constraints()
        if args.clear:
            importer.clear()
        importer.import_json(args.mkg_path)
    finally:
        importer.close()

    print(f"Imported {args.mkg_path} into Neo4j at {args.uri}")


if __name__ == "__main__":
    main()
