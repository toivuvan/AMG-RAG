import argparse
import os
from pathlib import Path

from Simple_AMG_RAG import QAChainProcessor


def validate_paths(input_path: str, require_vdb: bool, provider: str) -> None:
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Expected a MEDQA-style JSONL file, for example: "
            "dataset/MEDQA/questions/US/test.jsonl"
        )

    if require_vdb and not Path("new_VDB").exists():
        raise FileNotFoundError(
            "Vector database directory not found: new_VDB\n"
            "Create it first with: python create_VDB.py"
        )

    if provider in {"openai", "openai-compatible"} and not Path(".env").exists():
        raise FileNotFoundError(
            ".env not found. Create it with at least:\n"
            "OPENAI_API_KEY=your_key\n"
            "pubmed_api=optional_pubmed_key\n"
            "For openai-compatible providers, also add OPENAI_BASE_URL=..."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AMG-RAG baseline on a MEDQA-style JSONL file."
    )
    parser.add_argument(
        "--input",
        default="data_clean/data_clean/questions/US/test.jsonl",
        help="Path to MEDQA-style JSONL input.",
    )
    parser.add_argument(
        "--output",
        default="results/medqa_baseline.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of questions to run. Use -1 for all questions.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start offset in the input JSONL file.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Chat model name.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "openai-compatible", "ollama"],
        default="openai",
        help="LLM provider backend.",
    )
    parser.add_argument(
        "--skip-vdb-check",
        action="store_true",
        help="Skip checking that new_VDB exists.",
    )
    args = parser.parse_args()

    limit = None if args.limit == -1 else args.limit
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    validate_paths(args.input, require_vdb=not args.skip_vdb_check, provider=args.provider)

    processor = QAChainProcessor(model_name=args.model, provider=args.provider)
    processor.main(args.input, args.output, limit=limit, start=args.start)


if __name__ == "__main__":
    main()
