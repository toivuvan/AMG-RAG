# AMG-RAG Baseline Reproduction

This repo is prepared to run a practical end-to-end AMG-RAG baseline on MEDQA-style data.

## 1. Install dependencies

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add environment variables

Create `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
pubmed_api=optional_pubmed_api_key
```

For an OpenAI-compatible hosted API, use:

```env
OPENAI_API_KEY=your_provider_key
OPENAI_BASE_URL=https://your-provider-openai-compatible-endpoint/v1
pubmed_api=optional_pubmed_api_key
```

For local Ollama, `.env` is only needed for `pubmed_api`; the LLM runs locally.

## 3. Add data

Expected paths:

```text
data_clean/data_clean/questions/US/test.jsonl
data_clean/data_clean/textbooks/en/*.txt
```

The MEDQA JSONL rows should include `question`, `options`, `answer`, and `answer_idx`.

## 4. Build the vector database

```bash
python create_VDB.py
```

This creates `new_VDB/` using `all-mpnet-base-v2`, 512-word chunks, and 100-word overlap.

## 5. Run a small MEDQA baseline

Start with a small subset to check cost and stability:

```bash
python run_medqa.py --limit 20 --output results/medqa_baseline_20.csv
```

Run with local Ollama:

```bash
ollama pull llama3.1:8b
python run_medqa.py --provider ollama --model llama3.1:8b --limit 20 --output results/medqa_ollama_20.csv
```

Run with an OpenAI-compatible hosted API:

```bash
python run_medqa.py --provider openai-compatible --model your-model-name --limit 20 --output results/medqa_open_20.csv
```

Run all questions:

```bash
python run_medqa.py --limit -1 --output results/medqa_baseline_full.csv
```

## 6. Evaluate

```bash
python evaluate_results.py --input results/medqa_baseline_20.csv --report
```

For the class project, this is enough for Phase 1: an end-to-end baseline evaluated on a public medical QA benchmark.

## 7. Run the paper-inspired KG-RAG pipeline

The original `AMG-with-KG.py` builds a temporary per-question graph. This repo now also includes
`AMG_with_KG.py`, which implements a practical version of the paper idea:

- a persistent global MKG store at `artifacts/global_mkg.json`
- query-time subgraph retrieval from the global MKG
- dynamic PubMed/Wikipedia updates when the current graph lacks relevant entities/edges
- textbook retrieval from Chroma when `new_VDB/` exists

Run a small KG-RAG batch with local LLaMA 3.1:

```bash
python run_medqa_kg.py --provider ollama --model llama3.1:8b --limit 5 --output results/medqa_kg_5.csv
```

Evaluate:

```bash
python evaluate_results.py --input results/medqa_kg_5.csv --report
```

Run without PubMed if network latency is too high:

```bash
python run_medqa_kg.py --provider ollama --model llama3.1:8b --limit 5 --no-pubmed --output results/medqa_kg_no_pubmed_5.csv
```

Continue in batches:

```bash
python run_medqa_kg.py --provider ollama --model llama3.1:8b --start 0 --limit 50 --output results/medqa_kg.csv
python run_medqa_kg.py --provider ollama --model llama3.1:8b --start 50 --limit 50 --output results/medqa_kg.csv
```

## 8. Paper-aligned flow: background MKG + dynamic update

The paper describes two MKG update modes:

1. background/pre-populated MKG construction
2. dynamic query-time update when the graph lacks relevant knowledge

This repo implements that practical flow with `artifacts/global_mkg.json`.

Build a small background MKG from textbook chunks:

```bash
python build_global_mkg.py --provider ollama --model llama3.1:8b --max-chunks 10 --no-vector-db
```

Build more chunks after the small run works:

```bash
python build_global_mkg.py --provider ollama --model llama3.1:8b --max-chunks 100 --no-vector-db
```

Then run dynamic KG-RAG. The runner will reuse `artifacts/global_mkg.json` and add new entities/relations when needed:

```bash
python run_medqa_kg.py --provider ollama --model llama3.1:8b --limit 20 --output results/medqa_kg_20.csv
```

Evaluate:

```bash
python evaluate_results.py --input results/medqa_kg_20.csv --report
```

Optional Neo4j import:

```bash
python neo4j_mkg_store.py --mkg-path artifacts/global_mkg.json --uri bolt://localhost:7687 --user neo4j --password your_password --clear
```

Set these in `.env` if you do not want to pass them each time:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```
