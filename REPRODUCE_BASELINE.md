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
