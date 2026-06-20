import argparse
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import networkx as nx
import requests
import wikipedia
from decouple import config
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from create_VDB import MedicalQAChromaDB


@dataclass
class MedicalEntity:
    name: str
    description: str = ""
    entity_type: str = "medical_concept"
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)


@dataclass
class MedicalRelation:
    source: str
    target: str
    relation_type: str = "related_to"
    confidence: float = 0.5
    evidence: str = ""
    summary: str = ""
    sources: List[str] = field(default_factory=list)


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def extract_json_object(text: str, fallback: Any) -> Any:
    if not text:
        return fallback

    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not match:
        return fallback

    try:
        return json.loads(match.group(1))
    except Exception:
        return fallback


def llm_text(response: Any) -> str:
    if isinstance(response, AIMessage):
        return response.content or ""
    return getattr(response, "content", str(response))


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "which", "following",
    "patient", "patients", "history", "current", "most", "likely", "shows",
    "show", "after", "before", "during", "because", "about", "into", "over",
    "under", "than", "then", "been", "have", "has", "had", "his", "her",
    "she", "him", "man", "woman", "year", "old", "day", "week", "month",
    "blood", "pressure", "temperature", "pulse", "respirations", "normal",
}


def keyword_tokens(text: str) -> set:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", str(text).lower())
    return {token for token in tokens if token not in STOPWORDS}


class HFTransformersChat:
    def __init__(self, model: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model
        self.max_new_tokens = int(config("HF_MAX_NEW_TOKENS", default="768"))
        self.max_input_tokens = int(config("HF_MAX_INPUT_TOKENS", default="4096"))
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        load_kwargs = {
            "device_map": config("HF_DEVICE_MAP", default="auto"),
            "trust_remote_code": True,
        }
        if config("HF_LOAD_IN_4BIT", default="false").lower() in {"1", "true", "yes"}:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        self.model.eval()

    def invoke(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class LLMFactory:
    @staticmethod
    def create(provider: str, model: str):
        if provider == "ollama":
            return ChatOllama(model=model, temperature=0.0, format="json")
        if provider == "openai-compatible":
            return ChatOpenAI(
                model=model,
                temperature=0.0,
                api_key=config("OPENAI_API_KEY"),
                base_url=config("OPENAI_BASE_URL"),
            )
        if provider == "openrouter":
            return ChatOpenAI(
                model=model,
                temperature=0.0,
                api_key=config("OPENROUTER_API_KEY"),
                base_url=config("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1"),
                default_headers={
                    "HTTP-Referer": config("OPENROUTER_SITE_URL", default="http://localhost"),
                    "X-Title": config("OPENROUTER_APP_NAME", default="AMG-RAG"),
                },
            )
        if provider == "gemini":
            return ChatOpenAI(
                model=model,
                temperature=0.0,
                api_key=config("GEMINI_API_KEY"),
                base_url=config("GEMINI_BASE_URL", default="https://generativelanguage.googleapis.com/v1beta/openai/"),
            )
        if provider == "openai":
            return ChatOpenAI(model=model, temperature=0.0, api_key=config("OPENAI_API_KEY"))
        if provider == "hf-transformers":
            return HFTransformersChat(model)
        raise ValueError("provider must be one of: gemini, hf-transformers, openai, openai-compatible, openrouter, ollama")


class PubMedSearcher:
    def __init__(self, api_key: str = "", tool: str = "AMG_RAG_project", email: str = ""):
        self.api_key = api_key
        self.tool = tool
        self.email = email
        self.last_request_time = 0.0
        self.headers = {
            "User-Agent": f"{self.tool}/1.0 ({self.email or 'no-email-provided'})"
        }

    def _throttle(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.34:
            time.sleep(0.34 - elapsed)
        self.last_request_time = time.time()

    def search(self, query: str, max_results: int = 2) -> List[Dict[str, str]]:
        self._throttle()
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "tool": self.tool,
        }
        if self.api_key:
            search_params["api_key"] = self.api_key
        if self.email:
            search_params["email"] = self.email

        try:
            search_response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=search_params,
                headers=self.headers,
                timeout=15,
            )
            search_response.raise_for_status()
            content_type = search_response.headers.get("content-type", "")
            if "json" not in content_type.lower():
                snippet = " ".join(search_response.text[:300].split())
                raise ValueError(f"NCBI esearch returned non-JSON response: {snippet}")

            search_data = search_response.json()
            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []

            self._throttle()
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "tool": self.tool,
            }
            if self.api_key:
                fetch_params["api_key"] = self.api_key
            if self.email:
                fetch_params["email"] = self.email

            fetch_response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=fetch_params,
                headers=self.headers,
                timeout=20,
            )
            fetch_response.raise_for_status()
            if "<PubmedArticle" not in fetch_response.text and "<PubmedBookArticle" not in fetch_response.text:
                snippet = " ".join(fetch_response.text[:300].split())
                raise ValueError(f"NCBI efetch returned non-PubMed XML response: {snippet}")
            return self._parse_pubmed_xml(fetch_response.text)
        except Exception as exc:
            return [{"source": "PubMed", "id": "", "content": f"PubMed retrieval failed: {exc}"}]

    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict[str, str]]:
        root = ET.fromstring(xml_text)
        results = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//MedlineCitation/PMID") or ""
            title = " ".join(article.findtext(".//ArticleTitle", default="").split())
            journal = article.findtext(".//Journal/Title") or article.findtext(".//ISOAbbreviation") or ""
            year = (
                article.findtext(".//JournalIssue/PubDate/Year")
                or article.findtext(".//ArticleDate/Year")
                or ""
            )
            authors = []
            for author in article.findall(".//AuthorList/Author")[:3]:
                last = author.findtext("LastName") or ""
                initials = author.findtext("Initials") or ""
                name = " ".join(part for part in [last, initials] if part).strip()
                if name:
                    authors.append(name)

            abstract_parts = []
            for abstract in article.findall(".//Abstract/AbstractText"):
                text = " ".join("".join(abstract.itertext()).split())
                if text:
                    abstract_parts.append(text)
            abstract_text = " ".join(abstract_parts)

            if not (pmid or title or abstract_text):
                continue

            results.append({
                "source": "PubMed",
                "id": pmid,
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "authors": "; ".join(authors),
                "content": abstract_text[:1800] if abstract_text else title,
            })
        return results


class GlobalMKGStore:
    def __init__(self, path: str = "artifacts/global_mkg.json"):
        self.path = Path(path)
        self.graph = nx.DiGraph()
        self.entities: Dict[str, MedicalEntity] = {}
        self.relations: List[MedicalRelation] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for node in data.get("nodes", []):
            entity = MedicalEntity(**node)
            self.add_entity(entity, save=False)
        for edge in data.get("edges", []):
            relation = MedicalRelation(**edge)
            self.add_relation(relation, save=False)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [asdict(entity) for entity in self.entities.values()],
            "edges": [asdict(relation) for relation in self.relations],
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_entity(self, entity: MedicalEntity, save: bool = True) -> None:
        key = normalize_name(entity.name)
        existing = self.entities.get(key)
        if existing:
            existing.description = entity.description or existing.description
            existing.confidence = max(existing.confidence, entity.confidence)
            existing.sources = sorted(set(existing.sources + entity.sources))
            entity = existing
        else:
            self.entities[key] = entity

        self.graph.add_node(
            key,
            name=entity.name,
            description=entity.description,
            entity_type=entity.entity_type,
            confidence=entity.confidence,
            sources=entity.sources,
        )
        if save:
            self.save()

    def add_relation(self, relation: MedicalRelation, save: bool = True, bidirectional: bool = True) -> None:
        source_key = normalize_name(relation.source)
        target_key = normalize_name(relation.target)
        if not source_key or not target_key or source_key == target_key:
            return

        for existing in self.relations:
            if (
                normalize_name(existing.source) == source_key
                and normalize_name(existing.target) == target_key
                and existing.relation_type == relation.relation_type
            ):
                existing.confidence = max(existing.confidence, relation.confidence)
                existing.evidence = relation.evidence or existing.evidence
                existing.summary = relation.summary or existing.summary
                existing.sources = sorted(set(existing.sources + relation.sources))
                relation = existing
                break
        else:
            self.relations.append(relation)

        self.graph.add_edge(
            source_key,
            target_key,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            evidence=relation.evidence,
            summary=relation.summary,
            sources=relation.sources,
        )
        if bidirectional:
            self.graph.add_edge(
                target_key,
                source_key,
                relation_type=f"reverse_{relation.relation_type}",
                confidence=relation.confidence,
                evidence=relation.evidence,
                summary=relation.summary,
                sources=relation.sources,
                reverse=True,
            )
        if save:
            self.save()

    def match_entities(self, names: List[str]) -> List[MedicalEntity]:
        matches = []
        for name in names:
            key = normalize_name(name)
            if key in self.entities:
                matches.append(self.entities[key])
        return matches

    def subgraph_context(self, names: List[str], depth: int = 1, threshold: float = 0.8) -> Dict[str, Any]:
        selected = set()
        best_scores: Dict[str, float] = {}
        frontier = []
        for name in names:
            key = normalize_name(name)
            if key in self.graph:
                selected.add(key)
                best_scores[key] = 1.0
                frontier.append((key, 1.0, 0))

        while frontier:
            node, score, current_depth = frontier.pop(0)
            if current_depth >= depth:
                continue

            for neighbor in self.graph.successors(node):
                edge_data = self.graph.get_edge_data(node, neighbor) or {}
                edge_confidence = float(edge_data.get("confidence", 0))
                accumulated_score = score * edge_confidence
                if accumulated_score < threshold:
                    continue

                if accumulated_score <= best_scores.get(neighbor, 0):
                    continue

                selected.add(neighbor)
                best_scores[neighbor] = accumulated_score
                frontier.append((neighbor, accumulated_score, current_depth + 1))

        nodes = []
        for node in selected:
            data = self.graph.nodes[node]
            nodes.append({
                "id": data.get("name", node),
                "description": data.get("description", ""),
                "type": data.get("entity_type", ""),
                "confidence": data.get("confidence", 0),
                "path_score": best_scores.get(node, 0),
            })

        edges = []
        for source, target, data in self.graph.edges(data=True):
            if source in selected and target in selected and float(data.get("confidence", 0)) >= threshold:
                edges.append({
                    "source": self.graph.nodes[source].get("name", source),
                    "target": self.graph.nodes[target].get("name", target),
                    "relation": data.get("relation_type", "related_to"),
                    "confidence": data.get("confidence", 0),
                    "evidence": data.get("evidence", ""),
                    "summary": data.get("summary", ""),
                    "reverse": data.get("reverse", False),
                })

        return {"nodes": nodes, "edges": edges, "threshold": threshold, "max_depth": depth}


class AMGKGSystem:
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1:8b",
        mkg_path: str = "artifacts/global_mkg.json",
        use_pubmed: bool = True,
        use_wikipedia: bool = True,
        use_vector_db: bool = True,
        max_entities: int = 6,
        max_retrieved_entities: int = 3,
        confidence_threshold: float = 0.8,
        use_evidence_verifier: bool = False,
        evidence_relevance_threshold: float = 0.8,
    ):
        self.llm = LLMFactory.create(provider, model)
        self.pubmed = PubMedSearcher(
            api_key=config("pubmed_api", default=""),
            tool=config("NCBI_TOOL", default="AMG_RAG_project"),
            email=config("NCBI_EMAIL", default=""),
        )
        self.store = GlobalMKGStore(mkg_path)
        self.use_pubmed = use_pubmed
        self.use_wikipedia = use_wikipedia
        self.use_vector_db = use_vector_db
        self.max_entities = max_entities
        self.max_retrieved_entities = max_retrieved_entities
        self.confidence_threshold = confidence_threshold
        self.use_evidence_verifier = use_evidence_verifier
        self.evidence_relevance_threshold = evidence_relevance_threshold
        self.db = MedicalQAChromaDB() if use_vector_db and Path("new_VDB").exists() else None
        self.llm_min_interval_seconds = float(config("LLM_MIN_INTERVAL_SECONDS", default="5"))
        self.llm_max_retries = int(config("LLM_MAX_RETRIES", default="2"))
        self.max_reasoning_trace_calls = int(config("MAX_REASONING_TRACE_CALLS", default="2"))
        self.last_llm_request_time = 0.0

    def throttle_llm(self) -> None:
        elapsed = time.time() - self.last_llm_request_time
        if elapsed < self.llm_min_interval_seconds:
            time.sleep(self.llm_min_interval_seconds - elapsed)
        self.last_llm_request_time = time.time()

    def invoke_json(self, prompt: str, fallback: Any) -> Any:
        last_error = None
        for attempt in range(self.llm_max_retries + 1):
            try:
                self.throttle_llm()
                response = self.llm.invoke(prompt)
                return extract_json_object(llm_text(response), fallback)
            except Exception as exc:
                last_error = exc
                message = str(exc)
                logging.exception("LLM JSON invocation failed")
                if "429" not in message and "quota" not in message.lower() and "rate" not in message.lower():
                    break
                if attempt < self.llm_max_retries:
                    time.sleep(65)

        if isinstance(fallback, dict):
            data = dict(fallback)
            data["_error"] = str(last_error) if last_error else "Unknown LLM error"
            return data
        return fallback

    def retrieve_textbook_context(self, question: str, options: Dict[str, str]) -> str:
        if not self.db:
            return ""
        docs = []
        try:
            docs.extend(self.db.main(mode="query", query_text=question, n_results=3))
            for option in options.values():
                docs.extend(self.db.main(mode="query", query_text=option, n_results=1))
        except Exception:
            return ""
        return "\n\n".join(item[0].page_content for item in docs if item and item[0])[:5000]

    def evidence_relevance_score(self, item: Dict[str, str], term: str, question: str, options: Dict[str, str]) -> int:
        item_text = " ".join([
            item.get("title", ""),
            item.get("content", ""),
            item.get("journal", ""),
        ])
        item_tokens = keyword_tokens(item_text)
        term_tokens = keyword_tokens(term)
        question_tokens = keyword_tokens(question)
        option_tokens = keyword_tokens(" ".join(options.values()) if isinstance(options, dict) else "")
        return (
            3 * len(item_tokens & term_tokens)
            + 2 * len(item_tokens & question_tokens)
            + len(item_tokens & option_tokens)
        )

    def generate_search_phrases(
        self,
        question: str,
        options: Dict[str, str],
        seed_entities: List[MedicalEntity],
    ) -> List[str]:
        entity_names = [entity.name for entity in seed_entities]
        prompt = f"""
Return only valid JSON.
Generate up to 4 precise PubMed search phrases for this MEDQA question.
The phrases should combine the core disease/drug/symptom/mechanism terms instead of using isolated entities.
Prefer clinically specific phrases such as "cisplatin ototoxicity sensorineural hearing loss" over broad terms such as "cisplatin".
Use answer options only to disambiguate the clinical concept; do not create a phrase from one answer option alone.
Avoid generic patient words, ages, and irrelevant vitals.

Schema:
{{"search_phrases":["phrase 1","phrase 2"]}}

Question:
{question}

Options:
{json.dumps(options, ensure_ascii=False)}

Seed entities:
{json.dumps(entity_names, ensure_ascii=False)}
"""
        data = self.invoke_json(prompt, {"search_phrases": []})
        phrases = []
        seen = set()
        for phrase in data.get("search_phrases", []) if isinstance(data, dict) else []:
            phrase = re.sub(r"\s+", " ", str(phrase).strip())
            key = normalize_name(phrase)
            if not key or key in seen:
                continue
            if len(keyword_tokens(phrase)) < 2:
                continue
            seen.add(key)
            phrases.append(phrase)
            if len(phrases) >= 4:
                break

        if not phrases:
            phrases = entity_names[: min(self.max_entities, 4)]
        return phrases

    def retrieve_external_context(
        self,
        terms: List[str],
        question: str = "",
        options: Optional[Dict[str, str]] = None,
        max_total: int = 4,
    ) -> List[Dict[str, str]]:
        options = options or {}
        evidence = []
        seen = set()
        for term in terms[: min(self.max_entities, 4)]:
            pubmed_results = []
            if self.use_pubmed:
                candidates = [
                    item for item in self.pubmed.search(term, max_results=3)
                    if item.get("content") and not item.get("content", "").startswith("PubMed retrieval failed:")
                ]
                scored = [
                    (self.evidence_relevance_score(item, term, question, options), item)
                    for item in candidates
                ]
                pubmed_results = [
                    item for score, item in sorted(scored, key=lambda pair: pair[0], reverse=True)
                    if score >= 3
                ][:1]
                for item in pubmed_results:
                    item = dict(item)
                    item["query"] = term
                    key = item.get("pmid") or item.get("id") or item.get("title")
                    if key and key not in seen:
                        seen.add(key)
                        evidence.append(item)

            if len(evidence) >= max_total:
                break

            if not pubmed_results and self.use_wikipedia and len(evidence) < max_total:
                try:
                    result = wikipedia.search(term, results=1)
                    if result:
                        wiki_item = {
                            "source": "Wikipedia",
                            "id": result[0],
                            "query": term,
                            "content": wikipedia.summary(result[0], sentences=2)[:1200],
                        }
                        key = f"wiki:{result[0]}"
                        if key not in seen and self.evidence_relevance_score(wiki_item, term, question, options) >= 3:
                            seen.add(key)
                            evidence.append(wiki_item)
                except Exception:
                    pass
        if self.use_evidence_verifier and evidence:
            return self.verify_evidence_relevance(question, options, terms, evidence)
        return evidence

    def verify_evidence_relevance(
        self,
        question: str,
        options: Dict[str, str],
        terms: List[str],
        evidence: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        evidence_payload = []
        for index, item in enumerate(evidence):
            evidence_payload.append({
                "evidence_id": index,
                "source": item.get("source", ""),
                "id": item.get("pmid") or item.get("id", ""),
                "title": item.get("title", ""),
                "journal": item.get("journal", ""),
                "year": item.get("year", ""),
                "content": item.get("content", "")[:1200],
            })

        prompt = f"""
Return only valid JSON.
You are a strict medical evidence verifier for a MEDQA question.
Score each evidence item for whether it is directly useful for answering the question.
Use relevance_score from 0.0 to 1.0.
Keep only evidence that directly discusses the patient's key condition, mechanism, diagnosis, treatment, adverse effect, or calculation needed by the question.
Down-rank broad, tangential, unrelated, outdated, or search-term-only matches.
Do not reward a paper just because it shares one keyword with the question.

Schema:
{{"evidence_scores":[{{"evidence_id":0, "relevance_score":0.0, "keep":false, "reason":"brief reason"}}]}}

Question:
{question}

Options:
{json.dumps(options, ensure_ascii=False)}

Seed medical terms:
{json.dumps(terms, ensure_ascii=False)}

Evidence candidates:
{json.dumps(evidence_payload, ensure_ascii=False)}
"""
        data = self.invoke_json(prompt, {"evidence_scores": []})
        if isinstance(data, dict) and data.get("_error"):
            return evidence

        scores = {}
        for item in data.get("evidence_scores", []):
            if not isinstance(item, dict):
                continue
            try:
                evidence_id = int(item.get("evidence_id"))
            except (TypeError, ValueError):
                continue
            keep_value = item.get("keep", False)
            if isinstance(keep_value, str):
                keep = keep_value.strip().lower() in {"true", "yes", "1", "keep"}
            else:
                keep = bool(keep_value)
            scores[evidence_id] = {
                "relevance_score": float(item.get("relevance_score", 0.0) or 0.0),
                "keep": keep,
                "reason": item.get("reason", ""),
            }

        verified = []
        for index, item in enumerate(evidence):
            score = scores.get(index)
            if not score:
                continue
            if score["keep"] and score["relevance_score"] >= self.evidence_relevance_threshold:
                enriched = dict(item)
                enriched["relevance_score"] = score["relevance_score"]
                enriched["relevance_reason"] = score["reason"]
                verified.append(enriched)
        return verified

    def format_retrieved_papers(self, evidence: List[Dict[str, str]]) -> List[Dict[str, str]]:
        papers = []
        seen = set()
        for item in evidence:
            if item.get("source") != "PubMed":
                continue
            pmid = item.get("pmid") or item.get("id") or ""
            key = pmid or item.get("title", "")
            if not key or key in seen:
                continue
            seen.add(key)
            papers.append({
                "pmid": pmid,
                "authors": item.get("authors", ""),
                "title": item.get("title", ""),
                "journal": item.get("journal", ""),
                "year": item.get("year", ""),
                "snippet": item.get("content", "")[:500],
            })
        return papers

    def format_paper_reference(self, paper: Dict[str, str]) -> str:
        authors = paper.get("authors", "").strip()
        first_author = authors.split(";")[0].strip() if authors else ""
        author_text = f"{first_author} et al." if first_author else "Unknown authors"
        title = paper.get("title", "").strip() or "Untitled"
        journal = paper.get("journal", "").strip() or "Unknown journal"
        year = paper.get("year", "").strip() or "n.d."
        pmid = paper.get("pmid", "").strip()
        pmid_text = f" PMID: {pmid}." if pmid else ""
        return f"{author_text}, *{title}*, {journal}, {year}.{pmid_text}"

    def format_final_response(
        self,
        question: str,
        options: Dict[str, str],
        answer: str,
        reasoning: str,
        retrieved_papers: List[Dict[str, str]],
    ) -> str:
        choices = ", ".join(f"{key}: {value}" for key, value in options.items())
        answer_text = options.get(answer, "") if isinstance(options, dict) else ""
        answer_line = f"{answer} ({answer_text})" if answer_text else answer

        if retrieved_papers:
            paper_lines = [
                f"{idx}) {self.format_paper_reference(paper)}"
                for idx, paper in enumerate(retrieved_papers, start=1)
            ]
            papers_text = " ".join(paper_lines)
        else:
            papers_text = "No PubMed papers were retrieved."

        return (
            f"Question: {question}\n"
            f"Choices: {choices}\n"
            f"Answer: {answer_line}\n"
            f"Reasoning: {reasoning}\n"
            f"Retrieved Papers: {papers_text}"
        )

    def extract_entities(
        self,
        question: str,
        options: Dict[str, str],
        textbook_context: str = "",
        evidence: Optional[List[Dict[str, str]]] = None,
    ) -> List[MedicalEntity]:
        evidence_text = "\n\n".join(
            f"[{item['source']} {item.get('id','')}] {item['content']}" for item in (evidence or [])
        )[:3000]
        prompt = f"""
Return only valid JSON.
Act as a Medical Entity Recognizer (MER). Extract the most important seed medical entities from the question stem.
Use options only for disambiguation. Do not include answer choices as seed entities unless the same concept is explicitly mentioned in the question stem.
Avoid incidental background conditions, demographics, normal vitals, and broad symptoms unless they are central to the diagnosis or management decision.
Prefer specific diseases, drugs, mechanisms, procedures, pathogens, anatomy, adverse effects, and decisive clinical findings.
Return at most {self.max_entities} entities, ordered by importance for answering the question.
Use this schema:
{{"entities":[{{"name":"...", "entity_type":"disease|drug|symptom|mechanism|treatment|finding|medical_concept", "description":"...", "confidence":0.0}}]}}

Question:
{question}

Options:
{json.dumps(options, ensure_ascii=False)}

Optional textbook context:
{textbook_context[:3000]}

Optional PubMed/Wikipedia context:
{evidence_text}
"""
        data = self.invoke_json(prompt, {"entities": []})
        entities = []
        for item in data.get("entities", [])[: self.max_entities]:
            if isinstance(item, dict) and item.get("name"):
                entities.append(MedicalEntity(
                    name=item["name"],
                    description=item.get("description", ""),
                    entity_type=item.get("entity_type", "medical_concept"),
                    confidence=float(item.get("confidence", 0.7)),
                    sources=["MER seed extraction"],
                ))

        if not entities:
            for option in list(options.values())[: self.max_entities]:
                entities.append(MedicalEntity(name=option, description=option, confidence=0.5, sources=["options"]))
        return entities

    def extract_retrieved_entities(
        self,
        seed_entities: List[MedicalEntity],
        textbook_context: str,
        evidence: List[Dict[str, str]],
    ) -> List[MedicalEntity]:
        evidence_text = "\n\n".join(
            f"[{item['source']} {item.get('id','')}] {item['content']}" for item in evidence
        )[:5000]
        seed_names = [entity.name for entity in seed_entities]
        prompt = f"""
Return only valid JSON.
Extract additional retrieved medical entities from the provided textbook and PubMed/Wikipedia context.
Do not repeat seed entities. Only include entities directly relevant to at least one seed entity.
Avoid generic terms such as patient, disease, treatment, symptom, study, mechanism.
Only include entities with confidence at least 0.8.
Schema:
{{"entities":[{{"name":"...", "entity_type":"disease|drug|symptom|mechanism|treatment|finding|anatomy|adverse_effect|medical_concept", "description":"...", "confidence":0.0, "linked_seed":"seed entity name"}}]}}

Seed entities:
{json.dumps(seed_names, ensure_ascii=False)}

Textbook context:
{textbook_context[:2500]}

PubMed/Wikipedia context:
{evidence_text}
"""
        data = self.invoke_json(prompt, {"entities": []})
        seed_keys = {normalize_name(name) for name in seed_names}
        retrieved = []
        seen = set(seed_keys)
        for item in data.get("entities", []) if isinstance(data, dict) else []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"]).strip()
            key = normalize_name(name)
            confidence = float(item.get("confidence", 0.0) or 0.0)
            if not key or key in seen or confidence < self.confidence_threshold:
                continue
            seen.add(key)
            retrieved.append(MedicalEntity(
                name=name,
                description=item.get("description", ""),
                entity_type=item.get("entity_type", "medical_concept"),
                confidence=confidence,
                sources=[f"retrieved entity linked to {item.get('linked_seed', '')}".strip()],
            ))
            if len(retrieved) >= self.max_retrieved_entities:
                break
        return retrieved

    def merge_entities(
        self,
        seed_entities: List[MedicalEntity],
        retrieved_entities: List[MedicalEntity],
    ) -> List[MedicalEntity]:
        merged: Dict[str, MedicalEntity] = {}
        for entity in seed_entities + retrieved_entities:
            key = normalize_name(entity.name)
            existing = merged.get(key)
            if existing:
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.description = entity.description or existing.description
                existing.sources = sorted(set(existing.sources + entity.sources))
            else:
                merged[key] = entity
        return list(merged.values())

    def enrich_entities(
        self,
        entities: List[MedicalEntity],
        textbook_context: str,
        evidence: List[Dict[str, str]],
    ) -> List[MedicalEntity]:
        if not entities:
            return entities

        evidence_text = "\n\n".join(
            f"[{item['source']} {item.get('id','')}] {item['content']}" for item in evidence
        )[:3500]
        prompt = f"""
Return only valid JSON.
Improve the descriptions and confidence scores of these medical entities using the textbook and PubMed/Wikipedia context.
Do not add new entities. Preserve entity names.
Schema:
{{"entities":[{{"name":"...", "description":"...", "confidence":0.0}}]}}

Entities:
{json.dumps([asdict(entity) for entity in entities], ensure_ascii=False)}

Textbook context:
{textbook_context[:2500]}

PubMed/Wikipedia context:
{evidence_text}
"""
        data = self.invoke_json(prompt, {"entities": []})
        updates = {
            normalize_name(item.get("name", "")): item
            for item in data.get("entities", []) if isinstance(item, dict)
        } if isinstance(data, dict) else {}

        enriched = []
        for entity in entities:
            update = updates.get(normalize_name(entity.name), {})
            enriched.append(MedicalEntity(
                name=entity.name,
                description=update.get("description", entity.description),
                entity_type=entity.entity_type,
                confidence=max(entity.confidence, float(update.get("confidence", entity.confidence) or entity.confidence)),
                sources=sorted(set(entity.sources + ["context enrichment"])),
            ))
        return enriched

    def infer_relations(
        self,
        question: str,
        options: Dict[str, str],
        entities: List[MedicalEntity],
        evidence: List[Dict[str, str]],
        textbook_context: str,
    ) -> List[MedicalRelation]:
        evidence_text = "\n\n".join(f"[{item['source']} {item.get('id','')}] {item['content']}" for item in evidence)[:5000]
        entity_payload = [asdict(entity) for entity in entities]
        prompt = f"""
Return only valid JSON.
Infer clinically meaningful relationships between the medical entities.
Use concise evidence. Confidence is 0.0 to 1.0. Only include relationships you would rate at least 0.8.
Use relation types such as treats, causes, adverse_effect_of, mechanism_of_action, risk_factor_for, symptom_of, indicates, contraindicated_with, differential_diagnosis, related_to.
Schema:
{{"relations":[{{"source":"entity name", "target":"entity name", "relation_type":"...", "confidence":0.0, "evidence":"...", "summary":"short contextual summary for graph traversal"}}]}}

Question:
{question}

Options:
{json.dumps(options, ensure_ascii=False)}

Entities:
{json.dumps(entity_payload, ensure_ascii=False)}

Textbook context:
{textbook_context[:2500]}

External evidence:
{evidence_text}
"""
        data = self.invoke_json(prompt, {"relations": []})
        relations = []
        entity_names = {normalize_name(entity.name): entity.name for entity in entities}
        for item in data.get("relations", []):
            if not isinstance(item, dict):
                continue
            source = item.get("source", "")
            target = item.get("target", "")
            if normalize_name(source) not in entity_names or normalize_name(target) not in entity_names:
                continue
            confidence = float(item.get("confidence", 0.5))
            if confidence < self.confidence_threshold:
                continue
            relations.append(MedicalRelation(
                source=entity_names[normalize_name(source)],
                target=entity_names[normalize_name(target)],
                relation_type=item.get("relation_type", "related_to"),
                confidence=confidence,
                evidence=item.get("evidence", ""),
                summary=item.get("summary", item.get("evidence", "")),
                sources=["LLM relation inference"],
            ))
        return relations

    def update_global_mkg(
        self,
        entities: List[MedicalEntity],
        relations: List[MedicalRelation],
        seed_entity_names: Optional[List[str]] = None,
    ) -> None:
        seed_keys = {normalize_name(name) for name in (seed_entity_names or [])}
        connected_keys = set(seed_keys)
        for relation in relations:
            if relation.confidence < self.confidence_threshold:
                continue
            source_key = normalize_name(relation.source)
            target_key = normalize_name(relation.target)
            if source_key in seed_keys or target_key in seed_keys:
                connected_keys.add(source_key)
                connected_keys.add(target_key)

        for entity in entities:
            entity_key = normalize_name(entity.name)
            if seed_keys and entity_key not in connected_keys:
                continue
            self.store.add_entity(entity, save=False)
        for relation in relations:
            if relation.confidence < self.confidence_threshold:
                continue
            if seed_keys and (
                normalize_name(relation.source) not in connected_keys
                or normalize_name(relation.target) not in connected_keys
            ):
                continue
            self.store.add_relation(relation, save=False)
        self.store.save()

    def answer_question(self, question_data: Dict[str, Any], update_store: bool = True) -> Dict[str, Any]:
        question = question_data["question"]
        options = question_data.get("options", {})
        textbook_context = self.retrieve_textbook_context(question, options)
        seed_entities = self.extract_entities(question, options)
        seed_entity_names = [entity.name for entity in seed_entities]
        search_phrases = self.generate_search_phrases(question, options, seed_entities) if seed_entities else []
        evidence = self.retrieve_external_context(search_phrases, question=question, options=options) if search_phrases else []
        retrieved_papers = self.format_retrieved_papers(evidence)
        seed_entities = self.enrich_entities(seed_entities, textbook_context, evidence)
        retrieved_entities = self.extract_retrieved_entities(seed_entities, textbook_context, evidence)
        entities = self.merge_entities(seed_entities, retrieved_entities)
        entity_names = [entity.name for entity in entities]

        preexisting = self.store.subgraph_context(seed_entity_names, depth=1, threshold=self.confidence_threshold)
        missing_terms = [name for name in seed_entity_names if not self.store.match_entities([name])]
        should_dynamic_update = bool(missing_terms) or len(preexisting["edges"]) == 0

        relations: List[MedicalRelation] = []
        if should_dynamic_update:
            relations = self.infer_relations(question, options, entities, evidence, textbook_context)
            if update_store:
                self.update_global_mkg(entities, relations, seed_entity_names=seed_entity_names)

        graph_context = self.store.subgraph_context(entity_names, depth=2, threshold=self.confidence_threshold)
        reasoning_traces = self.generate_reasoning_traces(
            question,
            options,
            entities,
            graph_context,
            textbook_context,
            evidence,
        )
        answer = self.generate_answer(
            question,
            options,
            textbook_context,
            evidence,
            graph_context,
            reasoning_traces=reasoning_traces,
            retrieved_papers=retrieved_papers,
        )
        final_response = self.format_final_response(
            question=question,
            options=options,
            answer=answer.get("answer", "NAN"),
            reasoning=answer.get("reasoning", "") or answer.get("explanation", ""),
            retrieved_papers=retrieved_papers,
        )
        return {
            "question": question,
            "options": options,
            "expected_answer": question_data.get("answer", ""),
            "answer_idx": question_data.get("answer_idx", ""),
            "answer": answer.get("answer", "NAN"),
            "confidence": answer.get("confidence", 0.0),
            "explanation": answer.get("explanation", "") or answer.get("_error", ""),
            "reasoning": answer.get("reasoning", ""),
            "final_response": final_response,
            "reasoning_traces": reasoning_traces,
            "entities": [asdict(entity) for entity in entities],
            "relations": [asdict(relation) for relation in relations],
            "graph_context": graph_context,
            "search_context": evidence,
            "retrieved_papers": retrieved_papers,
            "medical_terms": seed_entity_names,
            "search_phrases": search_phrases,
            "retrieved_entities": [asdict(entity) for entity in retrieved_entities],
            "documents": textbook_context,
            "graph_stats": {
                "global_nodes": len(self.store.entities),
                "global_edges": len(self.store.relations),
                "context_nodes": len(graph_context["nodes"]),
                "context_edges": len(graph_context["edges"]),
                "dynamic_update": should_dynamic_update,
            },
        }

    def generate_answer(
        self,
        question: str,
        options: Dict[str, str],
        textbook_context: str,
        evidence: List[Dict[str, str]],
        graph_context: Dict[str, Any],
        reasoning_traces: Optional[List[Dict[str, str]]] = None,
        retrieved_papers: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        evidence_text = "\n\n".join(f"[{item['source']} {item.get('id','')}] {item['content']}" for item in evidence)[:3500]
        traces_text = json.dumps(reasoning_traces or [], ensure_ascii=False)[:5000]
        papers_text = json.dumps(retrieved_papers or [], ensure_ascii=False)[:3000]
        prompt = f"""
Return only valid JSON.
Synthesize the final answer to the medical multiple-choice question using the reasoning traces, graph context, and retrieved evidence.
If options are present, answer with one of A, B, C, D, or E.
Do not invent citations. Use only the retrieved papers listed below.
Schema:
{{"answer":"A|B|C|D|E|NAN", "confidence":0.0, "reasoning":"brief reasoning", "explanation":"brief explanation"}}

Question:
{question}

Options:
{json.dumps(options, ensure_ascii=False)}

Knowledge graph context:
{json.dumps(graph_context, ensure_ascii=False)[:5000]}

Reasoning traces:
{traces_text}

Retrieved papers:
{papers_text}

Textbook context:
{textbook_context[:3000]}

External evidence:
{evidence_text}
"""
        data = self.invoke_json(prompt, {"answer": "NAN", "confidence": 0.0, "reasoning": "", "explanation": ""})
        answer = str(data.get("answer", "NAN")).strip().upper()
        match = re.search(r"\b([A-E])\b", answer)
        if not match:
            combined = " ".join([
                str(data.get("reasoning", "")),
                str(data.get("explanation", "")),
            ])
            match = re.search(r"(?:answer|option|choice)\s*[:\-]?\s*\(?([A-E])\)?", combined, flags=re.IGNORECASE)
        data["answer"] = match.group(1) if match else "NAN"
        data["confidence"] = float(data.get("confidence", 0.0) or 0.0)
        return data

    def collect_entity_graph_summaries(self, entity_name: str, graph_context: Dict[str, Any]) -> List[str]:
        summaries = []
        normalized = normalize_name(entity_name)
        for edge in graph_context.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            if normalized not in {normalize_name(source), normalize_name(target)}:
                continue
            summary = edge.get("summary") or edge.get("evidence") or ""
            if summary:
                summaries.append(
                    f"{source} --[{edge.get('relation', 'related_to')}, confidence={edge.get('confidence', 0)}]--> {target}: {summary}"
                )
        return summaries

    def generate_reasoning_traces(
        self,
        question: str,
        options: Dict[str, str],
        entities: List[MedicalEntity],
        graph_context: Dict[str, Any],
        textbook_context: str,
        evidence: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        evidence_text = "\n\n".join(f"[{item['source']} {item.get('id','')}] {item['content']}" for item in evidence)[:2500]
        traces = []
        trace_calls = 0
        for entity in entities:
            if trace_calls >= self.max_reasoning_trace_calls:
                break
            summaries = self.collect_entity_graph_summaries(entity.name, graph_context)
            if not summaries:
                continue

            prompt = f"""
Return only valid JSON.
Generate a concise reasoning trace for how this medical entity helps answer the question.
Use the graph edge summaries first. Use textbook/external evidence only as supporting context.
Schema:
{{"trace":"brief reasoning trace"}}

Question:
{question}

Options:
{json.dumps(options, ensure_ascii=False)}

Focus entity:
{entity.name}

Graph edge summaries:
{json.dumps(summaries, ensure_ascii=False)[:3000]}

Textbook context:
{textbook_context[:1500]}

External evidence:
{evidence_text}
"""
            data = self.invoke_json(prompt, {"trace": ""})
            trace = str(data.get("trace", "")).strip() if isinstance(data, dict) else ""
            if trace:
                trace_calls += 1
                traces.append({
                    "entity": entity.name,
                    "trace": trace,
                    "graph_summaries": summaries,
                })
        return traces


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AMG KG-RAG question from a MEDQA JSONL file.")
    parser.add_argument("--input", default="data_clean/data_clean/questions/US/test.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--provider", choices=["gemini", "hf-transformers", "openai", "openai-compatible", "openrouter", "ollama"], default="ollama")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--mkg-path", default="artifacts/global_mkg.json")
    parser.add_argument("--no-pubmed", action="store_true")
    parser.add_argument("--no-wikipedia", action="store_true")
    parser.add_argument("--no-vector-db", action="store_true")
    parser.add_argument("--verify-evidence", action="store_true")
    parser.add_argument("--evidence-relevance-threshold", type=float, default=0.8)
    args = parser.parse_args()

    questions = load_jsonl(args.input)
    system = AMGKGSystem(
        provider=args.provider,
        model=args.model,
        mkg_path=args.mkg_path,
        use_pubmed=not args.no_pubmed,
        use_wikipedia=not args.no_wikipedia,
        use_vector_db=not args.no_vector_db,
        use_evidence_verifier=args.verify_evidence,
        evidence_relevance_threshold=args.evidence_relevance_threshold,
    )
    result = system.answer_question(questions[args.index])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
