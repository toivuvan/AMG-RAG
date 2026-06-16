# Pipeline Hệ Thống AMG-RAG Với Medical Knowledge Graph

Tài liệu này mô tả luồng hoạt động của hệ thống AMG-RAG khi nhận một câu hỏi y tế, ví dụ từ MEDQA. Pipeline bám theo tinh thần của paper: textbook được lưu trong Vector Database, còn Medical Knowledge Graph (MKG) được xây dựng/cập nhật động dựa trên câu hỏi, textbook context và PubMed/Wikipedia evidence.

## 1. Chuẩn Bị Offline

### 1.1. Tạo Vector Database Từ Textbook

Lệnh chạy:

```bash
python create_VDB.py --reset
```

Luồng xử lý:

```text
Textbook .txt
-> chia chunk 512 từ, overlap 100 từ
-> embedding bằng all-mpnet-base-v2
-> lưu vào ChromaDB tại new_VDB/
```

Vai trò:

- Textbook chỉ đóng vai trò retrieval corpus.
- Hệ thống không build repository graph offline từ toàn bộ textbook.
- Khi có câu hỏi, hệ thống truy xuất top-k textbook chunks liên quan từ ChromaDB.

### 1.2. MKG Ban Đầu

MKG có thể bắt đầu rỗng hoặc đã có một số tri thức được mồi từ các tác vụ PubMed/Wikipedia background.

Trong pipeline chính:

```text
artifacts/global_mkg.json
```

là nơi lưu các node/edge đã được tạo trong quá trình hỏi đáp hoặc background update.

Điểm quan trọng:

- Textbook không được quét offline để dựng graph lớn.
- Textbook chỉ nằm trong Vector DB.
- MKG được xây/cập nhật khi có context liên quan.

## 2. Khi Nhận Câu Hỏi MEDQA

Ví dụ input:

```json
{
  "question": "A 67-year-old man with bladder cancer develops tinnitus and sensorineural hearing loss after chemotherapy. The beneficial effect of the drug is most likely due to which mechanism?",
  "options": {
    "A": "Inhibition of thymidine synthesis",
    "B": "Inhibition of proteasome",
    "C": "Hyperstabilization of microtubules",
    "D": "Generation of free radicals",
    "E": "Cross-linking of DNA"
  },
  "answer": "Cross-linking of DNA",
  "answer_idx": "E"
}
```

Khi chạy benchmark:

```bash
python run_medqa_kg.py --provider ollama --model llama3.1:8b --limit 20 --output results/medqa_kg_20.csv
```

hệ thống đọc câu hỏi từ:

```text
data_clean/data_clean/questions/US/test.jsonl
```

## 3. Truy Xuất Textbook Context

Hệ thống query ChromaDB bằng:

```text
question
option A
option B
option C
option D
option E
```

Kết quả là các đoạn textbook liên quan, ví dụ:

```text
Cisplatin is a platinum-based chemotherapeutic agent.
Platinum compounds form DNA cross-links.
Cisplatin can cause ototoxicity.
```

Context này được dùng ở các bước:

- trích xuất entity,
- suy luận relation,
- sinh reasoning,
- sinh đáp án cuối.

## 4. Trích Xuất Medical Terms Bằng MER

Ngay sau khi nhận câu hỏi, hệ thống dùng một LLM agent đóng vai trò **MER - Medical Entity Recognizer** để trích xuất các medical terms/entities cốt lõi:

```text
q -> MER -> n1, n2, ..., nm
```

MER nhận:

```text
question
options
```

Ví dụ medical terms:

```text
bladder cancer
chemotherapy
cisplatin
tinnitus
sensorineural hearing loss
DNA cross-linking
```

Các medical terms này vừa là foundational nodes của MKG, vừa là truy vấn đầu vào cho PubMed/Wikipedia.

## 5. Tạo Search Context Từ PubMed/Wikipedia

Với từng medical term `ni`, hệ thống truy xuất:

```text
1. PubMed esearch -> lấy PMID
2. PubMed efetch -> lấy abstract
3. Nếu PubMed không có kết quả hợp lệ, fallback sang Wikipedia search/summary
```

Tương ứng với công thức trong paper:

```text
d(ni) = Search(ni)
```

Ví dụ evidence:

```json
[
  {
    "source": "PubMed",
    "id": "PMID...",
    "content": "Cisplatin is associated with ototoxicity..."
  },
  {
    "source": "Wikipedia",
    "id": "Cisplatin",
    "content": "Cisplatin is a platinum-based chemotherapy medication..."
  }
]
```

Có thể tắt PubMed nếu môi trường không có network. Khi đó hệ thống sẽ dùng Wikipedia nếu `--no-wikipedia` không được bật:

```bash
python run_medqa_kg.py --provider ollama --model llama3.1:8b --limit 20 --no-pubmed
```

## 6. Enrich Seed Entities Và Trích Xuất Retrieved Entities

Sau khi có medical terms từ MER và context từ PubMed/Wikipedia, hệ thống retrieve textbook chunks từ ChromaDB rồi xử lý hai nhóm entity:

### 6.1. Enrich Seed Entities

Seed entities là các entity được MER trích trực tiếp từ câu hỏi/options.

LLM nhận:

```text
seed entities
textbook context
PubMed/Wikipedia context
```

và cập nhật:

```text
description
confidence
```

Không thêm entity mới ở bước enrich.

### 6.2. Extract Retrieved Entities

Retrieved entities là các entity mới được LLM phát hiện sau khi đọc context từ textbook và PubMed/Wikipedia.

Ràng buộc:

- Không lặp lại seed entities.
- Phải liên quan trực tiếp đến ít nhất một seed entity.
- Confidence `>= 0.8`.
- Giới hạn tối đa, mặc định `3` retrieved entities / câu hỏi.
- Tránh entity quá chung như `patient`, `disease`, `treatment`, `study`, `mechanism`.

Ví dụ:

```json
{
  "retrieved_entities": [
    {
      "name": "cisplatin",
      "entity_type": "drug",
      "description": "Platinum chemotherapy associated with ototoxicity and DNA cross-linking.",
      "confidence": 0.92,
      "linked_seed": "bladder cancer"
    }
  ]
}
```

Sau đó hệ thống merge:

```text
seed entities + retrieved entities
```

để infer relations và xây MKG.

Ví dụ:

```json
{
  "entities": [
    {
      "name": "bladder cancer",
      "entity_type": "disease",
      "description": "Cancer treated with chemotherapy",
      "confidence": 0.8
    },
    {
      "name": "cisplatin",
      "entity_type": "drug",
      "description": "Platinum chemotherapy associated with ototoxicity",
      "confidence": 0.9
    },
    {
      "name": "sensorineural hearing loss",
      "entity_type": "symptom",
      "description": "Ototoxic adverse effect",
      "confidence": 0.9
    },
    {
      "name": "DNA cross-linking",
      "entity_type": "mechanism",
      "description": "Mechanism of platinum chemotherapy",
      "confidence": 0.9
    }
  ]
}
```

## 7. Truy Vấn MKG Hiện Có

Hệ thống tìm các entity vừa trích xuất trong:

```text
artifacts/global_mkg.json
```

Nếu graph đã có tri thức liên quan, hệ thống lấy subgraph:

```text
cisplatin -> mechanism_of_action -> DNA cross-linking
cisplatin -> adverse_effect_of -> sensorineural hearing loss
```

Nếu chưa có, subgraph có thể rỗng và hệ thống sẽ cập nhật động.

## 8. Dynamic Construction/Update MKG

Đây là bước chính của AMG-RAG.

LLM nhận:

```text
question
options
textbook context
PubMed/Wikipedia context
subgraph hiện có nếu có
```

Sau đó hệ thống tạo/cập nhật:

- nodes,
- relations,
- confidence score,
- relation summary,
- evidence,
- source metadata.

Ví dụ relations:

```json
{
  "relations": [
    {
      "source": "cisplatin",
      "target": "bladder cancer",
      "relation_type": "treats",
      "confidence": 0.8,
      "evidence": "Cisplatin is used in chemotherapy for bladder cancer"
    },
    {
      "source": "cisplatin",
      "target": "sensorineural hearing loss",
      "relation_type": "adverse_effect_of",
      "confidence": 0.9,
      "evidence": "Cisplatin can cause ototoxicity"
    },
    {
      "source": "cisplatin",
      "target": "DNA cross-linking",
      "relation_type": "mechanism_of_action",
      "confidence": 0.9,
      "evidence": "Platinum compounds form DNA cross-links",
      "summary": "Cisplatin exerts its antitumor effect by forming DNA cross-links."
    }
  ]
}
```

Các ràng buộc khi ghi vào MKG:

- Chỉ giữ relation có confidence `>= 0.8`, tương ứng threshold `8/10` trong paper.
- Không lưu mọi relation LLM sinh ra để giảm hallucination và nhiễu từ retrieval.
- Mỗi edge phải có `summary` ngắn để LLM hiểu bối cảnh khi graph được truy vấn lại.
- Mỗi relation được thêm theo hướng gốc và hỗ trợ duyệt ngược để reasoning hai chiều.
- Retrieved entity chỉ được lưu nếu nó có ít nhất một relation đạt ngưỡng với seed entity.

Các node/edge mới được ghi vào:

```text
artifacts/global_mkg.json
```

Nhờ vậy, những câu hỏi sau có thể tái sử dụng tri thức đã được tạo trước đó.

## 9. Lấy Final Graph Context Bằng Confidence Propagation

Sau khi cập nhật MKG, hệ thống truy vấn lại graph:

```text
entities trong câu hỏi
-> bắt đầu từ node gốc với score = 1.0
-> duyệt graph tối đa 1-hop hoặc 2-hop
-> score node con = score node cha * confidence(edge)
-> dừng nhánh nếu score lũy kế < threshold
```

Thuật toán này bám theo công thức trong paper:

```text
s(nj) = s(ni) * s(rij)
```

Trong đó:

- `s(ni)` là score lũy kế của node cha,
- `s(rij)` là confidence của relation,
- `s(nj)` là score của node con.

Nếu `s(nj) < 0.8`, hệ thống không đi sâu tiếp vào nhánh đó.

Ví dụ final graph context:

```json
{
  "nodes": [
    {
      "id": "cisplatin",
      "type": "drug",
      "confidence": 0.9,
      "path_score": 1.0
    },
    {
      "id": "DNA cross-linking",
      "type": "mechanism",
      "confidence": 0.9,
      "path_score": 0.9
    }
  ],
  "edges": [
    {
      "source": "cisplatin",
      "target": "DNA cross-linking",
      "relation": "mechanism_of_action",
      "confidence": 0.9,
      "evidence": "Platinum compounds form DNA cross-links",
      "summary": "Cisplatin exerts its antitumor effect by forming DNA cross-links."
    }
  ]
}
```

## 10. Sinh Reasoning Traces Theo Algorithm 1

Sau khi có final graph context, hệ thống không trả lời ngay. Nó sinh các reasoning traces theo từng entity/medical term.

Với mỗi entity `ni`, hệ thống thu thập các edge summaries liên quan sau khi đã duyệt graph bằng confidence propagation:

```text
ci = LLM(ni, graph edge summaries, textbook context, external evidence)
```

Ví dụ reasoning trace:

```json
{
  "entity": "cisplatin",
  "trace": "Cisplatin is relevant because the patient developed ototoxicity after chemotherapy, and graph edges connect cisplatin to sensorineural hearing loss and DNA cross-linking.",
  "graph_summaries": [
    "cisplatin --[adverse_effect_of, confidence=0.9]--> sensorineural hearing loss: Cisplatin can cause ototoxicity.",
    "cisplatin --[mechanism_of_action, confidence=0.9]--> DNA cross-linking: Cisplatin exerts its antitumor effect by forming DNA cross-links."
  ]
}
```

Các trace này tương ứng với `c_i` trong Algorithm 1 của paper.

## 11. Tổng Hợp Đáp Án Cuối

LLM nhận:

```text
question
options
final graph context
reasoning traces
retrieved papers metadata
textbook context
PubMed/Wikipedia evidence
```

và trả về:

```json
{
  "answer": "E",
  "confidence": 0.87,
  "reasoning": "The patient likely received cisplatin, which causes ototoxicity and acts by DNA cross-linking.",
  "explanation": "Cisplatin is a platinum chemotherapy agent whose antitumor mechanism is cross-linking DNA."
}
```

Ở tầng trình bày sản phẩm hoặc báo cáo, output cuối nên render thành 3 khối:

```text
Answer:
E. Cross-linking of DNA

Reasoning:
The patient likely received cisplatin because the graph connects bladder cancer chemotherapy,
ototoxicity/sensorineural hearing loss, and DNA cross-linking as the mechanism of action.

Retrieved Papers:
1. PMID: ..., Authors: ..., Title: ..., Journal: ..., Year: ...
```

Lưu ý: `Retrieved Papers` không để LLM tự bịa. Hệ thống lấy trực tiếp metadata từ PubMed XML:

```text
PMID
authors
title
journal
year
snippet
```

Trong code, hệ thống sinh thêm trường `final_response` để render giống ví dụ trong paper:

```text
Question: ...
Choices: A: ..., B: ..., C: ..., D: ..., E: ...
Answer: E (Cross-linking of DNA)
Reasoning: ...
Retrieved Papers: 1) Author et al., *Title*, Journal, Year. PMID: ...
```

## 12. Output CSV

Mỗi câu MEDQA được lưu thành một dòng CSV gồm:

```text
q_idx
question
options
expected_answer
expected_answer_text
model_answer
confidence
explanation
reasoning
final_response
reasoning_traces
entities
retrieved_entities
relations
graph_context
search_context
retrieved_papers
medical_terms
graph_stats
documents
```

Ví dụ:

```text
q_idx: 1
expected_answer: E
model_answer: E
confidence: 0.87
explanation: Cisplatin acts by cross-linking DNA.
graph_stats: {"global_nodes":120,"global_edges":240,"context_nodes":4,"context_edges":3,"dynamic_update":true}
```

## 13. Evaluation

Lệnh chạy:

```bash
python evaluate_results.py --input results/medqa_kg_20.csv --report
```

Script so sánh:

```text
expected_answer
vs
model_answer
```

và tính:

- accuracy,
- macro-F1,
- classification report.

## 14. Sơ Đồ Tóm Tắt

```text
OFFLINE

Textbooks
   |
   v
create_VDB.py
   |
   v
ChromaDB / new_VDB/


OPTIONAL BACKGROUND UPDATE

PubMed/Wikipedia search items
   |
   v
LLM extract nodes/edges
   |
   v
MKG / artifacts/global_mkg.json


ONLINE / QUERY-TIME

MEDQA question
   |
   v
Retrieve textbook chunks from ChromaDB
   |
   v
MER extracts medical terms
   |
   v
Search PubMed/Wikipedia with medical terms
   |
   v
Enrich seed entities and extract controlled retrieved entities
   |
   v
Query existing MKG
   |
   v
Dynamically construct/update MKG
   |
   v
Retrieve final graph context
   |
   v
Generate per-entity reasoning traces
   |
   v
Synthesize final answer from traces + graph + evidence
   |
   v
Final answer + confidence + reasoning + retrieved papers
   |
   v
CSV output + evaluation
```

## 15. Kết Luận

Pipeline này hiện thực hóa AMG-RAG theo hướng:

```text
textbook vector retrieval
+
PubMed/Wikipedia evidence retrieval
+
dynamic Medical Knowledge Graph construction/update
+
graph-based reasoning
```

Điểm quan trọng là textbook không được dùng để tạo repository graph offline. Textbook chỉ được lưu trong Vector DB và được truy xuất khi có câu hỏi. MKG được xây/cập nhật động dựa trên context lấy được từ textbook và PubMed/Wikipedia.
