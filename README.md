# mevzuat-rag

Question answering over Turkish banking and data-protection legislation, built to be **measured** and **safe to put in front of users**: hybrid retrieval, article-level citations, personal-data masking, prompt-injection refusal and abstention, wired together as a LangGraph pipeline.

Corpus: three laws from mevzuat.gov.tr, split into 245 articles (521 chunks).

| Law | No. |
|---|---|
| Kişisel Verilerin Korunması Kanunu (KVKK) | 6698 |
| Bankacılık Kanunu | 5411 |
| Ödeme Hizmetleri ve Elektronik Para Kuruluşları Kanunu | 6493 |

## Pipeline

```
question -> guard -> retrieve -> generate -> answer + citations
              |          |
              |          +-- weak evidence -> abstain
              +-- prompt injection -> refuse
```

- **guard**: masks TCKN, IBAN, card numbers (checksum-validated), phone numbers and e-mail addresses before the text reaches retrieval or a model; refuses prompt-injection attempts.
- **retrieve**: BM25 (Turkish-aware tokenizer with prefix stemming) and dense embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, ONNX via fastembed), fused with reciprocal rank fusion. Abstains when the best dense match is below a similarity threshold.
- **generate**: any OpenAI-compatible endpoint (vLLM, Ollama, hosted API). Citations returned by the model are checked against the retrieved articles and anything else is dropped. Without an endpoint, an extractive generator returns the best-matching sentences verbatim, so the whole pipeline runs offline.

## Results

Measured with `mevzuat-rag eval` on 48 hand-written questions, each labelled with the article that answers it, plus 8 out-of-scope questions (`data/eval/questions.jsonl`).

| Retrieval | Recall@1 | Recall@3 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.604 | 0.771 | 0.875 | 0.720 |
| Dense | 0.354 | 0.542 | 0.646 | 0.482 |
| Hybrid (RRF) | 0.583 | 0.833 | **0.917** | **0.734** |

Abstention at the chosen threshold (0.55): all 48 in-scope questions are answered, 5 of 8 out-of-scope questions are refused.

What the numbers say:

- Lexical search beats a small multilingual embedding model on legal Turkish by a wide margin; fusion still adds 4 points of Recall@5 over BM25 alone.
- Similarity-based abstention catches clearly off-topic questions (recipes, weather, programming) but not questions from neighbouring legal areas (rent, labour, tax law). Separating those needs a relevance grader, not a threshold.

## Limitations

- The evaluation set is small and written by one person; the abstention threshold was chosen on the same set, so that figure is optimistic.
- Only retrieval and abstention are scored. Answer quality with a generative model (faithfulness, answer relevance) is not measured yet.
- No reranker. The four hybrid misses are questions whose answer sits in an article that shares little vocabulary with the question.

## Run

```bash
uv sync
uv run pytest
uv run mevzuat-rag eval
uv run mevzuat-rag ask "Bankalar belgelerini kaç yıl saklamak zorundadır?"
```

To generate answers with a model, point it at an OpenAI-compatible endpoint:

```bash
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=<model> LLM_API_KEY=<key-if-needed>
```

## Layout

```
src/mevzuat_rag/corpus.py     article parser
src/mevzuat_rag/retrieval.py  BM25, dense, RRF
src/mevzuat_rag/guard.py      PII masking, injection check
src/mevzuat_rag/llm.py        generators, citation validation
src/mevzuat_rag/graph.py      LangGraph pipeline
src/mevzuat_rag/evaluate.py   metrics
```

The legislation texts are public documents published by the Presidency's Mevzuat Bilgi Sistemi.
