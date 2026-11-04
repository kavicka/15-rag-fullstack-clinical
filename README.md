# 15 · P7: RAG app over clinical trials (FastAPI + React)

🇸🇰 [Slovenská verzia](README.sk.md)

## TL;DR
- A full-stack RAG app: a FastAPI backend that streams answers over Server-Sent Events, and a React + TypeScript UI
  with clickable citations and a sources panel that shows each stage's score for every retrieved chunk.
- Corpus: 396 completed interventional trials from 8 therapeutic areas (ClinicalTrials.gov API v2), 694k words.
  Eval: 156 questions built from structured fields (126 answerable with exact gold values, 30 unanswerable).
- **Answers (Qwen3-1.7B, cloud CPU): 15.9 % correct without retrieval → 96.8 % with hybrid search + reranker.**
  92.1 % are correct *and* cite the right trial. Unanswerable questions got "I don't know" in 30 / 30 cases, and no
  answer cited a trial that wasn't among its sources.
- **Retrieval: section-aware chunks + reranker put the passage with the answer at rank 1 for 85.7 % of questions**
  (BM25 42.1 %, dense 66.7 %, hybrid 49.2 %). At rank ≤ 5, section-aware chunks beat naive 220-word chunks in every
  mode (e.g. BM25 0.905 vs 0.770).
- One provider interface: Ollama (default, 0 €), a local Hugging Face model, AWS Bedrock or Azure OpenAI, chosen in
  `.env`. Bedrock and Azure are prepared for steps 18 and 19, not run here.

![The app answering with a citation, and the retrieved chunks with per-stage scores](assets/ui.png)

## Problem
Teams in clinical R&D ask questions like "how many patients did trial X enrol?" or "what masking did it use?".
An LLM alone either does not know or makes up a confident answer. RAG retrieves the relevant records first and
makes the model answer from them, with a citation a human can check. This step answers two questions with numbers:
1. which retrieval setup actually finds the passage that holds the answer, and
2. how much retrieval improves answers over the LLM alone, including saying "I don't know" when it should.

## Setup
- **Data:** ClinicalTrials.gov API v2 (free, no key). Up to 50 completed interventional trials with posted results for
  each of: type 2 diabetes, breast cancer, asthma, Alzheimer's disease, rheumatoid arthritis, hypertension, major
  depressive disorder and psoriasis, giving 396 unique trials. The id list is frozen in `eval/study_ids.txt`, so
  `make data` downloads the same corpus again.
- **Eval set** (`eval/questions.jsonl`, seed 0): 7 question types × 18 (phase, enrollment, lead sponsor, start year,
  masking, intervention, minimum age), plus 15 questions about a fact no record has (budget) and 15 about trials that
  do not exist. Questions name the trial by its title, never by NCT id.
- **Models:** embeddings `BAAI/bge-small-en-v1.5`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`. The LLM in the
  measured run is `Qwen/Qwen3-1.7B` through transformers (greedy decoding, thinking off), because the cloud machine
  has no Ollama. On the Mac the default is `qwen3:4b` in Ollama, which hasn't been measured yet.
- **Hardware:** all numbers come from a cloud Linux machine with 4 CPU cores and no GPU, so the latencies are CPU
  latencies.

## How it works
```
ClinicalTrials.gov ─► rag/ingest.py ─► rag/chunking.py ─► rag/index.py ─► BM25 (memory) + Qdrant (vectors)
                                                                              │
React UI ◄── SSE ── api/main.py ◄── rag/answer.py (prompt, citations) ◄── rag/retrieve.py (4 modes)
```
- **Records become text with sections** (`rag/ingest.py`). API enums like `PHASE2` / `DOUBLE` become "Phase 2" /
  "Double", because both BM25 and the LLM read words, not codes.
- **Two chunkings** (`rag/chunking.py`). `fixed` glues the record together and cuts every 220 words. `section` cuts
  along the record's sections and prefixes each chunk with `[NCT id] title | section`. Why: a chunk saying
  "Enrollment: 240 participants" is useless if nothing in it says which trial it belongs to.
- **Four retrieval modes** (`rag/retrieve.py`):
  - `bm25`: own 40-line implementation with the Lucene idf. rank_bm25's classic idf goes negative for words in over
    half the corpus.
  - `dense`: normalized bge vectors in Qdrant, so cosine equals the dot product (step 10).
  - `hybrid`: Reciprocal Rank Fusion, `Σ 1/(60 + rank)`. It uses only ranks, so BM25 scores and cosine similarities
    never have to be put on one scale.
  - `rerank`: a cross-encoder re-reads the top 20 hybrid candidates *together with the question*. It is too slow for
    the whole corpus but precise on a short list, which is the classic two-stage design.
- **Grounded prompt** (`rag/answer.py`). Every source is labelled with its trial id, and the model must cite after each
  fact and reply exactly "I don't know" when the sources do not contain the answer. After generation, any cited id
  that was not among the sources is flagged as a made-up citation.
- **API** (`api/main.py`): `POST /api/ask` streams `sources → token… → done` as SSE. SSE rather than WebSockets
  because the stream is one-way, plain HTTP, and passes through proxies and load balancers (steps 17–18). Pydantic
  validates input (question 3–500 chars, k 1–10).
- **UI** (`frontend/`): React 19 + TypeScript + Vite. It parses the SSE stream from `fetch` by hand, since
  `EventSource` cannot POST. Citations open the full source record, and the sources panel shows per-stage score bars
  so you can see *why* an answer came out the way it did. This UI is the step's visual example. The Racing Lab cars
  were not used because retrieval and citations have no natural driving analogue.
- **Secrets:** everything is read from the environment through `pydantic-settings`. `.env` is git-ignored and only
  `.env.example` is committed. AWS uses its default credential chain (SSO profile), never keys in a file.

## Results
### Retrieval
"evidence@k" means that a chunk which **contains the gold fact** (e.g. the exact line `Enrollment: 240 participants`)
is in the top k. "trial@5" only asks for any chunk of the right trial. 141 questions (the fake-trial ones have no gold
record).

| Chunking | Mode | evidence@1 | evidence@5 | evidence@10 | MRR@10 | trial@5 | p50 latency |
|---|---|---|---|---|---|---|---|
| fixed (220 words) | bm25 | 0.532 | 0.770 | 0.810 | 0.627 | 1.000 | 0.2 ms |
| fixed | dense | 0.540 | 0.786 | 0.841 | 0.654 | 0.986 | 75 ms |
| fixed | hybrid | 0.516 | 0.770 | 0.841 | 0.637 | 1.000 | 135 ms |
| fixed | rerank | 0.571 | 0.817 | 0.865 | 0.683 | 0.993 | 1041 ms |
| section + header | bm25 | 0.421 | 0.905 | 0.984 | 0.613 | 1.000 | 0.2 ms |
| section | dense | 0.667 | 0.968 | 0.992 | 0.788 | 1.000 | 144 ms |
| section | hybrid | 0.492 | 0.976 | 1.000 | 0.705 | 1.000 | 143 ms |
| section | **rerank** | **0.857** | **0.992** | 1.000 | **0.917** | 1.000 | 1042 ms |

![Retrieval](assets/retrieval.png)

### Answers
Section chunks, top 5 chunks in the prompt, 156 questions. "Grounded" = correct **and** cites the gold trial.
"False abstain" = said "I don't know" although the answer was retrievable.

| Variant | Accuracy | Grounded | False abstain | "I don't know" on unanswerable | Made-up citations | p50 / p95 latency |
|---|---|---|---|---|---|---|
| LLM alone (baseline) | 0.159 | 0.000 | 0.675 | 1.000 | 0.000 | 1.3 / 5.2 s |
| bm25 | 0.889 | 0.833 | 0.079 | 1.000 | 0.000 | 8.7 / 17.6 s |
| dense | 0.921 | 0.873 | 0.016 | 1.000 | 0.000 | 9.4 / 14.2 s |
| hybrid | 0.960 | **0.929** | 0.008 | 1.000 | 0.000 | 8.0 / 15.4 s |
| **rerank** | **0.968** | 0.921 | 0.008 | 1.000 | 0.000 | 10.1 / 17.1 s |

![Answers](assets/generation.png)

Clicking a citation opens the source record, so every answer can be checked in one click:

![Source record behind a citation](assets/ui-source.png)

## Error analysis
- **The LLM alone isn't hallucinating much, it mostly refuses** (67.5 % "I don't know"). Its 15.9 % "accuracy" is
  mostly *leakage from the question*: 12 of its 20 correct answers are intervention questions, where the drug name is
  already in the trial title ("Evaluation of Sonelokimab for …" → "Sonelokimab"). A fairer eval would hide the
  answer from the question.
- **BM25 fails on minimum age** (0.556 vs 1.000 with rerank). "What is the minimum age for participants" shares almost
  no rare words with `Minimum age: 18 Years`, so BM25 brings other chunks of the right trial and the model correctly
  says "I don't know" (all 8 of its misses there). Dense retrieval understands "minimum age ≈ at least 18 years old".
- **Near-duplicate trials.** "A Study to Test Different Doses of BI 1569912 in People With Depression" (225
  participants) has a sibling whose title is the same plus "Who Take Anti-depressive Medicine" (243). Rerank answered
  243 and cited the sibling, and BM25 gave up. The question is ambiguous for any retriever that ranks by overlap with
  the title.
- **The source can disagree with itself.** For one trial the Design section says `Masking: Triple`, but the summary
  says "double-masked study design". The model believed the prose. Masking is the weakest type for dense, hybrid and
  rerank (0.78–0.83).
- **"Masking: None" reads like "missing".** Several answers said "did not specify the type of masking" for
  `Masking: None`. Rendering it as "None (open label)" would remove the ambiguity.
- **A scorer bug, found by reading answers.** "The trial did not use any masking" was first scored as wrong. The
  rule was fixed (a test now covers both phrasings), and the stored answers were re-scored with
  `eval/run_eval.py --rescore` (no LLM re-run). This changed one rerank answer: 0.960 → 0.968.
- **Hybrid has a worse rank 1 than dense alone (0.492 vs 0.667) but a better top 5 (0.976 vs 0.968).** RRF rewards
  chunks both retrievers like, and BM25 likes every chunk whose header repeats the title. The reranker is what turns
  a good top 20 into a good rank 1.

## What I learned
- A retrieval metric has to check for the *answer*, not the *document*. Without that, BM25 looked perfect
  (trial@5 = 1.0).
- Retrieval quality explains most of the answer quality: the order of evidence@5 is the order of accuracy.
- A small 1.7B model is enough for extractive QA once the right passage is in the prompt. It never made up a citation.
- Read the wrong answers: two of the "model errors" were really an ambiguous eval question and a too-strict scorer.

## Run it
```bash
cp .env.example .env              # then edit; never commit .env
make sync                         # uv sync + npm ci
make data                         # frozen trial list → data/studies.jsonl, eval/questions.jsonl
make index                        # chunk + embed both chunkings (embedded Qdrant in data/qdrant)
ollama pull qwen3:4b
make eval-retrieval               # → results.json
make eval-generation              # → results.json, eval/runs/*.jsonl
make plot                         # → assets/*.png
make api                          # http://localhost:8000/docs
make web                          # http://localhost:5173
make test lint
```
To reproduce the measured run without Ollama: `LLM_PROVIDER=transformers LLM_MODEL=Qwen/Qwen3-1.7B make eval-generation`.
Optional: `docker compose up -d qdrant` and `QDRANT_URL=http://localhost:6333` for a real Qdrant server. Embedded
Qdrant allows one process at a time, so stop the API before running the eval.

## Limitations
- Questions are generated from templates and quote exact titles, which flatters keyword search and leaks intervention
  names. Paraphrased questions (written by an LLM, then checked by hand) are the next step.
- Answers are checked by rules (number present, name words present). This is transparent but can't judge long
  free-text answers, so the eval only asks short-fact questions.
- Only single-trial questions. Questions across trials ("which psoriasis trials tested X?") are not measured.
- Measured with Qwen3-1.7B on a cloud CPU, not with `qwen3:4b` on the Mac. Latencies are CPU latencies.
- Bedrock and Azure providers are written but not run (they cost money; planned for steps 18 and 19).
