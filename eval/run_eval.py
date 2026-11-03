"""Measure retrieval and generation on eval/questions.jsonl and merge the numbers into results.json.

Retrieval (no LLM needed): for every chunking × mode, where does the first chunk of the gold trial rank?
  recall@k = share of questions with the gold trial in the top k; MRR = mean of 1/rank.
Generation (needs an LLM): no retrieval vs every retrieval mode, section chunking, top 5 chunks.
  accuracy        answerable questions answered correctly (rule check vs the gold value)
  grounded        correct AND cites the gold trial: the answer is right for the right reason
  false_abstain   answerable questions where the model said "I don't know"
  abstain_unans   unanswerable questions where it correctly said "I don't know"
  bad_citations   answers citing a trial that was not among the sources (made-up citation)

Run:  uv run python eval/run_eval.py --retrieval
      uv run python eval/run_eval.py --generation [--limit 20] [--variants no_rag rerank]
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.answer import finalize, stream_answer  # noqa: E402
from rag.chunking import CHUNKERS  # noqa: E402
from rag.config import ROOT, get_settings  # noqa: E402
from rag.llm import get_llm  # noqa: E402
from rag.retrieve import MODES, Retriever  # noqa: E402
from rag.scoring import is_correct  # noqa: E402

RESULTS = ROOT / "results.json"
RUNS = ROOT / "eval" / "runs"
K = 5


def load_questions() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open()]


def hardware() -> str:
    import torch

    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return f"{platform.system()} {platform.machine()}, torch device {dev}"


def _ws(text: str) -> str:
    return " ".join(text.split())


def has_evidence(chunk: dict, q: dict) -> bool:
    return chunk["nct_id"] == q["nct_id"] and _ws(q["evidence"]) in _ws(chunk["text"])


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


def merge_results(section: str, data: dict) -> None:
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res[section] = data
    RESULTS.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")


def eval_retrieval(qs: list[dict], modes: list[str]) -> None:
    s = get_settings()
    qs = [q for q in qs if q["nct_id"]]  # fake trials have no gold document
    out: dict = {
        "meta": {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "hardware": hardware(),
            "embed_model": s.embed_model,
            "rerank_model": s.rerank_model,
            "n_questions": len(qs),
            "modes": modes,
        },
    }
    for chunking in CHUNKERS:
        r = Retriever(chunking, dense=any(m != "bm25" for m in modes))
        stats = json.loads((s.data_dir / "index" / chunking / "stats.json").read_text())
        out[chunking] = {"n_chunks": stats["n_chunks"], "avg_words": stats["avg_words"]}
        for mode in modes:
            ranks, ev_ranks, lat = [], [], []
            for q in qs:
                hits, ms = r.timed_search(q["question"], mode, k=10)
                ids = [h.chunk["nct_id"] for h in hits]
                ranks.append(ids.index(q["nct_id"]) + 1 if q["nct_id"] in ids else None)
                if q["evidence"]:  # budget questions have no evidence chunk
                    ev = [i for i, h in enumerate(hits, start=1) if has_evidence(h.chunk, q)]
                    ev_ranks.append(ev[0] if ev else None)
                lat.append(ms)
            n, ne = len(ranks), len(ev_ranks)
            out[chunking][mode] = {
                # the main metric: a chunk that actually contains the answer is in the top k
                **{f"evidence@{k}": round(sum(1 for x in ev_ranks if x and x <= k) / ne, 3) for k in (1, 5, 10)},
                "evidence_mrr@10": round(sum(1 / x for x in ev_ranks if x) / ne, 3),
                # easier: any chunk of the right trial is in the top k
                **{f"trial@{k}": round(sum(1 for x in ranks if x and x <= k) / n, 3) for k in (1, 5)},
                "latency_ms_p50": pct(lat, 0.5),
                "latency_ms_p95": pct(lat, 0.95),
            }
            print(chunking, mode, out[chunking][mode])
        if r.qdrant:
            r.qdrant.close()
    merge_results("retrieval", out)


def eval_generation(qs: list[dict], variants: list[str]) -> None:
    s = get_settings()
    llm = get_llm(s)
    r = Retriever(s.chunking)
    RUNS.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "meta": {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "hardware": hardware(),
            "llm": llm.name,
            "chunking": s.chunking,
            "k": K,
            "n_questions": len(qs),
        }
    }
    for v in variants:
        records = []
        for q in qs:
            t0 = time.perf_counter()
            hits = None if v == "no_rag" else r.search(q["question"], v, K)
            text = "".join(stream_answer(llm, q["question"], hits))
            ms = (time.perf_counter() - t0) * 1000
            a = finalize(text, hits)
            ans = q["kind"] != "unanswerable"
            correct = ans and not a.abstained and is_correct(q["kind"], q["gold"], a.text)
            records.append(
                {
                    "id": q["id"],
                    "kind": q["kind"],
                    "answer": a.text,
                    "correct": correct,
                    "grounded": correct and q["nct_id"] in a.citations,
                    "abstained": a.abstained,
                    "bad_citations": a.unsupported_citations,
                    "retrieved": [h.chunk["nct_id"] for h in hits or []],
                    "gold_retrieved": bool(hits) and q["nct_id"] in [h.chunk["nct_id"] for h in hits],
                    "ms": round(ms),
                }
            )
        (RUNS / f"generation_{v}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
        a_rec = [x for x in records if x["kind"] != "unanswerable"]
        u_rec = [x for x in records if x["kind"] == "unanswerable"]
        out[v] = {
            "accuracy": round(sum(x["correct"] for x in a_rec) / len(a_rec), 3),
            "grounded": round(sum(x["grounded"] for x in a_rec) / len(a_rec), 3),
            "false_abstain": round(sum(x["abstained"] for x in a_rec) / len(a_rec), 3),
            "abstain_unans": round(sum(x["abstained"] for x in u_rec) / len(u_rec), 3) if u_rec else None,
            "bad_citations": round(sum(bool(x["bad_citations"]) for x in records) / len(records), 3),
            "latency_ms_p50": pct([x["ms"] for x in records], 0.5),
            "latency_ms_p95": pct([x["ms"] for x in records], 0.95),
            "by_kind": {
                k: round(statistics.mean(x["correct"] for x in a_rec if x["kind"] == k), 3)
                for k in sorted({x["kind"] for x in a_rec})
            },
        }
        print(v, {k: val for k, val in out[v].items() if k != "by_kind"})
    merge_results("generation", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--modes", nargs="+", default=list(MODES), help="retrieval modes to evaluate")
    ap.add_argument("--variants", nargs="+", default=["no_rag", *MODES])
    ap.add_argument("--limit", type=int, default=None, help="every n-th question only, for a quick smoke run")
    args = ap.parse_args()
    qs = load_questions()
    if args.limit:
        qs = qs[:: max(1, len(qs) // args.limit)]
    if args.retrieval:
        eval_retrieval(qs, args.modes)
    if args.generation:
        eval_generation(qs, args.variants)


if __name__ == "__main__":
    main()
