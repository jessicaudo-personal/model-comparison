"""Compare models on grounded question answering (RAG) under three conditions:

  no_context  - the question alone (what does the model remember?)
  retrieved   - the question + passages found by keyword search (does it use the material?)
  oracle      - the question + the known-correct source doc(s) (is a failure knowledge or capability?)

Retrieval runs ONCE per question, so every model sees identical context.

Usage:
    python rag_compare.py
    python rag_compare.py --conditions retrieved oracle --top-k 2 --retrieval bm25
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from compare import compute_cost, load_models
from providers import get_client
from rag import BM25, format_chunks, load_chunks

ABSTAIN_PHRASE = "Not in the provided documents."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the passages provided. Do not use outside knowledge. "
    f"If the passages do not contain the answer, reply exactly: {ABSTAIN_PHRASE} "
    "Otherwise answer in 1-3 sentences, then add a final line starting with 'Evidence:' "
    "that quotes the supporting sentence from the passages."
)

ABSTAIN_PATTERNS = [
    "not in the provided documents", "i don't have", "i do not have", "i couldn't find",
    "i could not find", "i can't find", "i cannot find", "unable to find", "no information",
    "not aware of", "i don't know", "don't have access", "not mentioned", "isn't mentioned",
    "doesn't mention", "does not mention", "not provided", "not specified", "no mention",
]

FIELDS = [
    "timestamp", "condition", "label", "provider", "model", "qid", "qtype", "question",
    "gold_answer", "retrieved_docs", "retrieval_hit", "response", "latency_s",
    "input_tokens", "output_tokens", "cost_usd", "error",
    "auto_abstained", "auto_match", "matches_memory",   # heuristics, computed for you
    "correct", "unsupported_claims", "notes",           # fill in by hand (correct: 1 or 0)
]


def load_questions(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def keyword_match(text, spec):
    """spec like 'a|b/c' = must contain a AND (b OR c). Returns '' when there is no spec."""
    if not spec:
        return ""
    t = text.lower()
    return all(any(alt.strip().lower() in t for alt in kw.split("/")) for kw in spec.split("|"))


def looks_like_abstention(text):
    t = text.lower().replace("\u2019", "'")
    return any(p in t for p in ABSTAIN_PATTERNS)


def gold_docs(q):
    return [s.strip() for s in q["source_docs"].split(";") if s.strip()]


def get_context(condition, q, chunks, bm25, args):
    """Returns (context_text, list_of_doc_names_used)."""
    if condition == "no_context":
        return "", []
    if condition == "oracle":
        wanted = gold_docs(q)  # unanswerable questions have no gold doc, so they get every doc
        selected = [c for c in chunks if c["doc"] in wanted] if wanted else chunks
    elif args.retrieval == "full":
        selected = chunks
    else:
        selected = bm25.search(q["question"], args.top_k)
    return format_chunks(selected), sorted({c["doc"] for c in selected})


def build_messages(condition, question, context):
    if condition == "no_context":
        return [{"role": "user", "content": question}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},  # question last
    ]


def ask(entry, messages, temperature, max_tokens):
    client = get_client(entry["provider"])
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=entry["model"], messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    latency = time.perf_counter() - start
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return text, latency, getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="models.json")
    ap.add_argument("--questions", default="questions.csv")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--conditions", nargs="+", default=["no_context", "retrieved", "oracle"],
                    choices=["no_context", "retrieved", "oracle"])
    ap.add_argument("--retrieval", choices=["bm25", "full"], default="bm25",
                    help="bm25 = top-k keyword search; full = paste every doc")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()

    models = load_models(args.models)
    if not models:
        raise SystemExit('No enabled models. Set "enabled": true in models.json.')
    questions = load_questions(args.questions)
    chunks = load_chunks(args.docs)
    bm25 = BM25(chunks)

    # Retrieve once per question so every model gets identical context.
    contexts = {(q["id"], c): get_context(c, q, chunks, bm25, args) for q in questions for c in args.conditions}
    print(f"{len(questions)} questions x {len(args.conditions)} conditions x {len(models)} models "
          f"= {len(questions) * len(args.conditions) * len(models)} calls\n")

    rows = []
    for entry in models:
        for cond in args.conditions:
            print(f"== {entry['label']} | {cond} ==")
            for q in questions:
                context, docs_used = contexts[(q["id"], cond)]
                wanted = gold_docs(q)
                hit = "" if cond != "retrieved" or not wanted else all(d in docs_used for d in wanted)
                row = {k: "" for k in FIELDS}
                row.update(
                    timestamp=datetime.now().isoformat(timespec="seconds"), condition=cond,
                    label=entry["label"], provider=entry["provider"], model=entry["model"],
                    qid=q["id"], qtype=q["type"], question=q["question"], gold_answer=q["gold_answer"],
                    retrieved_docs=";".join(docs_used), retrieval_hit=hit,
                )
                try:
                    text, latency, in_tok, out_tok = ask(
                        entry, build_messages(cond, q["question"], context), args.temperature, args.max_tokens
                    )
                    cost = compute_cost(entry, in_tok, out_tok)
                    row.update(
                        response=text, latency_s=round(latency, 2), input_tokens=in_tok,
                        output_tokens=out_tok, cost_usd="" if cost is None else round(cost, 6),
                        auto_abstained=looks_like_abstention(text),
                        auto_match=keyword_match(text, q["gold_keywords"]),
                        matches_memory=keyword_match(text, q["memory_keywords"]),
                    )
                    print(f"  {q['id']} ({q['type']}) {latency:.1f}s")
                except Exception as e:  # keep going so one failure doesn't kill the run
                    row["error"] = str(e)[:300]
                    print(f"  {q['id']} ERROR: {row['error']}")
                rows.append(row)

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"rag_run_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")
    print("Open it and fill in 'correct' (1/0) and 'unsupported_claims' (a count) for each row,")
    print("then run:  python score_rag.py   (the numbers below use auto-heuristics only)\n")

    from score_rag import summarize
    summarize(out)


if __name__ == "__main__":
    main()
