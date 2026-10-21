"""Summarize a RAG results CSV into the metrics that matter.

Uses your manual 'correct' column when filled in (1/0), otherwise the auto-heuristics.
Usage:
    python score_rag.py                         # latest results/rag_run_*.csv
    python score_rag.py results/rag_run_X.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def truthy(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def judged_correct(row):
    """Returns (True/False/None, source). For unanswerable rows, 'correct' means it abstained."""
    manual = row.get("correct", "").strip()
    if manual:
        return truthy(manual), "manual"
    if row["qtype"] == "unanswerable":
        return truthy(row["auto_abstained"]), "auto"
    if row["auto_match"] == "":
        return None, "none"
    return truthy(row["auto_match"]), "auto"


def pct(num, den):
    return "n/a" if den == 0 else f"{100 * num / den:.0f}% ({num}/{den})"


def summarize(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["condition"])].append(r)

    out_rows = []
    print("=== Summary ===")
    for (label, cond), rs in groups.items():
        errors = sum(1 for r in rs if r["error"])
        ok = [r for r in rs if not r["error"]]
        answerable = [r for r in ok if r["qtype"] != "unanswerable"]
        unans = [r for r in ok if r["qtype"] == "unanswerable"]
        cf = [r for r in ok if r["qtype"] == "counterfactual"]

        def count_correct(group):
            judged = [judged_correct(r)[0] for r in group]
            judged = [j for j in judged if j is not None]
            return sum(judged), len(judged)

        a_ok, a_n = count_correct(answerable)
        u_ok, u_n = count_correct(unans)
        c_ok, c_n = count_correct(cf)
        false_abs = sum(truthy(r["auto_abstained"]) for r in answerable)
        mem = sum(truthy(r["matches_memory"]) for r in cf)
        hits = [r for r in ok if r["retrieval_hit"] != ""]
        hit_n = sum(truthy(r["retrieval_hit"]) for r in hits)
        unsupp = [float(r["unsupported_claims"]) for r in ok if r["unsupported_claims"].strip() != ""]
        manual_n = sum(1 for r in ok if r["correct"].strip())

        stats = {
            "model": label, "condition": cond,
            "correct_on_answerable": pct(a_ok, a_n),
            "abstained_on_answerable": pct(false_abs, len(answerable)),
            "correct_abstention_on_unanswerable": pct(u_ok, u_n),
            "counterfactual_followed_doc": pct(c_ok, c_n),
            "counterfactual_used_memory": pct(mem, len(cf)),
            "retrieval_hit_rate": pct(hit_n, len(hits)) if hits else "n/a",
            "avg_unsupported_claims": f"{sum(unsupp) / len(unsupp):.2f}" if unsupp else "not scored",
            "errors": errors, "rows_manually_scored": f"{manual_n}/{len(ok)}",
        }
        out_rows.append(stats)

        print(f"\n== {label} | {cond} ==")
        print(f"  Correct on answerable questions:        {stats['correct_on_answerable']}")
        print(f"  Abstained on answerable questions:      {stats['abstained_on_answerable']}  (false abstention)")
        print(f"  Correctly abstained on unanswerable:    {stats['correct_abstention_on_unanswerable']}")
        print(f"  Counterfactual: followed the document:  {stats['counterfactual_followed_doc']}")
        print(f"  Counterfactual: answered from memory:   {stats['counterfactual_used_memory']}")
        if hits:
            print(f"  Retrieval found the right doc(s):       {stats['retrieval_hit_rate']}")
        print(f"  Avg unsupported claims (manual):        {stats['avg_unsupported_claims']}")
        print(f"  Errors: {errors} | rows scored by hand: {stats['rows_manually_scored']}")

    summary_path = Path(path).with_name(Path(path).name.replace("rag_run_", "rag_summary_"))
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nSummary saved: {summary_path}")
    print("Note: auto-heuristics are rough (keyword matching). Check the rows by hand before quoting numbers.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        files = sorted(Path("results").glob("rag_run_*.csv"))
        if not files:
            raise SystemExit("No results/rag_run_*.csv found. Run rag_compare.py first.")
        target = files[-1]
    summarize(target)
