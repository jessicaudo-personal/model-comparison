"""Run the same prompts against several models and log quality-relevant data to CSV.

Usage:
    python compare.py
    python compare.py --models models.json --prompts prompts.txt --temperature 0.7 --max-tokens 400
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from providers import get_client

FIELDS = [
    "timestamp", "label", "provider", "model", "prompt", "response",
    "latency_s", "input_tokens", "output_tokens", "tokens_per_sec",
    "cost_usd", "error", "score", "notes",  # score + notes: fill in by hand afterward
]


def load_prompts(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def load_models(path):
    models = json.loads(Path(path).read_text(encoding="utf-8"))
    return [m for m in models if m.get("enabled")]


def run_one(entry, prompt, temperature, max_tokens):
    client = get_client(entry["provider"])
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=entry["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", None)
    out_tok = getattr(usage, "completion_tokens", None)
    return text, latency, in_tok, out_tok


def compute_cost(entry, in_tok, out_tok):
    ip, op = entry.get("input_price_per_m"), entry.get("output_price_per_m")
    if ip is None or op is None or in_tok is None or out_tok is None:
        return None
    return (in_tok * ip + out_tok * op) / 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models.json")
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=400)
    args = ap.parse_args()

    models = load_models(args.models)
    prompts = load_prompts(args.prompts)
    if not models:
        raise SystemExit("No enabled models. Set \"enabled\": true in models.json.")
    print(f"Running {len(prompts)} prompts x {len(models)} models\n")

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results") / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.csv"
    rows = []

    for entry in models:
        print(f"== {entry['label']} ==")
        for i, prompt in enumerate(prompts, 1):
            row = {k: "" for k in FIELDS}
            row.update(timestamp=datetime.now().isoformat(timespec="seconds"),
                       label=entry["label"], provider=entry["provider"],
                       model=entry["model"], prompt=prompt)
            try:
                text, latency, in_tok, out_tok = run_one(entry, prompt, args.temperature, args.max_tokens)
                row.update(response=text, latency_s=round(latency, 2),
                           input_tokens=in_tok, output_tokens=out_tok)
                if out_tok and latency > 0:
                    row["tokens_per_sec"] = round(out_tok / latency, 1)
                cost = compute_cost(entry, in_tok, out_tok)
                if cost is not None:
                    row["cost_usd"] = round(cost, 6)
                print(f"  [{i}/{len(prompts)}] {latency:.1f}s")
            except Exception as e:  # keep going so one failure doesn't kill the run
                row["error"] = str(e)[:300]
                print(f"  [{i}/{len(prompts)}] ERROR: {row['error']}")
            rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Summary ===")
    for entry in models:
        mine = [r for r in rows if r["label"] == entry["label"]]
        ok = [r for r in mine if not r["error"]]
        if ok:
            avg_lat = sum(float(r["latency_s"]) for r in ok) / len(ok)
            costs = [float(r["cost_usd"]) for r in ok if r["cost_usd"] != ""]
            cost_txt = f"${sum(costs):.4f}" if costs else "n/a"
            print(f"{entry['label']}: {len(ok)}/{len(mine)} ok, avg latency {avg_lat:.1f}s, total cost {cost_txt}")
        else:
            print(f"{entry['label']}: 0/{len(mine)} ok (check errors in the CSV)")
    print(f"\nSaved: {out_path}\nOpen it and fill in the 'score' and 'notes' columns (e.g. score 1-5).")


if __name__ == "__main__":
    main()
