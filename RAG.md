# RAG Extension: Grounded Question Answering

Tests how well each model **uses information it is handed**, a different skill from what it remembers.

## Conditions (same questions, same models)

| Condition | Context given | Question it answers |
|---|---|---|
| `no_context` | None | What does the model remember (or make up)? |
| `retrieved` | Top-k passages from keyword (BM25) search | Does it use the material and stay grounded? |
| `oracle` | The known-correct source doc(s) | If it still fails here, the problem is capability, not retrieval |

Retrieval runs once per question, so every model sees identical passages. The only variable is the model.

## Question set (`questions.csv`, 14 questions)

- 6 **answerable** (one fact in one doc)
- 2 **multihop** (combine facts, some with arithmetic)
- 4 **unanswerable** (the docs don't say; the right behavior is to abstain)
- 2 **counterfactual** (the doc deliberately contradicts real-world facts; a grounded model repeats the doc)

## Test data

`docs/kestrel_*.md` describe a **fictional** product, so no model can know the answers from training.
`docs/ml_primer.md` contains **deliberately altered facts** (a wrong publication year and wrong bytes-per-weight)
to test whether a model reads the context or recalls its memory. Do not use it as a reference.

## Run

```
python rag_compare.py                       # all conditions, all enabled models
python rag_compare.py --conditions retrieved oracle --top-k 2
python rag_compare.py --retrieval full      # paste every doc instead of searching
python score_rag.py                         # summarize the latest run
```

## Scoring columns to fill in by hand (in `results/rag_run_*.csv`)

- `correct`: 1 or 0. For unanswerable questions, 1 = it correctly said the docs don't contain the answer.
  For counterfactual questions, 1 = it matched the **document** (not real-world knowledge).
- `unsupported_claims`: count of claims in the answer that the passages do not support.
- `notes`: anything interesting.

Leave `correct` blank to fall back on the auto-heuristics (keyword matching), which are rough.

## Metrics reported

| Metric | Meaning |
|---|---|
| Correct on answerable | Right answer on questions the docs can answer |
| Abstained on answerable | False abstention: refused when it shouldn't have |
| Correct abstention | Said "not in the docs" on unanswerable questions |
| Counterfactual followed doc / used memory | Read the context vs. recalled outside knowledge |
| Retrieval hit rate | Did search find the right doc(s)? (separates retrieval failures from model failures) |
| Avg unsupported claims | Faithfulness, measured by hand |

In the `no_context` condition, abstaining on Kestrel questions is expected (the model can't know them), so
read that condition as "does it admit ignorance or invent an answer?"

## Limits

14 questions and one run each is a small sample: a single answer swings percentages a lot. Report counts
alongside percentages, and have a second person score a subset.
