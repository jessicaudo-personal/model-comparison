# Closed vs. Open-Weights Models: Quality and Grounding

![status](https://img.shields.io/badge/status-v2%20complete-brightgreen) ![models](https://img.shields.io/badge/models-GPT--4o--mini%20%7C%20Llama%203.2-blue) ![eval](https://img.shields.io/badge/eval-free--form%20%2B%20RAG-orange)

**Question:** When is a free, locally run open-weights model good enough, and when does a paid closed model earn its cost? And when does adding retrieval matter more than model size?

## What I built

A small Python harness that sends the same prompts to multiple models through one OpenAI-compatible codebase and logs response, latency, tokens, and cost to CSV for scoring. Switching providers (OpenAI, Ollama, OpenRouter, Venice) means changing a base URL and model name. A second script adds a retrieval-augmented (RAG) grounding test with a keyword search, a question set, and a scoring summary.

**Models:** GPT-4o-mini (cloud API) vs. Llama 3.2 (local, via Ollama, on a Mac). Both parts use the same two models.

## Part 1: Free-form comparison (12 prompts)

Prompts covered explanation, factual recall, math, a logic puzzle, coding, summarization, strict instruction following, creative writing, a hallucination trap, French, a sensitive-tone message, and a business judgment question. Temperature 0.7, one run each, scored 1 to 5 by hand. *[Edit: note whether you scored blind.]*

| | GPT-4o-mini | Llama 3.2 (local) |
|---|---|---|
| Avg quality score (1-5) | **4.7** | 3.3 |
| Outright failures | 0 | 2 |
| Avg latency | ~1.9s | ~2.0s |
| Total output tokens | 1,244 | 1,302 |
| Marginal cost | *[fill in from pricing page]* | $0 |
| Hallucinated on fictional company? | No | No |

- **Reliability, not speed or cost, was the difference.** Llama failed two prompts outright: it described a "context window" as a chat-history view in messaging apps, and it contradicted itself on a three-person height puzzle.
- **Instruction following separated the models more than knowledge did.** Llama added preambles, returned five options when asked for one message, and repeated an item in a list that required three distinct benefits.
- **Failures hid in easy-looking prompts,** which makes "route easy requests to the cheap model" harder than it sounds.

## Part 2: Grounded question answering (RAG)

**Design.** 14 questions over short documents about a *fictional* product, so no model could know the answers from training: 6 answerable, 2 multi-hop (including arithmetic), 4 unanswerable (the right behavior is to abstain), and 2 counterfactual (a document deliberately contradicts real-world facts, to test whether the model reads the context or recalls memory). Each question ran under three conditions with identical context for both models:

- **No context:** the question alone (what does it remember, or invent?)
- **Retrieved:** top-3 passages from keyword (BM25) search
- **Oracle:** the known-correct document, to separate retrieval failures from model failures

Temperature 0, answers required to quote supporting evidence or reply "Not in the provided documents." Scoring: *[Edit: describe how you scored. First-pass scoring was done with AI assistance against the source documents; state how many rows you reviewed yourself.]*

| | GPT-4o-mini | Llama 3.2 |
|---|---|---|
| **No context:** correct on answerable (n=10) | 0 | 0 |
| **No context:** correct abstention (n=4) | 3 | 4 |
| **Retrieved:** correct on answerable (n=10) | 10 | 9 |
| **Retrieved:** correct abstention (n=4) | 4 | 4 |
| **Oracle:** correct on answerable (n=10) | 10 | 10 |
| **Oracle:** correct abstention (n=4) | 4 | 4 |
| Counterfactual, with context: followed the document (n=2) | 2 | 2 |
| Counterfactual, no context: answered from memory (n=2) | 2 | 2 |
| Retrieval found the right document (n=10) | 10 | 10 |

### Findings

1. **For both models, the gap was knowledge, not capability.** Going from no context to retrieved passages took them from 0 of 10 to 10 and 9 of 10. Handing over the perfect document (oracle) added almost nothing, because search had already found the right document every time.
2. **Both abstained correctly when given context,** including a question where a model could have wrongly inferred an answer from a founder's name.
3. **Both read the context over memory.** On deliberately altered facts, each repeated the document's version with context and reverted to the real-world answer without it.
4. **Without context, models hallucinated through name collisions.** GPT answered about the real ASP.NET "Kestrel" web server, including a launch date. Llama said it couldn't find information, then asserted Kestrel was a Rust server released in 2013, and invented that "Plus" was an AT&T plan. Average unsupported claims per answer without context: 0.43 (GPT) and 0.29 (Llama). Keyword-based auto-scoring counted Llama's first answer as an abstention, which is why hand-scoring mattered.
5. **Llama's one retrieved-condition miss was a multi-step failure.** It ignored a 40% student discount that was in its passages and answered the undiscounted price. GPT used the same passages correctly, and Llama solved it in the oracle condition. The failure was in using the information, not finding it.
6. **Small formatting slips on abstentions.** Llama often added a meaningless "Evidence:" line when declining, and once cited an unrelated sentence.

## What this means for a product team

- **Use retrieval first for fresh, private, or obscure information.** It fixed both models here at no extra model cost. Bigger models fix capability gaps, not missing knowledge.
- **A small local model can handle extract-and-quote tasks well,** especially with a strict abstention instruction. It was weaker on free-form reasoning and constraints (Part 1) and on multi-step use of context (the discount miss).
- **Measure abstention separately from accuracy.** A model that always answers looks accurate until it meets an unanswerable question, and one that always refuses looks safe but is useless. Track both.
- **Test for name collisions.** Models confidently describe a famous product that shares a name with yours. Unsupported-claim counts without context made this visible.
- **Gate model swaps on evals like these** and an A/B test, not on benchmark scores alone. Token counts aren't comparable across model families, so compare cost on the same task.

## Limitations

Small samples: 12 prompts in Part 1 and 14 questions in Part 2, one run each, one scorer, and two models of very different size and price class. The grounding questions were easy (short documents, one-sentence answers), so a harder test with long documents and many distractors would likely separate the models more. Search was simple keyword ranking, and the "retrieval hit" metric checks at document level, not passage level. Two scoring calls were borderline (one hedged no-context answer, and counterfactual no-context rows scored against a document the model never saw). The first local call included model load time. These results show tradeoffs, not a ranking of model families.

## Next steps

Rerun at temperature 0 for Part 1; add a larger Llama and a second quantization level (4-bit vs. 8-bit); build longer documents with distractor passages; try embedding-based retrieval against keyword search; add a second blind scorer; add real per-token prices for a cost-per-successful-task metric.

## Repo guide and reproduce

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your own keys
ollama pull llama3.2

python compare.py             # Part 1: free-form comparison
python rag_compare.py         # Part 2: grounded QA (three conditions)
python score_rag.py           # summarize the latest RAG run
```

| File | Purpose |
|---|---|
| `compare.py`, `providers.py`, `models.json`, `prompts.txt` | Part 1 harness and model roster |
| `rag_compare.py`, `rag.py`, `score_rag.py` | Part 2 harness, keyword search, scoring summary |
| `questions.csv`, `docs/` | Grounding question set and test documents (`docs/ml_primer.md` contains deliberately altered facts) |
| `RAG.md` | Detailed notes on the RAG extension and its columns |
| `results/` | Raw and scored run CSVs |
