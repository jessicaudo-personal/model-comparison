# Closed vs. Open-Weights Models: A Small-Scale Comparison

![status](https://img.shields.io/badge/status-v1%20complete-brightgreen) ![models](https://img.shields.io/badge/models-GPT--4o--mini%20%7C%20Llama%203.2-blue)

**Question:** When is a free, locally run open-weights model good enough, and when does a paid closed model earn its cost?

## What I built

A small Python harness that sends the same 12 prompts to multiple models (currently llama 3.2 vs. gpt-4o-mini) through one OpenAI-compatible codebase, then logs response, latency, token counts, and cost to CSV for scoring. Switching providers (OpenAI, Ollama, OpenRouter, Venice) means changing a base URL and model name, not rewriting code.

## Method

- **Models:** GPT-4o-mini (cloud API) vs. Llama 3.2 (local, via Ollama, on a Mac)
- **Prompts (12):** concept explanation, factual recall, math word problem, logic puzzle, coding, summarization, strict instruction following, creative writing, a hallucination trap (a fictional company), French, a sensitive-tone message, and a business judgment question
- **Settings:** temperature 0.7, max 400 tokens, one run per prompt
- **Scoring:** 1 to 5 per response, by hand. *[Edit: note whether you scored blind.]*

## Results

| | GPT-4o-mini | Llama 3.2 (local) |
|---|---|---|
| Avg quality score (1-5) | **4.7** | 3.3 |
| Outright failures | 0 | 2 |
| Avg latency | ~1.9s | ~2.0s |
| Total output tokens | 1,244 | 1,302 |
| Marginal cost | (will add actual price from pricing page later) | $0 |
| Hallucinated on fictional company? | No | No |

## What I found

1. **Speed and cost were not the differentiator; reliability was.** On this hardware the local model matched the cloud model on latency and cost nothing to run, but it failed two prompts outright: it described a "context window" as a chat-history view in messaging apps, and it contradicted itself on a three-person height puzzle (stating Bob is taller than Chen, then ranking Chen above Bob).
2. **Instruction following separated the models more than knowledge did.** Both got the factual lookup and the arithmetic right. Llama missed constraints: it added preambles, returned five options when asked for one message, and repeated "latency" in a list that required three distinct benefits.
3. **Both declined to invent facts about a fictional company.** That is the highest-stakes behavior in the set, and a small model passing it is not guaranteed.
4. **Failures hid in easy-looking prompts.** That makes "route easy requests to the cheap model" harder than it sounds, because difficulty is not obvious before the model answers.

## What this means for a product team

- A small local model is a reasonable fit for low-stakes drafting and simple lookups, especially where privacy or offline use matters.
- It should not be the unmonitored default for reasoning or tightly constrained output. A router needs a fallback to a stronger model, plus a way to detect when the cheap model has failed.
- Token counts are not comparable across model families (different tokenizers), so cost comparisons need the same task, not the same token total.
- A change like quantizing or swapping a default model should be gated on evals like these, then an A/B test, not on benchmark scores alone.

## Limitations

12 prompts, one run each, one scorer, and models of very different size and price class. These results show tradeoffs, not a ranking of model families. The first local call included model load time.

## Next steps

Rerun at temperature 0 for stability; add a larger Llama and a second quantization level (4-bit vs. 8-bit); add more reasoning and constraint prompts; have a second person score blind; add real per-token prices for a cost-per-successful-task metric.

## Reproduce it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your own keys
ollama pull llama3.2
python compare.py
```

Code: `compare.py`, `providers.py`, `models.json`, `prompts.txt`. Raw results: `results/`.
