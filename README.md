# Model Comparison

Runs the same prompts across closed and open models (OpenAI, Ollama, OpenRouter, Venice)
using the OpenAI SDK, and logs latency, tokens, cost, and responses to CSV for scoring.

## Setup (macOS)
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then add your keys
```

## Run
```
ollama pull llama3.2       # only if using the local model
python compare.py
```

## Customize
- `models.json`: add models, set `"enabled": true`, fill in prices from each provider's pricing page.
- `prompts.txt`: one prompt per line, `#` for comments.
- Results land in `results/` as CSV. Fill in the `score` and `notes` columns by hand.
