"""Provider settings. Every provider here speaks the OpenAI-style API,
so one SDK (openai) can talk to all of them by changing base_url + key."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDERS = {
    "openai":     {"base_url": None,                            "key_env": "OPENAI_API_KEY"},
    "ollama":     {"base_url": "http://localhost:11434/v1",     "key_env": None},  # local, no key
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",  "key_env": "OPENROUTER_API_KEY"},
    "venice":     {"base_url": "https://api.venice.ai/api/v1",  "key_env": "VENICE_API_KEY"},
}


def get_client(provider: str) -> OpenAI:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Add it to PROVIDERS in providers.py.")
    cfg = PROVIDERS[provider]
    if cfg["key_env"]:
        api_key = os.getenv(cfg["key_env"])
        if not api_key:
            raise RuntimeError(f"Missing {cfg['key_env']}. Add it to your .env file.")
    else:
        api_key = "ollama"  # Ollama ignores the key, but the SDK requires a non-empty value
    return OpenAI(base_url=cfg["base_url"], api_key=api_key)
