"""Tiny, dependency-free retrieval helpers: load docs, split into sections, BM25 search."""
import math
import re
from collections import Counter
from pathlib import Path

STOPWORDS = set(
    "a an the is are was were of to in on for and or does do did how what when where which who "
    "whom with at by from as it its be can has have that this these those their there into".split()
)


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def load_chunks(docs_dir="docs"):
    """One chunk per markdown section (split on lines starting with '#')."""
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        section, lines = path.stem, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                if "".join(lines).strip():
                    chunks.append({"doc": path.name, "section": section, "text": "\n".join(lines).strip()})
                section, lines = line.lstrip("# ").strip(), []
            else:
                lines.append(line)
        if "".join(lines).strip():
            chunks.append({"doc": path.name, "section": section, "text": "\n".join(lines).strip()})
    if not chunks:
        raise SystemExit(f"No .md files found in '{docs_dir}'.")
    return chunks


def format_chunks(chunks):
    return "\n\n".join(f"[source: {c['doc']} | {c['section']}]\n{c['text']}" for c in chunks)


class BM25:
    """Classic keyword-ranking formula. Simple, fast, and easy to explain in an interview."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [tokenize(c["section"] + " " + c["text"]) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query, k=3):
        q = tokenize(query)
        scored = []
        for i, d in enumerate(self.docs):
            dl = len(d) or 1
            score = 0.0
            for t in q:
                f = self.tf[i].get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    score += self.idf[t] * f * (self.k1 + 1) / norm
            scored.append((score, i))
        scored.sort(reverse=True)
        return [self.chunks[i] for s, i in scored[:k] if s > 0]
