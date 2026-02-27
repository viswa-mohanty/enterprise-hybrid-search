from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


# Load embedding model once at startup (local SentenceTransformer)
model = SentenceTransformer("all-mpnet-base-v2")


@dataclass
class SearchResult:
    """
    Container for a single search result with detailed scoring components.
    """

    title: str
    owner: str
    keyword_score: float
    vector_score: float
    final_score: float
    content: str


class HybridSearchEngine:
    """
    Lightweight wrapper around the functional API for environments
    that prefer a class-based interface.
    """

    def __init__(self, data_path: Path | str) -> None:
        self.data_path = Path(data_path)

        # Load raw policy documents from JSON.
        self._documents = self._load_documents()

        # Startup embedding phase for this engine instance.
        initialize_embeddings(self._documents)

    def _load_documents(self) -> List[dict]:
        """Load policy documents from the provided JSON file."""
        with self.data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("policies.json must contain a list of documents")
        return data

    def search(self, query: str, top_k: int = 5, alpha: float | None = None) -> List[SearchResult]:
        """
        Run a hybrid search using the functional hybrid_search() helper.

        The alpha parameter is accepted for backward compatibility but
        ignored in favor of the fixed weights in hybrid_search().
        """
        results = hybrid_search(query=query, policies=self._documents)
        return results[: max(1, min(int(top_k), len(results)))]


# -------------------------------------------------------------------------
# Functional orchestration API
# -------------------------------------------------------------------------

def _build_policy_text(policy: dict) -> str:
    """
    Build a single searchable string from key policy fields.
    """
    parts: List[str] = []
    for key in ("title", "region", "owner", "content"):
        value = policy.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _tokenize(text: str) -> List[str]:
    """
    Very small tokenizer for keyword matching:
    - lowercases the text
    - splits on non-alphanumeric characters
    """
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t]


def get_embedding(text: str, dim: int = 1536) -> np.ndarray:
    """
    Get a normalized embedding for the given text using the global
    SentenceTransformer model instance.
    """
    embedding = model.encode(text)
    embedding = embedding.astype("float32")
    norm = np.linalg.norm(embedding)
    if norm == 0.0:
        return embedding
    return embedding / norm


def keyword_score(query_tokens: List[str], policy_tokens: List[str]) -> int:
    """
    Simple keyword score:
    - counts how many unique query tokens appear in the policy tokens.
    """
    if not query_tokens or not policy_tokens:
        return 0

    query_terms = set(query_tokens)
    policy_terms = set(policy_tokens)
    return sum(1 for term in query_terms if term in policy_terms)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Manual cosine similarity using NumPy only:

        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    """
    if a is None or b is None:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 0.0

    a = a.astype("float32")
    b = b.astype("float32")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks of length chunk_size.
    Each new chunk overlaps the previous chunk by overlap characters.
    Returns a list of non-empty chunk strings.
    """
    if not text or chunk_size <= 0:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def initialize_embeddings(policies: List[dict]) -> None:
    """
    Startup embedding phase.

    This function is intended to be called once at application startup
    (or before the first query) to precompute and attach an embedding
    vector to each policy in memory.

    For each policy:
      - take the policy's content text and split it into chunks via chunk_text()
      - for each chunk, compute an embedding and store text + embedding
      - set policy["chunks"] = [{"text": chunk_text, "embedding": numpy.ndarray}, ...]
      - do not set policy["embedding"]

    Chunks are only computed for policies that do not already have a \"chunks\" field.
    """
    if not policies:
        return

    for policy in policies:
        # Skip policies that already have chunks.
        if isinstance(policy.get("chunks"), list) and len(policy.get("chunks", [])) > 0:
            continue

        content_text = str(policy.get("content", "")).strip()
        if not content_text:
            continue

        # Split content into chunks and compute an embedding per chunk.
        raw_chunks = chunk_text(content_text)
        policy["chunks"] = []
        for chunk_text_str in raw_chunks:
            embedding = get_embedding(chunk_text_str)
            policy["chunks"].append({"text": chunk_text_str, "embedding": embedding})

        # Build full searchable text and tokenize it once at startup.
        # Only compute tokens if they are not already present.
        if "tokens" not in policy:
            full_text = _build_policy_text(policy)
            policy["tokens"] = _tokenize(full_text)


def hybrid_search(
    query: str,
    policies: List[dict],
    keyword_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> List[SearchResult]:
    """
    Orchestrate a full hybrid search over the provided policies.

    Steps:
    1. Query-time embedding phase:
       - compute the query embedding once.
    2. Loop through policies and for each:
       - build policy text and tokens
       - compute keyword score
       - reuse the precomputed document embedding
       - compute cosine similarity with the query embedding
    3. Normalize keyword scores into [0, 1].
    4. Fuse scores:
           final_score = keyword_weight * normalized_kw + vector_weight * vec_array
    5. Sort by final_score descending.
    6. Return the top 5 results.

    Document chunk embeddings are assumed to have been precomputed by
    `initialize_embeddings(policies)` and stored on each policy as
    `policy[\"chunks\"]` (list of {\"text\", \"embedding\"} dicts).
    """
    query = query.strip()
    if not query or not policies:
        return []

    # 1. Query-time embedding phase: compute query embedding once.
    query_embedding = get_embedding(query)

    # Prepare storage for raw scores and intermediate data.
    raw_keyword_scores: List[int] = []
    vector_scores: List[float] = []
    contents: List[str] = []
    titles: List[str] = []
    owners: List[str] = []

    # Tokenize query once for keyword matching.
    query_tokens = _tokenize(query)

    # 2. Loop through each policy and compute scores.
    for policy in policies:
        policy_tokens = policy.get("tokens", [])

        # Keyword score based on simple term overlap.
        k_score = keyword_score(query_tokens, policy_tokens)
        raw_keyword_scores.append(k_score)

        # Vector score: max cosine similarity between query and any chunk embedding.
        # Track best-matching chunk text for use in the result.
        chunks = policy.get("chunks", [])
        max_similarity = 0.0
        best_chunk_text = ""
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            emb = chunk.get("embedding")
            if not isinstance(emb, np.ndarray) or query_embedding is None:
                continue
            similarity = cosine_similarity(query_embedding, emb)
            if similarity > max_similarity:
                max_similarity = similarity
                best_chunk_text = str(chunk.get("text", ""))
        v_score = max_similarity
        vector_scores.append(v_score)

        # Cache metadata for building results later (content = best-matching chunk).
        contents.append(best_chunk_text)
        titles.append(str(policy.get("title", "Untitled policy")))
        owners.append(str(policy.get("owner", "Unknown owner")))

    # 3. Normalize keyword scores into [0, 1].
    kw_array = np.asarray(raw_keyword_scores, dtype="float32")
    max_kw = float(kw_array.max()) if kw_array.size > 0 else 0.0
    if max_kw == 0.0:
        normalized_kw = np.zeros_like(kw_array, dtype="float32")
    else:
        normalized_kw = kw_array / max_kw

    vec_array = np.asarray(vector_scores, dtype="float32")

    # 4. Weighted fusion of keyword and vector scores.
    final_scores = keyword_weight * normalized_kw + vector_weight * vec_array

    # 5. Sort by final_score descending and take top 5.
    n = len(policies)
    if n == 0:
        return []

    indices = np.argsort(final_scores)[::-1][: min(5, n)]

    results: List[SearchResult] = []
    for idx in indices:
        i = int(idx)
        results.append(
            SearchResult(
                title=titles[i],
                owner=owners[i],
                keyword_score=float(normalized_kw[i]),
                vector_score=float(vec_array[i]),
                final_score=float(final_scores[i]),
                content=contents[i],
            )
        )

    return results

