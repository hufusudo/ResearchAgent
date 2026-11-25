"""Hybrid RAG index combining Chroma dense vector search, BM25 sparse search, and RRF."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Literal

import chromadb
from rank_bm25 import BM25Okapi

from core.llm.embeddings import EmbedderProtocol
from core.schemas import RetrievedChunk, TextChunk


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for BM25 indexing and querying."""
    return re.findall(r"[a-z0-9]+", text.lower())


_CHROMA_INIT_LOCK = threading.Lock()


class HybridRAGIndex:
    """In-process + optionally persisted Hybrid RAG index over TextChunk objects."""

    def __init__(
        self,
        collection_name: str,
        embedder: EmbedderProtocol,
        persist_dir: Path | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedder = embedder
        self.persist_dir = persist_dir

        with _CHROMA_INIT_LOCK:
            if persist_dir is not None:
                persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(persist_dir))
            else:
                self._client = chromadb.EphemeralClient()

            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        self._chunks_by_id: dict[str, TextChunk] = {}
        self._ordered_chunks: list[TextChunk] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.RLock()

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks."""
        with self._lock:
            return len(self._ordered_chunks)

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        """Embed, upsert into Chroma, and rebuild the BM25 sparse index."""
        if not chunks:
            return

        new_or_updated = list(chunks)
        texts = [c.text for c in new_or_updated]
        embeddings = self.embedder.embed_documents(texts)
        metadatas = [
            {
                "paper_id": c.paper_id,
                "title": c.title,
                "year": c.year,
                "url": c.url,
                "section": c.section,
                "page": c.page,
            }
            for c in new_or_updated
        ]

        with self._lock:
            for chunk in new_or_updated:
                self._chunks_by_id[chunk.chunk_id] = chunk
            self._ordered_chunks = list(self._chunks_by_id.values())

            self._collection.upsert(
                ids=[c.chunk_id for c in new_or_updated],
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            tokenized_corpus = [
                tokenize(f"{c.title} {c.section} {c.text}") or ["empty"]
                for c in self._ordered_chunks
            ]
            self._bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 8,
        rrf_k: int = 60,
        mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
        max_per_paper: int | None = None,
    ) -> list[RetrievedChunk]:
        """Search indexed chunks using hybrid RRF, dense-only, or BM25-only scoring."""
        if top_k <= 0:
            return []

        with self._lock:
            if not self._ordered_chunks:
                return []
            n_docs = len(self._ordered_chunks)

        query_embedding = self.embedder.embed_query(query) if mode in ("hybrid", "dense") else None

        with self._lock:
            if not self._ordered_chunks:
                return []
            n_docs = len(self._ordered_chunks)
            if max_per_paper is not None and max_per_paper > 0:
                candidate_k = min(n_docs, max(top_k * 5, 40))
            else:
                candidate_k = min(n_docs, max(top_k * 3, 20))

            dense_scores: dict[str, float] = {}
            dense_ranks: dict[str, int] = {}
            if mode in ("hybrid", "dense") and query_embedding is not None:
                dense_res = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=candidate_k,
                    include=["distances"],
                )
                ids_row = (dense_res.get("ids") or [[]])[0]
                dist_row = (dense_res.get("distances") or [[]])[0]
                dense_pairs: list[tuple[str, float]] = []
                for cid, dist in zip(ids_row, dist_row):
                    if cid in self._chunks_by_id:
                        dense_pairs.append((cid, 1.0 - float(dist)))
                dense_pairs.sort(key=lambda item: item[1], reverse=True)
                for rank, (cid, score) in enumerate(dense_pairs, start=1):
                    dense_scores[cid] = score
                    dense_ranks[cid] = rank

            bm25_scores: dict[str, float] = {}
            bm25_ranks: dict[str, int] = {}
            if mode in ("hybrid", "bm25") and self._bm25 is not None:
                q_tokens = tokenize(query)
                if q_tokens:
                    raw_scores = self._bm25.get_scores(q_tokens)
                    scored_chunks = [
                        (chunk.chunk_id, float(score))
                        for chunk, score in zip(self._ordered_chunks, raw_scores)
                    ]
                else:
                    scored_chunks = [(chunk.chunk_id, 0.0) for chunk in self._ordered_chunks]

                scored_chunks.sort(key=lambda item: item[1], reverse=True)
                for cid, score in scored_chunks:
                    bm25_scores[cid] = score

                positive_rank = 0
                for cid, score in scored_chunks[:candidate_k]:
                    if score > 0.0 or mode == "bm25":
                        positive_rank += 1
                        bm25_ranks[cid] = positive_rank

            if mode == "dense":
                candidate_ids = list(dense_ranks.keys())
            elif mode == "bm25":
                candidate_ids = list(bm25_ranks.keys())
            else:
                candidate_ids = list(dict.fromkeys([*dense_ranks.keys(), *bm25_ranks.keys()]))

            results: list[RetrievedChunk] = []
            for cid in candidate_ids:
                d_rank = dense_ranks.get(cid)
                b_rank = bm25_ranks.get(cid)
                if mode == "dense":
                    rrf = 1.0 / (rrf_k + d_rank) if d_rank is not None else 0.0
                elif mode == "bm25":
                    rrf = 1.0 / (rrf_k + b_rank) if b_rank is not None else 0.0
                else:
                    rrf = (
                        (1.0 / (rrf_k + d_rank) if d_rank is not None else 0.0)
                        + (1.0 / (rrf_k + b_rank) if b_rank is not None else 0.0)
                    )

                results.append(
                    RetrievedChunk(
                        chunk=self._chunks_by_id[cid],
                        dense_score=round(dense_scores.get(cid, 0.0), 6),
                        bm25_score=round(bm25_scores.get(cid, 0.0), 6),
                        rrf_score=round(rrf, 6),
                    )
                )

        if mode == "dense":
            results.sort(key=lambda r: (r.dense_score, r.rrf_score), reverse=True)
        elif mode == "bm25":
            results.sort(key=lambda r: (r.bm25_score, r.rrf_score), reverse=True)
        else:
            results.sort(key=lambda r: (r.rrf_score, r.dense_score, r.bm25_score), reverse=True)

        if max_per_paper is None or max_per_paper <= 0:
            return results[:top_k]

        selected: list[RetrievedChunk] = []
        overflow: list[RetrievedChunk] = []
        paper_counts: dict[str, int] = {}
        for item in results:
            pid = item.chunk.paper_id
            if paper_counts.get(pid, 0) < max_per_paper:
                selected.append(item)
                paper_counts[pid] = paper_counts.get(pid, 0) + 1
                if len(selected) >= top_k:
                    break
            else:
                overflow.append(item)

        if len(selected) < top_k:
            selected.extend(overflow[: top_k - len(selected)])

        return selected[:top_k]
