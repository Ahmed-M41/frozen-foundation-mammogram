"""Stage 4a: build a FAISS index from GALLERY (train) embeddings and run kNN.

Index is built on demand (never persisted). Cosine path uses IndexFlatIP on
L2-normalized vectors; L2 path uses IndexFlatL2 on raw vectors. The gallery can
be filtered by view (CC / MLO / all) and dataset, enabling view-specific
retrieval ("all CC together", "all MLO together").
"""
import faiss
import numpy as np

import db


def _l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return x / norms


class RetrievalIndex:
    """kNN over gallery embeddings for one (embedder, view, distance)."""

    def __init__(self, train_ids, train_vecs, train_labels, distance="cosine"):
        self.distance = distance
        self.train_ids = np.asarray(train_ids)
        self.train_labels = np.asarray(train_labels)
        dim = train_vecs.shape[1]
        vecs = np.ascontiguousarray(train_vecs, dtype=np.float32)
        if distance == "cosine":
            vecs = _l2_normalize(vecs)
            self.index = faiss.IndexFlatIP(dim)
        elif distance == "l2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"unknown distance {distance}")
        self.index.add(vecs)

    def self_check(self):
        """Cosine self-similarity of the first gallery vector must be ~1.0."""
        if self.distance != "cosine" or self.index.ntotal == 0:
            return True
        v = self.index.reconstruct(0).reshape(1, -1)
        scores, _ = self.index.search(v, 1)
        return float(scores[0, 0]) > 0.99

    def search(self, query_vecs, k):
        """Return (scores[N,k], neighbor_sample_ids[N,k], neighbor_labels[N,k]).
        Cosine score is similarity; l2 score is squared L2 (smaller = closer)."""
        q = np.ascontiguousarray(query_vecs, dtype=np.float32)
        if self.distance == "cosine":
            q = _l2_normalize(q)
        scores, idx = self.index.search(q, k)
        nbr_ids = self.train_ids[idx]
        nbr_labels = self.train_labels[idx]
        return scores, nbr_ids, nbr_labels


def build_index(conn, embedder_name, distance, view=None, dataset=None,
                split="train"):
    """Build a RetrievalIndex from GALLERY rows. Returns None if no data.
    view=None means view-agnostic ('all' views) gallery."""
    ids, vecs, labels = db.load_embeddings(
        conn, embedder_name, split=split, view=view, dataset=dataset)
    if len(ids) == 0:
        return None
    return RetrievalIndex(ids, vecs, labels, distance=distance)
