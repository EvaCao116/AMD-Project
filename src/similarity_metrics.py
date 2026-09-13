# =========================================================================
# STEP 3: MEASURE SIMILARITY OF EMBEDDINGS
# =========================================================================
import re
import csv
import json
import math
from collections import Counter, defaultdict

class EmbeddingMetrics:

    def __init__(self):
        pass

def cosine_similarity_sparse(self, vec_a, vec_b):
    """
    Cosine similarity between two sparse vectors represented as
    {index: value} dicts. Base Python only (no numpy required).
    Returns a value between -1 and 1 (in practice 0 to 1 for
    frequency-based embeddings, since counts/tfidf are non-negative).
    """
    # Dot product: only need to iterate over the smaller dict
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(val * vec_b.get(idx, 0.0) for idx, val in vec_a.items())

    norm_a = math.sqrt(sum(val ** 2 for val in vec_a.values()))
    norm_b = math.sqrt(sum(val ** 2 for val in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity_dense(self, vec_a, vec_b):
    """
    Cosine similarity for dense vectors (e.g., numpy arrays or plain
    lists of floats) — used with the optional sentence-transformers path.
    Implemented in base Python so it works even without numpy.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def similarity_matrix(self, embeddings, sparse=True):
    """
    Computes the full pairwise similarity matrix for a list of embeddings.
    Returns a list of lists (n x n), where entry [i][j] is the cosine
    similarity between embedding i and embedding j.

    Note: for large corpora this is O(n^2) — fine for a sample of the
    dataset, but for the full dataset consider sampling or a vector
    index library instead.
    """
    sim_fn = cosine_similarity_sparse if sparse else cosine_similarity_dense
    n = len(embeddings)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sim = sim_fn(embeddings[i], embeddings[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix


def summarize_similarity(self, matrix, labels=None):
    """
    Prints summary statistics of the similarity matrix (excluding the
    diagonal, which is always 1.0 self-similarity) and returns the
    most similar and least similar off-diagonal pair.
    """
    n = len(matrix)
    off_diag_values = []
    max_pair, max_val = None, -2.0
    min_pair, min_val = None, 2.0

    for i in range(n):
        for j in range(i + 1, n):
            val = matrix[i][j]
            off_diag_values.append(val)
            if val > max_val:
                max_val, max_pair = val, (i, j)
            if val < min_val:
                min_val, min_pair = val, (i, j)

    if not off_diag_values:
        print("Not enough units to compare (need at least 2).")
        return None

    mean_sim = sum(off_diag_values) / len(off_diag_values)
    sorted_vals = sorted(off_diag_values)
    mid = len(sorted_vals) // 2
    median_sim = (
        sorted_vals[mid] if len(sorted_vals) % 2 == 1
        else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    )

    def label(idx):
        return labels[idx] if labels else f"unit_{idx}"

    print("\n--- Similarity Summary ---")
    print(f"Pairs compared: {len(off_diag_values)}")
    print(f"Mean similarity:   {mean_sim:.4f}")
    print(f"Median similarity: {median_sim:.4f}")
    print(f"Min similarity:    {min_val:.4f}  ({label(min_pair[0])} vs {label(min_pair[1])})")
    print(f"Max similarity:    {max_val:.4f}  ({label(max_pair[0])} vs {label(max_pair[1])})")

    return {
        "n_pairs": len(off_diag_values),
        "mean_similarity": mean_sim,
        "median_similarity": median_sim,
        "min_similarity": min_val,
        "min_pair": [label(min_pair[0]), label(min_pair[1])],
        "max_similarity": max_val,
        "max_pair": [label(max_pair[0]), label(max_pair[1])],
    }
