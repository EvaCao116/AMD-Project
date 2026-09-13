# =========================================================================
# STEP 2: EMBEDDING MODEL — create embeddings from your tokens
# =========================================================================
import re
import csv
import json
import math
from collections import Counter, defaultdict

class BasicEmbeddingManager:

    def __init__(self):
        pass

def build_vocabulary(self, list_of_token_lists):
    """Builds a sorted vocabulary (unique token -> index) from a corpus."""
    vocab = set()
    for tokens in list_of_token_lists:
        vocab.update(tokens)
    vocab = sorted(vocab)
    return {token: idx for idx, token in enumerate(vocab)}


def embed_bow(self, tokens, vocab):
    """
    Bag-of-words term-frequency embedding.
    Returns a dict {vocab_index: count} (sparse representation),
    since most documents only use a small fraction of the vocabulary.
    """
    vector = defaultdict(int)
    for tok in tokens:
        if tok in vocab:
            vector[vocab[tok]] += 1
    return dict(vector)


def compute_idf(self, list_of_token_lists, vocab):
    """
    Computes inverse-document-frequency for TF-IDF embeddings.
    idf(term) = log(N_docs / (1 + doc_count_containing_term))
    """
    n_docs = len(list_of_token_lists)
    doc_freq = Counter()
    for tokens in list_of_token_lists:
        unique_tokens = set(tokens)
        for tok in unique_tokens:
            doc_freq[tok] += 1

    idf = {}
    for tok, idx in vocab.items():
        df = doc_freq.get(tok, 0)
        idf[idx] = math.log(n_docs / (1 + df)) + 1  # +1 smoothing
    return idf


def embed_tfidf(self,tokens, vocab, idf):
    """
    TF-IDF embedding: term frequency weighted by inverse document frequency.
    Downweights very common tokens (like "the", "and") and upweights
    tokens that are distinctive to a given document — generally a
    stronger semantic signal than raw bag-of-words counts.
    """
    tf = embed_bow(tokens, vocab)
    vector = {idx: count * idf[idx] for idx, count in tf.items()}
    return vector


def embed_corpus(self, list_of_token_lists, strategy):
    """
    Creates embeddings for every unit (sentence/chunk/document) in the
    corpus, using the chosen base-Python embedding strategy.
    Returns (embeddings, vocab, extra) where extra holds strategy-specific
    artifacts (e.g., the idf dict for tfidf).
    """
    vocab = build_vocabulary(list_of_token_lists)

    if strategy == "bow":
        embeddings = [embed_bow(tokens, vocab) for tokens in list_of_token_lists]
        return embeddings, vocab, {}

    elif strategy == "tfidf":
        idf = compute_idf(list_of_token_lists, vocab)
        embeddings = [embed_tfidf(tokens, vocab, idf) for tokens in list_of_token_lists]
        return embeddings, vocab, {"idf": idf}

    else:
        raise ValueError(
            f"Unknown EMBEDDING_STRATEGY: {strategy}. "
            "Use 'bow', 'tfidf', or see embed_corpus_sentence_transformers() "
            "for the optional pretrained-model path."
        )
