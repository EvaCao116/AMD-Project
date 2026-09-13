import re
import csv
import json
import math
from collections import Counter, defaultdict

class EmbeddingAIManager:

    def __init__(self):
        pass


def embed_corpus_sentence_transformers(self, texts, model_name="all-MiniLM-L6-v2"):
    """
    OPTIONAL: real pretrained embedding model.
    Requires: pip install sentence-transformers

    This produces dense semantic embeddings from a transformer model
    (trained on much more data than your project dataset), rather than
    the sparse frequency-based vectors above. Useful if you want to
    compare "hand-rolled" embeddings vs. a pretrained model's embeddings
    as part of your interpretation section.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings  # dense numpy array, shape (n_texts, embedding_dim)
