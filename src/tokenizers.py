# =========================================================================
# STEP 1: TOKENIZATION / CHUNKING STRATEGY
# =========================================================================
import re
import csv
import json
import math
from collections import Counter, defaultdict

class TokenizerManager:

    def __init__(self):
        pass

    def tokenize_words(self, text):
        """
        Simple base-Python word tokenizer.
        Lowercases, strips punctuation, splits on whitespace.
        This is a transparent, easy-to-explain tokenization strategy —
        good for a first experiment because every step is inspectable.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        tokens = text.split()
        return tokens


    def tokenize_sentences(self, text):
        """
        Splits text into sentences using simple punctuation-based rules,
        then word-tokenizes each sentence. Returns a list of token lists
        (one list per sentence).
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if s]
        return [tokenize_words(s) for s in sentences]


    def chunk_fixed_size(self, text, chunk_size=CHUNK_SIZE_WORDS):
        """
        Splits a document's tokens into fixed-size chunks of N words.
        This mimics how LLM pipelines often chunk long documents for
        embedding/retrieval (e.g., for RAG systems).
        Returns a list of token-list chunks.
        """
        tokens = tokenize_words(text)
        return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]


    def apply_tokenization(self, text, strategy):
        """
        Dispatches to the chosen tokenization/chunking strategy.
        Always returns a list of "units", where each unit is a list of tokens.
        (For "word" strategy, this is a list containing a single token list,
        so all strategies share a consistent output shape.)
        """
        if strategy == "word":
            return [tokenize_words(text)]
        elif strategy == "sentence":
            return tokenize_sentences(text)
        elif strategy == "fixed_chunk":
            return chunk_fixed_size(text)
        else:
            raise ValueError(f"Unknown CHUNKING_STRATEGY: {strategy}")
