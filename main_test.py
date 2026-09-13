#!/usr/bin/env python3
"""
main_test.py

Tests the project files against the generated test data.

Expected project layout:

    as01_v02/
      main_test.py
      src/
        tokenizers.py
        embedders_no_ai.py
        embedders_aimodel.py
        similarity_metrics.py
        embeddinggemma.py
      data/
        embedding_inputs.json
        retrieval_eval.json

Run normal base-Python tests:

    python3 main_test.py

Run optional real EmbeddingGemma test:

    python3 main_test.py --run-gemma --gemma-model google/embeddinggemma-300m

This script is intentionally defensive because some of your helper files define
functions outside their classes or reference globals. The test runner imports
them, patches those references at runtime, then verifies the pipeline works.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

TOKENIZERS_PY = SRC_DIR / "tokenizers.py"
EMBEDDERS_NO_AI_PY = SRC_DIR / "embedders_no_ai.py"
EMBEDDERS_AI_PY = SRC_DIR / "embedders_aimodel.py"
SIMILARITY_METRICS_PY = SRC_DIR / "similarity_metrics.py"
EMBEDDINGGEMMA_PY = SRC_DIR / "embeddinggemma.py"

RETRIEVAL_EVAL_JSON = DATA_DIR / "retrieval_eval.json"
EMBEDDING_INPUTS_JSON = DATA_DIR / "embedding_inputs.json"
REPORT_JSON = DATA_DIR / "main_test_report.json"


DOCUMENTS = [
    {
        "id": "doc_python_1",
        "topic": "python",
        "text": "Python is a popular programming language used for web development, automation, data science, and machine learning.",
    },
    {
        "id": "doc_python_2",
        "topic": "python",
        "text": "A Python virtual environment isolates project dependencies so different applications can use different package versions.",
    },
    {
        "id": "doc_llm_1",
        "topic": "llm",
        "text": "Large language models generate text by predicting tokens based on patterns learned from massive training datasets.",
    },
    {
        "id": "doc_llm_2",
        "topic": "llm",
        "text": "Embedding models convert text into numerical vectors that can be compared using cosine similarity.",
    },
    {
        "id": "doc_search_1",
        "topic": "semantic_search",
        "text": "Semantic search retrieves documents based on meaning rather than exact keyword matching.",
    },
    {
        "id": "doc_search_2",
        "topic": "semantic_search",
        "text": "Vector databases store embeddings and allow efficient nearest-neighbor search over large document collections.",
    },
    {
        "id": "doc_food_1",
        "topic": "food",
        "text": "Sourdough bread is made using a fermented starter that gives the loaf a tangy flavor and chewy texture.",
    },
    {
        "id": "doc_food_2",
        "topic": "food",
        "text": "Tomato soup pairs well with grilled cheese sandwiches, especially on a cold day.",
    },
    {
        "id": "doc_space_1",
        "topic": "space",
        "text": "Mars is often called the red planet because iron oxide on its surface gives it a reddish appearance.",
    },
    {
        "id": "doc_space_2",
        "topic": "space",
        "text": "A telescope collects light from distant objects and helps astronomers observe stars, planets, and galaxies.",
    },
    {
        "id": "doc_music_1",
        "topic": "music",
        "text": "A piano produces sound when hammers strike strings inside the instrument.",
    },
    {
        "id": "doc_music_2",
        "topic": "music",
        "text": "Jazz music often includes improvisation, syncopated rhythms, and extended harmonies.",
    },
]


QUERIES = [
    {
        "id": "q_python_env",
        "text": "How do I keep Python package dependencies separate for each project?",
        "expected_topic": "python",
        "expected_doc_ids": ["doc_python_2"],
    },
    {
        "id": "q_embeddings",
        "text": "What turns sentences into vectors for similarity search?",
        "expected_topic": "llm",
        "expected_doc_ids": ["doc_llm_2", "doc_search_2"],
    },
    {
        "id": "q_semantic_search",
        "text": "Find documents by meaning instead of matching exact words.",
        "expected_topic": "semantic_search",
        "expected_doc_ids": ["doc_search_1"],
    },
    {
        "id": "q_bread",
        "text": "Which bread uses a fermented starter and tastes tangy?",
        "expected_topic": "food",
        "expected_doc_ids": ["doc_food_1"],
    },
    {
        "id": "q_mars",
        "text": "Why does Mars look red?",
        "expected_topic": "space",
        "expected_doc_ids": ["doc_space_1"],
    },
    {
        "id": "q_telescope",
        "text": "What instrument helps people observe distant galaxies?",
        "expected_topic": "space",
        "expected_doc_ids": ["doc_space_2"],
    },
    {
        "id": "q_piano",
        "text": "Which instrument makes sound by hitting strings with hammers?",
        "expected_topic": "music",
        "expected_doc_ids": ["doc_music_1"],
    },
]


#def write_json(path: Path, data: Any) -> None:
   # path.parent.mkdir(parents=True, exist_ok=True)
    #path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def make_json_safe(value):
    """
    Convert non-JSON Python objects into JSON-safe representations.
    Handles modules, classes, functions, Path objects, and other objects.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, set):
        return [make_json_safe(item) for item in sorted(value, key=str)]

    if isinstance(value, dict):
        return {
            str(make_json_safe(key)): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, types.ModuleType):
        return {
            "type": "module",
            "name": getattr(value, "__name__", None),
            "file": getattr(value, "__file__", None),
        }

    if callable(value):
        return {
            "type": "callable",
            "name": getattr(value, "__name__", repr(value)),
        }

    return {
        "type": type(value).__name__,
        "repr": repr(value),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_data = make_json_safe(data)
    path.write_text(
        json.dumps(safe_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def ensure_test_data() -> None:
    """
    Creates test data if data/retrieval_eval.json or data/embedding_inputs.json
    do not already exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RETRIEVAL_EVAL_JSON.exists():
        write_json(
            RETRIEVAL_EVAL_JSON,
            {
                "documents": DOCUMENTS,
                "queries": QUERIES,
            },
        )

    if not EMBEDDING_INPUTS_JSON.exists():
        all_texts = [doc["text"] for doc in DOCUMENTS] + [query["text"] for query in QUERIES]
        write_json(
            EMBEDDING_INPUTS_JSON,
            {
                "input": all_texts,
            },
        )


def load_module_from_path(
    module_name: str,
    path: Path,
    injected_globals: dict[str, Any] | None = None,
):
    """
    Loads a Python file by path.

    This version allows injecting globals before exec. That matters because
    tokenizers.py uses CHUNK_SIZE_WORDS as a default value but may not define it.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    source = path.read_text(encoding="utf-8")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)

    if injected_globals:
        module.__dict__.update(injected_globals)

    sys.modules[module_name] = module
    code = compile(source, str(path), "exec")
    exec(code, module.__dict__)
    return module

def run_test(name: str, fn: Callable[[], Any], report: dict[str, Any]) -> Any:
    """
    Runs one test and records a JSON-safe version of the result.
    Returns the original result so the test script can keep using modules/objects.
    """
    print(f"\n[TEST] {name}")

    try:
        result = fn()
        report["tests"][name] = {
            "status": "PASS",
            "result": make_json_safe(result),
        }
        print(f"[PASS] {name}")
        return result

    except Exception as exc:
        report["tests"][name] = {
            "status": "FAIL",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[FAIL] {name}")
        print(str(exc))
        return None

'''
def run_test(name: str, fn: Callable[[], Any], report: dict[str, Any]) -> Any:
    """
    Runs one test and records result in report.
    """
    print(f"\n[TEST] {name}")

    try:
        result = fn()
        report["tests"][name] = {
            "status": "PASS",
            "result": result,
        }
        print(f"[PASS] {name}")
        return result
    except Exception as exc:
        report["tests"][name] = {
            "status": "FAIL",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[FAIL] {name}")
        print(str(exc))
        return None
'''

def patch_tokenizers_module(tok_mod: Any) -> Any:
    """
    tokenizers.py has methods inside TokenizerManager, but those methods call
    tokenize_words(), tokenize_sentences(), and chunk_fixed_size() as globals.

    This patch connects those global names to the class instance methods.
    """
    tokenizer = tok_mod.TokenizerManager()

    tok_mod.tokenize_words = tokenizer.tokenize_words
    tok_mod.tokenize_sentences = tokenizer.tokenize_sentences
    tok_mod.chunk_fixed_size = tokenizer.chunk_fixed_size

    return tokenizer

def patch_basic_embedding_module(emb_mod: Any) -> Any:
    """
    embedders_no_ai.py defines functions outside the class with a self parameter,
    then internally calls them without self.

    This patch creates module-level wrappers with the expected call signatures,
    including embed_corpus().
    """
    manager = emb_mod.BasicEmbeddingManager()

    original_build_vocabulary = emb_mod.build_vocabulary
    original_embed_bow = emb_mod.embed_bow
    original_compute_idf = emb_mod.compute_idf
    original_embed_tfidf = emb_mod.embed_tfidf
    original_embed_corpus = emb_mod.embed_corpus

    def build_vocabulary_wrapper(list_of_token_lists):
        return original_build_vocabulary(manager, list_of_token_lists)

    def embed_bow_wrapper(tokens, vocab):
        return original_embed_bow(manager, tokens, vocab)

    def compute_idf_wrapper(list_of_token_lists, vocab):
        return original_compute_idf(manager, list_of_token_lists, vocab)

    def embed_tfidf_wrapper(tokens, vocab, idf):
        return original_embed_tfidf(manager, tokens, vocab, idf)

    def embed_corpus_wrapper(list_of_token_lists, strategy):
        return original_embed_corpus(manager, list_of_token_lists, strategy)

    emb_mod.build_vocabulary = build_vocabulary_wrapper
    emb_mod.embed_bow = embed_bow_wrapper
    emb_mod.compute_idf = compute_idf_wrapper
    emb_mod.embed_tfidf = embed_tfidf_wrapper
    emb_mod.embed_corpus = embed_corpus_wrapper

    return manager

'''
def patch_basic_embedding_module(emb_mod: Any) -> Any:
    """
    embedders_no_ai.py defines functions outside the class with a self parameter,
    then internally calls them without self.

    This patch creates module-level wrappers with the expected call signatures.
    """
    manager = emb_mod.BasicEmbeddingManager()

    original_build_vocabulary = emb_mod.build_vocabulary
    original_embed_bow = emb_mod.embed_bow
    original_compute_idf = emb_mod.compute_idf
    original_embed_tfidf = emb_mod.embed_tfidf

    def build_vocabulary_wrapper(list_of_token_lists):
        return original_build_vocabulary(manager, list_of_token_lists)

    def embed_bow_wrapper(tokens, vocab):
        return original_embed_bow(manager, tokens, vocab)

    def compute_idf_wrapper(list_of_token_lists, vocab):
        return original_compute_idf(manager, list_of_token_lists, vocab)

    def embed_tfidf_wrapper(tokens, vocab, idf):
        return original_embed_tfidf(manager, tokens, vocab, idf)

    emb_mod.build_vocabulary = build_vocabulary_wrapper
    emb_mod.embed_bow = embed_bow_wrapper
    emb_mod.compute_idf = compute_idf_wrapper
    emb_mod.embed_tfidf = embed_tfidf_wrapper

    return manager
'''

def patch_similarity_module(metrics_mod: Any) -> Any:
    """
    similarity_metrics.py defines functions outside the class with a self
    parameter, then internally calls cosine_similarity_sparse/dense without self.

    This patch creates wrappers with the expected call signatures.
    """
    manager = metrics_mod.EmbeddingMetrics()

    original_sparse = metrics_mod.cosine_similarity_sparse
    original_dense = metrics_mod.cosine_similarity_dense

    def cosine_similarity_sparse_wrapper(vec_a, vec_b):
        return original_sparse(manager, vec_a, vec_b)

    def cosine_similarity_dense_wrapper(vec_a, vec_b):
        return original_dense(manager, vec_a, vec_b)

    metrics_mod.cosine_similarity_sparse = cosine_similarity_sparse_wrapper
    metrics_mod.cosine_similarity_dense = cosine_similarity_dense_wrapper

    return manager


def load_retrieval_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(RETRIEVAL_EVAL_JSON.read_text(encoding="utf-8"))
    return data["documents"], data["queries"]


def top_k_sparse(
    query_embedding: dict[int, float],
    doc_embeddings: list[dict[int, float]],
    docs: list[dict[str, Any]],
    metrics_mod: Any,
    k: int = 3,
) -> list[dict[str, Any]]:
    rows = []

    for doc, doc_embedding in zip(docs, doc_embeddings):
        score = metrics_mod.cosine_similarity_sparse(query_embedding, doc_embedding)
        rows.append(
            {
                "doc_id": doc["id"],
                "topic": doc["topic"],
                "score": score,
            }
        )

    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:k]


def dense_cosine(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test embedding project files on local test data.")
    parser.add_argument(
        "--run-gemma",
        action="store_true",
        help="Actually load and run the EmbeddingGemma model. This can take time.",
    )
    parser.add_argument(
        "--gemma-model",
        default="google/embeddinggemma-300m",
        help="Gemma model ID or local path. Default: google/embeddinggemma-300m",
    )
    parser.add_argument(
        "--gemma-device",
        default=None,
        help="Optional device for Gemma: cpu, cuda, cuda:0, mps.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=40,
        help="Injected CHUNK_SIZE_WORDS value for tokenizers.py. Default: 40.",
    )
    args = parser.parse_args()

    ensure_test_data()

    report: dict[str, Any] = {
        "root_dir": str(ROOT_DIR),
        "src_dir": str(SRC_DIR),
        "data_dir": str(DATA_DIR),
        "tests": {},
        "retrieval_results": {},
    }

    print("Project root:", ROOT_DIR)
    print("Source dir:  ", SRC_DIR)
    print("Data dir:    ", DATA_DIR)

    # ------------------------------------------------------------------
    # 1. Import all modules.
    # ------------------------------------------------------------------

    tok_mod = run_test(
        "import tokenizers.py",
        lambda: load_module_from_path(
            "project_tokenizers",
            TOKENIZERS_PY,
            injected_globals={"CHUNK_SIZE_WORDS": args.chunk_size},
        ),
        report,
    )

    emb_mod = run_test(
        "import embedders_no_ai.py",
        lambda: load_module_from_path("project_embedders_no_ai", EMBEDDERS_NO_AI_PY),
        report,
    )

    ai_mod = run_test(
        "import embedders_aimodel.py",
        lambda: load_module_from_path("project_embedders_aimodel", EMBEDDERS_AI_PY),
        report,
    )

    metrics_mod = run_test(
        "import similarity_metrics.py",
        lambda: load_module_from_path("project_similarity_metrics", SIMILARITY_METRICS_PY),
        report,
    )

    gemma_mod = run_test(
        "import embeddinggemma.py",
        lambda: load_module_from_path("project_embeddinggemma", EMBEDDINGGEMMA_PY),
        report,
    )

    if tok_mod is None or emb_mod is None or metrics_mod is None:
        write_json(REPORT_JSON, report)
        print("\nOne or more core modules failed to import.")
        print(f"Report written to: {REPORT_JSON}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Patch modules where needed.
    # ------------------------------------------------------------------

    tokenizer = run_test(
        "patch tokenizers.py runtime globals",
        lambda: patch_tokenizers_module(tok_mod),
        report,
    )

    basic_embedding_manager = run_test(
        "patch embedders_no_ai.py runtime globals",
        lambda: patch_basic_embedding_module(emb_mod),
        report,
    )

    metrics_manager = run_test(
        "patch similarity_metrics.py runtime globals",
        lambda: patch_similarity_module(metrics_mod),
        report,
    )

    if tokenizer is None or basic_embedding_manager is None or metrics_manager is None:
        write_json(REPORT_JSON, report)
        print("\nRuntime patching failed.")
        print(f"Report written to: {REPORT_JSON}")
        sys.exit(1)

    documents, queries = load_retrieval_data()
    doc_texts = [doc["text"] for doc in documents]
    query_texts = [query["text"] for query in queries]

    # ------------------------------------------------------------------
    # 3. Test tokenization.
    # ------------------------------------------------------------------

    def test_word_tokenization():
        sample = documents[0]["text"]
        tokens = tokenizer.tokenize_words(sample)

        if not tokens:
            raise AssertionError("Tokenizer returned no tokens.")

        return {
            "sample_text": sample,
            "first_15_tokens": tokens[:15],
            "token_count": len(tokens),
        }

    run_test("tokenize_words on sample document", test_word_tokenization, report)

    def test_sentence_tokenization():
        sample = "Python is useful. Embeddings are vectors! Does this split?"
        sentences = tokenizer.tokenize_sentences(sample)

        if len(sentences) < 2:
            raise AssertionError("Sentence tokenizer returned fewer than 2 sentence units.")

        return {
            "sample_text": sample,
            "sentence_units": sentences,
            "sentence_count": len(sentences),
        }

    run_test("tokenize_sentences on sample text", test_sentence_tokenization, report)

    def test_fixed_chunking():
        sample = documents[0]["text"]
        chunks = tokenizer.chunk_fixed_size(sample, chunk_size=5)

        if not chunks:
            raise AssertionError("Fixed chunking returned no chunks.")

        return {
            "sample_text": sample,
            "chunks": chunks,
            "chunk_count": len(chunks),
        }

    run_test("chunk_fixed_size on sample document", test_fixed_chunking, report)

    # ------------------------------------------------------------------
    # 4. Test base-Python BOW and TF-IDF embeddings.
    # ------------------------------------------------------------------

    doc_token_lists = [tokenizer.tokenize_words(text) for text in doc_texts]
    query_token_lists = [tokenizer.tokenize_words(text) for text in query_texts]
    all_token_lists = doc_token_lists + query_token_lists

    def test_bow_embeddings():
        embeddings, vocab, extra = emb_mod.embed_corpus(all_token_lists, strategy="bow")

        if len(embeddings) != len(all_token_lists):
            raise AssertionError("BOW embedding count does not match input count.")

        if not vocab:
            raise AssertionError("Vocabulary is empty.")

        return {
            "embedding_count": len(embeddings),
            "vocab_size": len(vocab),
            "first_embedding_nonzero_terms": len(embeddings[0]),
            "extra": extra,
        }

    run_test("BOW embed_corpus", test_bow_embeddings, report)

    def test_tfidf_embeddings():
        embeddings, vocab, extra = emb_mod.embed_corpus(all_token_lists, strategy="tfidf")

        if len(embeddings) != len(all_token_lists):
            raise AssertionError("TF-IDF embedding count does not match input count.")

        if "idf" not in extra:
            raise AssertionError("TF-IDF did not return IDF data in extra.")

        return {
            "embedding_count": len(embeddings),
            "vocab_size": len(vocab),
            "idf_count": len(extra["idf"]),
            "first_embedding_nonzero_terms": len(embeddings[0]),
        }

    tfidf_result = run_test("TF-IDF embed_corpus", test_tfidf_embeddings, report)

    # ------------------------------------------------------------------
    # 5. Test sparse similarity matrix.
    # ------------------------------------------------------------------

    def test_sparse_similarity_matrix():
        embeddings, vocab, extra = emb_mod.embed_corpus(doc_token_lists, strategy="tfidf")
        matrix = metrics_mod.similarity_matrix(metrics_manager, embeddings, sparse=True)

        if len(matrix) != len(documents):
            raise AssertionError("Similarity matrix row count is wrong.")

        if len(matrix[0]) != len(documents):
            raise AssertionError("Similarity matrix column count is wrong.")

        labels = [doc["id"] for doc in documents]
        summary = metrics_mod.summarize_similarity(metrics_manager, matrix, labels=labels)

        return {
            "matrix_size": [len(matrix), len(matrix[0])],
            "summary": summary,
        }

    run_test("sparse TF-IDF similarity_matrix", test_sparse_similarity_matrix, report)

    # ------------------------------------------------------------------
    # 6. Test retrieval with TF-IDF.
    # ------------------------------------------------------------------

    def test_sparse_retrieval():
        all_embeddings, vocab, extra = emb_mod.embed_corpus(all_token_lists, strategy="tfidf")

        doc_embeddings = all_embeddings[: len(documents)]
        query_embeddings = all_embeddings[len(documents):]

        retrieval_rows = []
        hits_at_1 = 0
        hits_at_3 = 0

        for query, query_embedding in zip(queries, query_embeddings):
            top3 = top_k_sparse(
                query_embedding=query_embedding,
                doc_embeddings=doc_embeddings,
                docs=documents,
                metrics_mod=metrics_mod,
                k=3,
            )

            expected = set(query["expected_doc_ids"])
            top1_ids = {top3[0]["doc_id"]}
            top3_ids = {row["doc_id"] for row in top3}

            hit1 = bool(expected & top1_ids)
            hit3 = bool(expected & top3_ids)

            if hit1:
                hits_at_1 += 1
            if hit3:
                hits_at_3 += 1

            retrieval_rows.append(
                {
                    "query_id": query["id"],
                    "query_text": query["text"],
                    "expected_doc_ids": query["expected_doc_ids"],
                    "top3": top3,
                    "hit_at_1": hit1,
                    "hit_at_3": hit3,
                }
            )

        result = {
            "query_count": len(queries),
            "hits_at_1": hits_at_1,
            "hits_at_3": hits_at_3,
            "accuracy_at_1": hits_at_1 / len(queries),
            "accuracy_at_3": hits_at_3 / len(queries),
            "rows": retrieval_rows,
        }

        report["retrieval_results"]["tfidf_sparse"] = result
        return result

    run_test("TF-IDF semantic-ish retrieval test", test_sparse_retrieval, report)

    # ------------------------------------------------------------------
    # 7. Test embedders_aimodel.py presence.
    # ------------------------------------------------------------------

    def test_ai_model_file_shape():
        if ai_mod is None:
            raise AssertionError("embedders_aimodel.py did not import.")

        if not hasattr(ai_mod, "EmbeddingAIManager"):
            raise AssertionError("EmbeddingAIManager class not found.")

        if not hasattr(ai_mod, "embed_corpus_sentence_transformers"):
            raise AssertionError("embed_corpus_sentence_transformers function not found.")

        return {
            "has_EmbeddingAIManager": True,
            "has_embed_corpus_sentence_transformers": True,
            "note": "Not loading external sentence-transformers model in default test.",
        }

    run_test("embedders_aimodel.py API shape", test_ai_model_file_shape, report)

    # ------------------------------------------------------------------
    # 8. Test embeddinggemma.py parsing helpers.
    # ------------------------------------------------------------------

    def test_embeddinggemma_parse_helpers():
        if gemma_mod is None:
            raise AssertionError("embeddinggemma.py did not import.")

        if not hasattr(gemma_mod, "parse_input_file"):
            raise AssertionError("embeddinggemma.py missing parse_input_file().")

        if not hasattr(gemma_mod, "parse_text_input"):
            raise AssertionError("embeddinggemma.py missing parse_text_input().")

        texts = gemma_mod.parse_input_file(str(EMBEDDING_INPUTS_JSON))

        if not texts:
            raise AssertionError("parse_input_file returned no texts.")

        one = gemma_mod.parse_text_input("hello", field_name="input")
        many = gemma_mod.parse_text_input(["hello", "world"], field_name="input")

        return {
            "parsed_text_count": len(texts),
            "first_text": texts[0],
            "parse_single": one,
            "parse_many": many,
        }

    run_test("embeddinggemma.py parse helpers", test_embeddinggemma_parse_helpers, report)

    # ------------------------------------------------------------------
    # 9. Optional actual Gemma embedding test.
    # ------------------------------------------------------------------

    if args.run_gemma:
        def test_real_embeddinggemma_model():
            if gemma_mod is None:
                raise AssertionError("embeddinggemma.py did not import.")

            model = gemma_mod.load_model(
                model_name_or_path=args.gemma_model,
                device=args.gemma_device,
                cache_dir=None,
                max_seq_length=None,
                local_files_only=False,
            )

            sample_texts = [
                documents[0]["text"],
                documents[1]["text"],
                queries[0]["text"],
            ]

            embeddings = gemma_mod.encode_texts(
                model=model,
                texts=sample_texts,
                mode="auto",
                normalize=True,
                batch_size=4,
                show_progress=False,
            )

            if len(embeddings) != len(sample_texts):
                raise AssertionError("Gemma embedding count mismatch.")

            dim = len(embeddings[0])
            sim_doc0_query = dense_cosine(embeddings[0], embeddings[2])
            sim_doc1_query = dense_cosine(embeddings[1], embeddings[2])

            return {
                "model": args.gemma_model,
                "text_count": len(sample_texts),
                "embedding_dim": dim,
                "similarity_doc_python_1_to_query": sim_doc0_query,
                "similarity_doc_python_2_to_query": sim_doc1_query,
                "note": "For q_python_env, doc_python_2 should often score higher than doc_python_1.",
            }

        run_test("REAL EmbeddingGemma model encode", test_real_embeddinggemma_model, report)
    else:
        report["tests"]["REAL EmbeddingGemma model encode"] = {
            "status": "SKIPPED",
            "reason": "Run with --run-gemma to load the actual model.",
        }
        print("\n[SKIP] REAL EmbeddingGemma model encode")
        print("       Use --run-gemma to run the real model test.")

    # ------------------------------------------------------------------
    # 10. Final report.
    # ------------------------------------------------------------------

    write_json(REPORT_JSON, report)

    passed = sum(1 for row in report["tests"].values() if row["status"] == "PASS")
    failed = sum(1 for row in report["tests"].values() if row["status"] == "FAIL")
    skipped = sum(1 for row in report["tests"].values() if row["status"] == "SKIPPED")

    print("\n============================================================")
    print("TEST SUMMARY")
    print("============================================================")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    print(f"Report:  {REPORT_JSON}")

    if failed:
        print("\nSome tests failed. Open the JSON report for tracebacks.")
        sys.exit(1)

    print("\nAll required tests passed.")


if __name__ == "__main__":
    main()
