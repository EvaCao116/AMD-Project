"""
AS01 Part B - Tokenization & Embedding Experiments
Dataset: ADE Corpus V2 (ade-benchmark-corpus/ade_corpus_v2)

Tokenizer:  dmis-lab/biobert-base-cased-v1.1        (biomedical domain BPE/WordPiece)
Embedding:  pritamdeka/S-BioBert-snli-multinli-stsb  (biomedical sentence-transformer,
            SBERT-style bi-encoder fine-tuned from BioBERT for sentence similarity)

Main experiment (classification config):
  - Sample N=300 per class (ADE-positive label=1, ADE-negative label=0) from the
    `Ade_corpus_v2_classification` config.
  - Tokenize with the BioBERT tokenizer (report avg token count, vocab behavior).
  - Embed each sentence with the biomedical sentence-transformer.
  - Compute cosine similarity within-class (pos-pos, neg-neg) and across-class (pos-neg).
  - PCA-project embeddings to 2D and plot, colored by label.

Extension (drug_ade_relation config):
  - Repeat the within/across-class cosine similarity check using `is_severe_effect`
    instead of the ADE label, to connect back to the ~7% severe-effect minority
    class-imbalance risk already flagged in Part A.

Outputs:
  - part_b_results.json   (all numeric results, for citing in the write-up)
  - part_b_pca_classification.png
  - part_b_pca_severity.png
"""

import json
import re
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 42
N_PER_CLASS = 300

TOKENIZER_NAME = "dmis-lab/biobert-base-cased-v1.1"
EMBED_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

SEVERE_KEYWORDS = [
    "death", "died", "fatal", "hospitalization", "hospitalized",
    "seizure", "anaphylaxis", "anaphylactic", "cardiac arrest",
    "coma", "respiratory failure", "renal failure", "liver failure",
    "stroke", "suicide", "arrest",
]


def log(msg):
    print(f"[part_b] {msg}", file=sys.stderr, flush=True)


def sample_classification(seed=RNG_SEED, n_per_class=N_PER_CLASS):
    log("loading Ade_corpus_v2_classification ...")
    ds = load_dataset(
        "ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_classification"
    )["train"]
    df = ds.to_pandas()
    log(f"loaded {len(df)} rows, label counts: {df['label'].value_counts().to_dict()}")

    pos = df[df["label"] == 1].sample(n=n_per_class, random_state=seed)
    neg = df[df["label"] == 0].sample(n=n_per_class, random_state=seed)
    sample = pd.concat([pos, neg], ignore_index=True)
    return sample


def is_severe(effect_text):
    if effect_text is None:
        return False
    t = str(effect_text).lower()
    return any(kw in t for kw in SEVERE_KEYWORDS)


def sample_severity(seed=RNG_SEED):
    log("loading Ade_corpus_v2_drug_ade_relation ...")
    ds = load_dataset(
        "ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation"
    )["train"]
    df = ds.to_pandas()

    df["is_severe_effect"] = df["effect"].apply(is_severe)
    counts = df["is_severe_effect"].value_counts().to_dict()
    log(f"loaded {len(df)} rows, is_severe_effect counts: {counts}")

    n_severe = int(df["is_severe_effect"].sum())
    n_take = min(150, n_severe)
    if n_take < 20:
        log(f"WARNING: only {n_severe} severe rows available, using all of them")
        n_take = n_severe

    severe = df[df["is_severe_effect"]].sample(n=n_take, random_state=seed)
    nonsevere = df[~df["is_severe_effect"]].sample(n=n_take, random_state=seed)
    sample = pd.concat([severe, nonsevere], ignore_index=True)
    return sample, counts


def tokenizer_stats(tokenizer, texts):
    lengths = [len(tokenizer.tokenize(t)) for t in texts]
    unk_id = tokenizer.unk_token_id
    unk_counts = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        unk_counts.append(sum(1 for i in ids if i == unk_id))
    return {
        "n_texts": len(texts),
        "mean_token_len": float(np.mean(lengths)),
        "median_token_len": float(np.median(lengths)),
        "min_token_len": int(np.min(lengths)),
        "max_token_len": int(np.max(lengths)),
        "mean_unk_per_text": float(np.mean(unk_counts)),
        "pct_texts_with_unk": float(np.mean([c > 0 for c in unk_counts]) * 100),
    }


def within_across_similarity(embeddings, labels):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    assert len(classes) == 2, "expected exactly 2 classes"
    c0, c1 = classes

    emb0 = embeddings[labels == c0]
    emb1 = embeddings[labels == c1]

    sim00 = cosine_similarity(emb0)
    sim11 = cosine_similarity(emb1)
    sim01 = cosine_similarity(emb0, emb1)

    iu0 = np.triu_indices_from(sim00, k=1)
    iu1 = np.triu_indices_from(sim11, k=1)

    within0 = sim00[iu0]
    within1 = sim11[iu1]
    across = sim01.flatten()

    return {
        f"within_class_{c0}_mean": float(np.mean(within0)),
        f"within_class_{c0}_std": float(np.std(within0)),
        f"within_class_{c1}_mean": float(np.mean(within1)),
        f"within_class_{c1}_std": float(np.std(within1)),
        "across_class_mean": float(np.mean(across)),
        "across_class_std": float(np.std(across)),
        "separation_gap": float(
            (np.mean(within0) + np.mean(within1)) / 2 - np.mean(across)
        ),
    }


def plot_pca(embeddings, labels, label_names, title, out_path):
    pca = PCA(n_components=2, random_state=RNG_SEED)
    coords = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    labels = np.asarray(labels)
    for cls in np.unique(labels):
        mask = labels == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=label_names.get(cls, str(cls)),
            alpha=0.6, s=25,
        )
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"explained_variance_ratio": explained.tolist()}


def main():
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    results = {
        "tokenizer": TOKENIZER_NAME,
        "embedding_model": EMBED_MODEL_NAME,
        "n_per_class": N_PER_CLASS,
        "seed": RNG_SEED,
    }

    log(f"loading tokenizer {TOKENIZER_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    log(f"loading embedding model {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # ---- Main experiment: ADE classification (label 0/1) ----
    cls_sample = sample_classification()
    texts = cls_sample["text"].tolist()
    labels = cls_sample["label"].tolist()

    log("computing tokenizer stats (classification sample) ...")
    results["tokenizer_stats_classification"] = tokenizer_stats(tokenizer, texts)

    log("embedding classification sample ...")
    cls_embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    log("computing within/across-class cosine similarity (ADE label) ...")
    results["similarity_classification"] = within_across_similarity(cls_embeddings, labels)

    log("running PCA + plotting (classification) ...")
    pca_info_cls = plot_pca(
        cls_embeddings, labels,
        label_names={0: "non-ADE (0)", 1: "ADE (1)"},
        title="ADE Corpus V2 classification embeddings (PCA)",
        out_path="part_b_pca_classification.png",
    )
    results["pca_classification"] = pca_info_cls

    # ---- Extension: severity (is_severe_effect) on drug_ade_relation ----
    sev_sample, sev_counts = sample_severity()
    sev_texts = sev_sample["text"].tolist()
    sev_labels = sev_sample["is_severe_effect"].astype(int).tolist()

    results["severity_full_config_counts"] = {str(k): int(v) for k, v in sev_counts.items()}
    results["severity_sample_size_per_class"] = int(len(sev_sample) / 2)

    log("embedding severity sample ...")
    sev_embeddings = model.encode(sev_texts, batch_size=32, show_progress_bar=False)

    log("computing within/across-class cosine similarity (is_severe_effect) ...")
    results["similarity_severity"] = within_across_similarity(sev_embeddings, sev_labels)

    log("running PCA + plotting (severity) ...")
    pca_info_sev = plot_pca(
        sev_embeddings, sev_labels,
        label_names={0: "non-severe effect", 1: "severe effect"},
        title="ADE drug_ade_relation: severe vs non-severe effect embeddings (PCA)",
        out_path="part_b_pca_severity.png",
    )
    results["pca_severity"] = pca_info_sev

    with open("part_b_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log("DONE. Results written to part_b_results.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
