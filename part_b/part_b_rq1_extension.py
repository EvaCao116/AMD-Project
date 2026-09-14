"""
AS01 Part C - RQ1 supporting experiment
"Does sentence length or drug-effect distance affect embedding
separability between ADE and non-ADE sentences?"

Reuses the same tokenizer/embedding model as Part B
(dmis-lab/biobert-base-cased-v1.1 / pritamdeka/S-BioBert-snli-multinli-stsb).

Experiment 1 (sentence length):
  - Classification config, n=300/class (same seed as Part B).
  - Split each class at the dataset median sentence length (17 words,
    from Part A section 6c) into "short" (<=17) and "long" (>17).
  - Compute within/across-class cosine similarity separately within
    the short bucket and within the long bucket. Compare separation
    gaps.

Experiment 2 (drug-effect distance):
  - drug_ade_relation config sentences (all ADE-positive) split at the
    dataset median drug_effect_char_distance (56 chars, from Part A
    section 6c) into "near" (<=56) and "far" (>56).
  - A fixed reference sample of non-ADE sentences (classification,
    label=0) is embedded once.
  - For each bucket, compute mean cosine similarity to the non-ADE
    reference sample (i.e. how distinguishable that bucket is from
    non-ADE text). Compare near vs far.

Output: part_b_rq1_results.json
"""

import json
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

RNG_SEED = 42
N_PER_CLASS = 300
LENGTH_MEDIAN = 17
DISTANCE_MEDIAN = 56

TOKENIZER_NAME = "dmis-lab/biobert-base-cased-v1.1"
EMBED_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"


def log(msg):
    print(f"[rq1] {msg}", file=sys.stderr, flush=True)


def within_across(embeddings, labels):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    assert len(classes) == 2
    c0, c1 = classes
    emb0, emb1 = embeddings[labels == c0], embeddings[labels == c1]

    sim00 = cosine_similarity(emb0)
    sim11 = cosine_similarity(emb1)
    sim01 = cosine_similarity(emb0, emb1)

    iu0 = np.triu_indices_from(sim00, k=1)
    iu1 = np.triu_indices_from(sim11, k=1)

    within0, within1, across = sim00[iu0], sim11[iu1], sim01.flatten()

    return {
        "n_class0": int(len(emb0)),
        "n_class1": int(len(emb1)),
        "within_class0_mean": float(np.mean(within0)) if len(within0) else None,
        "within_class1_mean": float(np.mean(within1)) if len(within1) else None,
        "across_class_mean": float(np.mean(across)),
        "separation_gap": (
            float((np.mean(within0) + np.mean(within1)) / 2 - np.mean(across))
            if len(within0) and len(within1) else None
        ),
    }


def main():
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    results = {
        "tokenizer": TOKENIZER_NAME,
        "embedding_model": EMBED_MODEL_NAME,
        "seed": RNG_SEED,
        "length_median_words": LENGTH_MEDIAN,
        "distance_median_chars": DISTANCE_MEDIAN,
    }

    log("loading tokenizer/model ...")
    AutoTokenizer.from_pretrained(TOKENIZER_NAME)  # warm cache, unused directly here
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # ---------------- Experiment 1: sentence length ----------------
    log("loading classification config ...")
    cls_ds = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_classification")["train"]
    cls_df = cls_ds.to_pandas()
    cls_df["text_length_words"] = cls_df["text"].str.split().str.len()

    pos = cls_df[cls_df["label"] == 1].sample(n=N_PER_CLASS, random_state=RNG_SEED)
    neg = cls_df[cls_df["label"] == 0].sample(n=N_PER_CLASS, random_state=RNG_SEED)
    cls_sample = pd.concat([pos, neg], ignore_index=True)

    log("embedding classification sample ...")
    cls_sample["embedding"] = list(
        model.encode(cls_sample["text"].tolist(), batch_size=32, show_progress_bar=False)
    )

    short = cls_sample[cls_sample["text_length_words"] <= LENGTH_MEDIAN]
    long = cls_sample[cls_sample["text_length_words"] > LENGTH_MEDIAN]

    log(f"short bucket n={len(short)}, long bucket n={len(long)}")

    results["length_experiment"] = {
        "short_le_17_words": within_across(
            np.vstack(short["embedding"].values), short["label"].tolist()
        ),
        "long_gt_17_words": within_across(
            np.vstack(long["embedding"].values), long["label"].tolist()
        ),
    }

    # ---------------- Experiment 2: drug-effect distance ----------------
    log("loading drug_ade_relation config ...")
    rel_ds = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")["train"]
    rel_df = rel_ds.to_pandas()

    def first_distance(row):
        d_end = row["indexes"]["drug"]["end_char"]
        e_start = row["indexes"]["effect"]["start_char"]
        if len(d_end) == 0 or len(e_start) == 0:
            return None
        return abs(d_end[0] - e_start[0])

    rel_df["drug_effect_char_distance"] = rel_df.apply(first_distance, axis=1)
    rel_df = rel_df.dropna(subset=["drug_effect_char_distance"])

    near = rel_df[rel_df["drug_effect_char_distance"] <= DISTANCE_MEDIAN]
    far = rel_df[rel_df["drug_effect_char_distance"] > DISTANCE_MEDIAN]

    n_bucket = min(200, len(near), len(far))
    near_sample = near.sample(n=n_bucket, random_state=RNG_SEED)
    far_sample = far.sample(n=n_bucket, random_state=RNG_SEED)

    log(f"near bucket n={len(near_sample)}, far bucket n={len(far_sample)} (of {len(near)}/{len(far)} available)")

    # fixed non-ADE reference sample, reusing the classification label=0 sample
    non_ade_ref = neg  # 300 non-ADE sentences already sampled above
    log("embedding near/far/non-ADE-reference sentences ...")
    near_emb = model.encode(near_sample["text"].tolist(), batch_size=32, show_progress_bar=False)
    far_emb = model.encode(far_sample["text"].tolist(), batch_size=32, show_progress_bar=False)
    non_ade_emb = np.vstack(
        cls_sample.set_index(cls_sample.index)[cls_sample["label"] == 0]["embedding"].values
    )

    def to_non_ade_similarity(bucket_emb):
        sim = cosine_similarity(bucket_emb, non_ade_emb)
        return {
            "n_bucket": int(bucket_emb.shape[0]),
            "n_reference_non_ade": int(non_ade_emb.shape[0]),
            "mean_similarity_to_non_ade": float(np.mean(sim)),
            "std_similarity_to_non_ade": float(np.std(sim)),
        }

    results["distance_experiment"] = {
        "near_le_56_chars": to_non_ade_similarity(near_emb),
        "far_gt_56_chars": to_non_ade_similarity(far_emb),
        "near_available": int(len(near)),
        "far_available": int(len(far)),
    }
    results["distance_experiment"]["gap_far_minus_near"] = (
        results["distance_experiment"]["far_gt_56_chars"]["mean_similarity_to_non_ade"]
        - results["distance_experiment"]["near_le_56_chars"]["mean_similarity_to_non_ade"]
    )

    with open("part_b_rq1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log("DONE")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
