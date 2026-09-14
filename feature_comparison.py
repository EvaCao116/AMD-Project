"""Comparison visualizations: engineered features vs. their raw source columns
(AS01, Part A - Feature Engineering).

Reads the raw downloaded configs and the engineered CSVs already produced by
download_ade_corpus.py / feature_engineering.py, and produces charts that put
each new feature side-by-side with the raw column(s) it was derived from:

  - Extraction coverage: what fraction of rows a parsed numeric feature
    actually recovered, vs. the always-populated raw text it came from.
  - Dosage text breakdown: how the raw `dosage` strings split into
    successfully-parsed / unrecognized-format / purely-qualitative buckets.
  - Cardinality reduction: unique-value counts for high-cardinality raw
    categoricals (drug, effect) vs. the low-cardinality engineered flag
    (is_severe_effect).
  - Validation check: engineered text_length_words vs. a fresh word count
    computed directly from the raw text column, to confirm the two agree.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW_DIR = Path("data/ade_corpus_v2")
ENGINEERED_DIR = RAW_DIR / "engineered"
OUT_DIR = RAW_DIR / "profiling" / "comparisons"


def load_raw(config: str, split: str = "train") -> pd.DataFrame:
    path = RAW_DIR / config / f"{split}.json"
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def load_engineered(name: str) -> pd.DataFrame:
    return pd.read_csv(ENGINEERED_DIR / f"{name}_engineered.csv")


def save_bar_from_dict(data: dict, title: str, ylabel: str, path: Path, log_scale: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(data.keys()), list(data.values()), edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    for i, (_, v) in enumerate(data.items()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def extraction_coverage_chart() -> None:
    ade_df = load_engineered("drug_ade_relation")
    dose_df = load_engineered("drug_dosage_relation")

    coverage = {
        "drug_effect_char_distance\n(from indexes)": round(
            100 * ade_df["drug_effect_char_distance"].notna().mean(), 1
        ),
        "dose_value\n(from dosage text)": round(
            100 * dose_df["dose_value"].notna().mean(), 1
        ),
    }
    save_bar_from_dict(
        coverage,
        "Extraction coverage: % of rows successfully parsed",
        "% of rows with a non-null value",
        OUT_DIR / "extraction_coverage_bar.png",
    )
    print("Extraction coverage:", coverage)


def dosage_bucket_breakdown() -> None:
    raw = load_raw("Ade_corpus_v2_drug_dosage_relation")
    dose_df = load_engineered("drug_dosage_relation")

    has_digit = raw["dosage"].str.contains(r"\d", regex=True)
    parsed = dose_df["dose_value"].notna()

    buckets = {
        "Parsed\n(numeric value)": int(parsed.sum()),
        "Has digit,\nunrecognized format": int((has_digit & ~parsed).sum()),
        "No digit\n(qualitative only)": int((~has_digit).sum()),
    }
    save_bar_from_dict(
        buckets,
        "Raw `dosage` text: how it breaks down (279 rows)",
        "row count",
        OUT_DIR / "dosage_bucket_breakdown_bar.png",
    )
    print("Dosage buckets:", buckets)


def cardinality_reduction_chart() -> None:
    ade_raw = load_raw("Ade_corpus_v2_drug_ade_relation")
    ade_eng = load_engineered("drug_ade_relation")

    cardinality = {
        "drug\n(raw)": ade_raw["drug"].nunique(),
        "effect\n(raw)": ade_raw["effect"].nunique(),
        "is_severe_effect\n(engineered)": ade_eng["is_severe_effect"].nunique(),
    }
    save_bar_from_dict(
        cardinality,
        "Unique value count: raw categoricals vs. engineered flag",
        "unique values (log scale)",
        OUT_DIR / "cardinality_reduction_bar.png",
        log_scale=True,
    )
    print("Cardinality:", cardinality)


def text_length_validation_scatter() -> None:
    cls_df = load_engineered("classification")
    recomputed = cls_df["text"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(recomputed, cls_df["text_length_words"], alpha=0.05, s=8)
    max_val = max(recomputed.max(), cls_df["text_length_words"].max())
    ax.plot([0, max_val], [0, max_val], color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("word count recomputed directly from `text`")
    ax.set_ylabel("engineered text_length_words")
    ax.set_title("Validation: engineered feature vs. raw text word count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "text_length_validation_scatter.png", dpi=150)
    plt.close(fig)

    mismatches = int((recomputed != cls_df["text_length_words"]).sum())
    print(f"text_length_words validation: {mismatches} mismatches out of {len(cls_df)} rows")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extraction_coverage_chart()
    dosage_bucket_breakdown()
    cardinality_reduction_chart()
    text_length_validation_scatter()
    print("\nComparison charts written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()