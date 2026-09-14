"""Descriptive statistics and data profiling for ADE Corpus V2 (AS01, Part A).

Reads the raw downloaded configs from data/ade_corpus_v2/ (see
download_ade_corpus.py) and the engineered configs from
data/ade_corpus_v2/engineered/ (see feature_engineering.py), then produces:

  - Per-config summary table (rows, columns, dtypes, missing values)
  - Per-numeric-feature stats table (min, max, mean, median, std, quartiles, range)
  - Per-categorical/text-feature stats table (unique count, top values, frequencies)
  - Histograms / bar charts / boxplots for the features called out in the
    assignment (label balance, text length, drug/effect distance, severe-effect
    flag, dose value, dose unit)

All tables are written as CSV and all charts as PNG under
data/ade_corpus_v2/profiling/, so they can be pasted directly into the
assignment memo. This script only computes statistics and renders charts;
the written interpretation (normal vs. non-normal, impact on model dev) is
left to the student, per the assignment's no-AI-for-report-writing policy.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW_DIR = Path("data/ade_corpus_v2")
ENGINEERED_DIR = RAW_DIR / "engineered"
OUT_DIR = RAW_DIR / "profiling"

# (label, raw_json_path, engineered_csv_path)
CONFIGS = [
    ("classification", RAW_DIR / "Ade_corpus_v2_classification" / "train.json",
     ENGINEERED_DIR / "classification_engineered.csv"),
    ("drug_ade_relation", RAW_DIR / "Ade_corpus_v2_drug_ade_relation" / "train.json",
     ENGINEERED_DIR / "drug_ade_relation_engineered.csv"),
    ("drug_dosage_relation", RAW_DIR / "Ade_corpus_v2_drug_dosage_relation" / "train.json",
     ENGINEERED_DIR / "drug_dosage_relation_engineered.csv"),
]

NUMERIC_FEATURES = {
    "classification": ["label", "text_length_words"],
    "drug_ade_relation": ["drug_effect_char_distance"],
    "drug_dosage_relation": ["dose_value"],
}

CATEGORICAL_FEATURES = {
    "classification": ["label"],
    "drug_ade_relation": ["drug", "effect", "is_severe_effect"],
    "drug_dosage_relation": ["drug", "dose_unit"],
}

TEXT_FEATURES = {
    "classification": ["text"],
    "drug_ade_relation": ["text"],
    "drug_dosage_relation": ["text", "dosage"],
}


def load_config(label: str, raw_json: Path, engineered_csv: Path) -> pd.DataFrame:
    """Prefer the engineered CSV (has derived features); fall back to raw JSON."""
    if engineered_csv.exists():
        return pd.read_csv(engineered_csv)
    if raw_json.exists():
        with open(raw_json) as f:
            return pd.DataFrame(json.load(f))
    raise FileNotFoundError(
        f"Neither {engineered_csv} nor {raw_json} exists for config '{label}'. "
        "Run download_ade_corpus.py and feature_engineering.py first."
    )


def summary_table(label: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "config": label,
            "feature": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(100 * df[col].isna().mean(), 2),
        })
    return pd.DataFrame(rows)


def numeric_stats_table(label: str, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(float)
        rows.append({
            "config": label,
            "feature": col,
            "count": int(s.count()),
            "min": s.min(),
            "max": s.max(),
            "mean": round(s.mean(), 3),
            "median": s.median(),
            "std": round(s.std(), 3),
            "q1": s.quantile(0.25),
            "q3": s.quantile(0.75),
            "range": s.max() - s.min(),
        })
    return pd.DataFrame(rows)


def categorical_stats_table(label: str, df: pd.DataFrame, cols: list[str], top_n: int = 10) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        top = vc.head(top_n)
        rows.append({
            "config": label,
            "feature": col,
            "unique_values": int(df[col].nunique(dropna=True)),
            "top_values": "; ".join(f"{k}={v}" for k, v in top.items()),
        })
    return pd.DataFrame(rows)


def text_length_table(label: str, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        lengths = df[col].dropna().astype(str).str.split().str.len()
        rows.append({
            "config": label,
            "feature": col,
            "count": int(lengths.count()),
            "min_words": lengths.min(),
            "max_words": lengths.max(),
            "mean_words": round(lengths.mean(), 2),
            "median_words": lengths.median(),
        })
    return pd.DataFrame(rows)


def save_hist(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    series.dropna().plot(kind="hist", bins=bins, ax=ax, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_boxplot(series: pd.Series, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(series.dropna().astype(float), orientation="vertical")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_bar(series: pd.Series, title: str, xlabel: str, path: Path, top_n: int = 15) -> None:
    counts = series.value_counts(dropna=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_charts(dfs: dict[str, pd.DataFrame]) -> None:
    cls_df = dfs["classification"]
    ade_df = dfs["drug_ade_relation"]
    dose_df = dfs["drug_dosage_relation"]

    if "label" in cls_df.columns:
        save_bar(cls_df["label"], "Classification label distribution", "label",
                 OUT_DIR / "classification_label_bar.png")

    if "text_length_words" in cls_df.columns:
        save_hist(cls_df["text_length_words"], "Sentence length (words) - classification",
                   "words", OUT_DIR / "classification_text_length_hist.png")
        save_boxplot(cls_df["text_length_words"], "Sentence length (words) - classification",
                     "words", OUT_DIR / "classification_text_length_box.png")

    if "drug_effect_char_distance" in ade_df.columns:
        save_hist(ade_df["drug_effect_char_distance"], "Drug-effect char distance",
                   "characters", OUT_DIR / "drug_effect_char_distance_hist.png")
        save_boxplot(ade_df["drug_effect_char_distance"], "Drug-effect char distance",
                     "characters", OUT_DIR / "drug_effect_char_distance_box.png")

    if "is_severe_effect" in ade_df.columns:
        save_bar(ade_df["is_severe_effect"], "Severe-effect flag distribution",
                 "is_severe_effect", OUT_DIR / "is_severe_effect_bar.png")

    if "dose_value" in dose_df.columns:
        save_hist(dose_df["dose_value"], "Parsed dose value", "dose_value",
                   OUT_DIR / "dose_value_hist.png")
        save_boxplot(dose_df["dose_value"], "Parsed dose value", "dose_value",
                     OUT_DIR / "dose_value_box.png")

    if "dose_unit" in dose_df.columns:
        save_bar(dose_df["dose_unit"], "Parsed dose unit (top values)",
                 "dose_unit", OUT_DIR / "dose_unit_bar.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    summaries, numerics, categoricals, textlens = [], [], [], []

    for label, raw_json, engineered_csv in CONFIGS:
        df = load_config(label, raw_json, engineered_csv)
        dfs[label] = df

        print(f"\n=== {label} ===")
        print(f"observations: {len(df)}, features: {df.shape[1]}")

        summaries.append(summary_table(label, df))
        numerics.append(numeric_stats_table(label, df, NUMERIC_FEATURES.get(label, [])))
        categoricals.append(categorical_stats_table(label, df, CATEGORICAL_FEATURES.get(label, [])))
        textlens.append(text_length_table(label, df, TEXT_FEATURES.get(label, [])))

    summary_df = pd.concat(summaries, ignore_index=True)
    numeric_df = pd.concat(numerics, ignore_index=True)
    categorical_df = pd.concat(categoricals, ignore_index=True)
    textlen_df = pd.concat(textlens, ignore_index=True)

    summary_df.to_csv(OUT_DIR / "summary_missing_dtypes.csv", index=False)
    numeric_df.to_csv(OUT_DIR / "numeric_feature_stats.csv", index=False)
    categorical_df.to_csv(OUT_DIR / "categorical_feature_stats.csv", index=False)
    textlen_df.to_csv(OUT_DIR / "text_feature_word_length_stats.csv", index=False)

    make_charts(dfs)

    print("\n--- Summary (obs/features/missing/dtypes) ---")
    print(summary_df.to_string(index=False))
    print("\n--- Numeric feature stats ---")
    print(numeric_df.to_string(index=False))
    print("\n--- Categorical feature stats ---")
    print(categorical_df.to_string(index=False))
    print("\n--- Text feature word-length stats ---")
    print(textlen_df.to_string(index=False))

    print("\nTables and charts written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
