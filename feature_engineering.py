"""Feature engineering for ADE Corpus V2 (AS01, Part A).

Reads the raw downloaded configs from data/ade_corpus_v2/ and writes
engineered versions with new derived columns to data/ade_corpus_v2/engineered/.

New features:
  - drug_ade_relation: drug_effect_char_distance (numeric), is_severe_effect (binary)
  - drug_dosage_relation: dose_value (numeric), dose_unit (categorical)
  - classification: text_length_words (numeric)

This script only computes the features and prints descriptive stats.
Interpretation / write-up for the assignment memo is left to the student.
"""

import json
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/ade_corpus_v2")
OUT_DIR = RAW_DIR / "engineered"

SEVERE_KEYWORDS = [
    "death", "died", "fatal", "fatality",
    "hospitalization", "hospitalisation", "hospitalized", "hospitalised",
    "cardiac arrest", "respiratory failure", "respiratory arrest",
    "coma", "seizure", "anaphylaxis", "anaphylactic",
    "renal failure", "liver failure", "hepatic failure",
    "shock", "arrhythmia", "stroke",
]

DOSAGE_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg/m2|mg/kg|mcg|mg|g|ml|iu|units?)",
    flags=re.IGNORECASE,
)


def load_json(config: str, split: str = "train") -> pd.DataFrame:
    path = RAW_DIR / config / f"{split}.json"
    with open(path) as f:
        records = json.load(f)
    return pd.DataFrame(records)


def engineer_classification() -> pd.DataFrame:
    df = load_json("Ade_corpus_v2_classification")
    df["text_length_words"] = df["text"].str.split().str.len()
    return df


def engineer_drug_ade_relation() -> pd.DataFrame:
    df = load_json("Ade_corpus_v2_drug_ade_relation")

    def char_distance(idx: dict):
        drug_ends = idx["drug"]["end_char"]
        effect_starts = idx["effect"]["start_char"]
        if len(drug_ends) == 0 or len(effect_starts) == 0:
            return None
        return abs(int(effect_starts[0]) - int(drug_ends[0]))

    df["drug_effect_char_distance"] = df["indexes"].apply(char_distance)
    df["is_severe_effect"] = df["effect"].str.lower().apply(
        lambda e: any(kw in e for kw in SEVERE_KEYWORDS)
    )
    return df


def engineer_drug_dosage_relation() -> pd.DataFrame:
    df = load_json("Ade_corpus_v2_drug_dosage_relation")

    def parse_dose(dosage_text: str):
        match = DOSAGE_PATTERN.search(dosage_text)
        if not match:
            return pd.Series({"dose_value": None, "dose_unit": None})
        return pd.Series({
            "dose_value": float(match.group("value")),
            "dose_unit": match.group("unit").lower(),
        })

    parsed = df["dosage"].apply(parse_dose)
    return pd.concat([df, parsed], axis=1)


def print_summary(name: str, df: pd.DataFrame, new_cols: list[str]) -> None:
    print(f"\n=== {name} ({len(df)} rows) ===")
    for col in new_cols:
        print(f"\n-- {col} --")
        if pd.api.types.is_numeric_dtype(df[col]):
            print(df[col].describe())
            print("missing:", df[col].isna().sum())
        else:
            print(df[col].value_counts(dropna=False))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cls_df = engineer_classification()
    ade_df = engineer_drug_ade_relation()
    dose_df = engineer_drug_dosage_relation()

    cls_df.drop(columns=["indexes"], errors="ignore").to_csv(
        OUT_DIR / "classification_engineered.csv", index=False
    )
    ade_df.drop(columns=["indexes"]).to_csv(
        OUT_DIR / "drug_ade_relation_engineered.csv", index=False
    )
    dose_df.drop(columns=["indexes"]).to_csv(
        OUT_DIR / "drug_dosage_relation_engineered.csv", index=False
    )

    print_summary("classification", cls_df, ["text_length_words"])
    print_summary("drug_ade_relation", ade_df, ["drug_effect_char_distance", "is_severe_effect"])
    print_summary("drug_dosage_relation", dose_df, ["dose_value", "dose_unit"])

    print("\nEngineered files written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
