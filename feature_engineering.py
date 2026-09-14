"""Feature engineering for ADE Corpus V2 (AS01, Part A).

Reads the raw downloaded configs from data/ade_corpus_v2/ and writes
engineered versions with new derived columns to data/ade_corpus_v2/engineered/.

New features:
  - drug_ade_relation: drug_effect_char_distance (numeric), is_severe_effect (binary)
  - drug_dosage_relation: dose_value (numeric), dose_unit (categorical)
  - classification: text_length_words (numeric)
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
    r"(?P<value>\d+(?:\.\d+)?)"
    r"[\s-]*"
    r"(?P<unit>mg/m2|gm/m2|mg/kg|micro\s*g(?:/(?:kg|m2|g))?|mcg|mg|gm|g|ml|ui|iu|units?|u)\b",
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
        unit = re.sub(r"\s+", "", match.group("unit")).lower()
        if unit.startswith("gm"):
            unit = "g" + unit[2:]          # normalize "gm" -> "g", "gm/m2" -> "g/m2"
        return pd.Series({
            "dose_value": float(match.group("value")),
            "dose_unit": unit,
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


def build_feature_documentation_table() -> pd.DataFrame:
    """Structural metadata for each engineered feature.
    """
    rows = [
        {
            "new_feature": "text_length_words",
            "source_feature(s)": "text",
            "how_created": "Count of whitespace-separated tokens (str.split().str.len())",
            "data_type": "numeric (int)",
            "why_useful": "Cheap stand-in for text length, which may correlate with complexity or information density. Useful for stratification/QA variable downstream.",
        },
        {
            "new_feature": "drug_effect_char_distance",
            "source_feature(s)": "indexes.drug.end_char, indexes.effect.start_char",
            "how_created": "Absolute difference between the first drug span's end offset and the first effect span's start offset",
            "data_type": "numeric (float, nullable)",
            "why_useful": "Rough proxy for the proximity of the drug and effect mentions in the text, which may correlate with the likelihood of a true causal relationship.",
        },
        {
            "new_feature": "is_severe_effect",
            "source_feature(s)": "effect",
            "how_created": "Boolean flag: true if the effect text (lowercased) contains any of a fixed list of severity keywords",
            "data_type": "binary (bool)",
            "why_useful": "Collapses the effect text into a simple binary label for downstream analysis, e.g. to stratify or filter for severe vs. non-severe effects.",
        },
        {
            "new_feature": "dose_value",
            "source_feature(s)": "dosage",
            "how_created": "Regex-extracted leading numeric value paired with a recognized unit token; null if no recognized pattern matches",
            "data_type": "numeric (float, nullable)",
            "why_useful": "Turns the free-text dosage into a numeric value for downstream analysis, e.g. to stratify or filter by dose.",
        },
        {
            "new_feature": "dose_unit",
            "source_feature(s)": "dosage",
            "how_created": "Regex-extracted unit token accompanying the parsed dose value, normalized to lowercase (e.g. 'gm/m2' -> 'g/m2'); null if no recognized pattern matches",
            "data_type": "categorical (string, nullable)",
            "why_useful": "Provides the unit of the parsed dose value for downstream analysis, e.g. to stratify or filter by unit type.",
        },
    ]
    return pd.DataFrame(rows)


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

    feature_table = build_feature_documentation_table()
    feature_table.to_csv(OUT_DIR / "feature_documentation_table.csv", index=False)
    print("\n=== Feature documentation table ===")
    print(feature_table.to_string(index=False))

    print("\nEngineered files written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
