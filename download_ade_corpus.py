"""Download ADE Corpus V2 (all three configs) from Hugging Face and save locally.

Usage:
    python3 download_ade_corpus.py

Saves raw splits as CSV and JSON under data/ade_corpus_v2/<config_name>/.
"""

from pathlib import Path

from datasets import load_dataset

CONFIGS = [
    "Ade_corpus_v2_classification",
    "Ade_corpus_v2_drug_ade_relation",
    "Ade_corpus_v2_drug_dosage_relation",
]

REPO_ID = "ade-benchmark-corpus/ade_corpus_v2"
OUT_DIR = Path("data/ade_corpus_v2")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for config in CONFIGS:
        print(f"Downloading config: {config}")
        ds = load_dataset(REPO_ID, config)

        config_dir = OUT_DIR / config
        config_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in ds.items():
            df = split_data.to_pandas()

            csv_path = config_dir / f"{split_name}.csv"
            json_path = config_dir / f"{split_name}.json"

            df.to_csv(csv_path, index=False)
            df.to_json(json_path, orient="records", indent=2)

            print(f"  {split_name}: {len(df)} rows -> {csv_path}")

    print("\nDone. Data saved under:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
