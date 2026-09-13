# Assignment 1: Datasheets, Tokenization, Embeddings, and Knowledge Representation

Course: **95864 AI Model Development**  
Assignment: **AS01 — Datasheets for AI Model Development & Experimenting with Knowledge Representation**

This repository contains Python code and test data for Assignment 1. The assignment explores how dataset properties, feature engineering, tokenization, embeddings, and similarity metrics influence knowledge representation and downstream AI model behavior.

> **Course policy note:** The assignment instructions prohibit using AI to write, format, or edit the final assignment report/memo. This README is intended as code/project documentation only. If AI assistance was used to generate code or documentation, disclose or cite that use according to the assignment policy.

---

## Table of Contents

1. [Assignment Purpose](#assignment-purpose)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Downloading Hugging Face Models with `downloader.py`](#downloading-hugging-face-models-with-downloaderpy)
5. [Create Test Data](#create-test-data)
6. [Run the Main Test Script](#run-the-main-test-script)
7. [Run EmbeddingGemma from the Command Line](#run-embeddinggemma-from-the-command-line)
8. [Run EmbeddingGemma as a Local HTTP Server](#run-embeddinggemma-as-a-local-http-server)
9. [Assignment Part A: Dataset Datasheet](#assignment-part-a-dataset-datasheet)
10. [Assignment Part B: Tokenization and Embedding Experiments](#assignment-part-b-tokenization-and-embedding-experiments)
11. [Assignment Part C: Research Plan and Analysis Memo](#assignment-part-c-research-plan-and-analysis-memo)
12. [Reproducibility Checklist](#reproducibility-checklist)
13. [Troubleshooting](#troubleshooting)
14. [Submission Reminder](#submission-reminder)
15. [References](#references)

---

## Assignment Purpose

The goal of this assignment is to practice early-stage AI model development methods, including:

- Selecting and documenting a dataset for a future LLM training or fine-tuning project.
- Creating a dataset datasheet/codebook.
- Profiling dataset features with descriptive statistics.
- Engineering new features from existing features.
- Applying tokenization and chunking strategies.
- Creating embeddings from tokenized text.
- Measuring similarity between embeddings.
- Interpreting how data representation choices can influence:
  - Model performance
  - Generalization
  - Bias and representation
  - Dataset quality
  - Future training and fine-tuning decisions

The broader goal is to understand how data and feature choices shape AI model behavior.

---

## Repository Structure

Recommended project layout:

```text
as01_v02/
├── README.md
├── downloader.py
├── main_test.py
├── make_test_data.py
├── data/
│   ├── embedding_inputs.json
│   ├── embedding_lines.txt
│   ├── openai_embeddings_request.json
│   ├── embed_request.json
│   ├── retrieval_eval.json
│   ├── embeddings_output.json
│   └── main_test_report.json
├── src/
│   ├── tokenizers.py
│   ├── embedders_no_ai.py
│   ├── embedders_aimodel.py
│   ├── similarity_metrics.py
│   └── embeddinggemma.py
└── models/
    ├── embedders/
    │   └── google/
    │       └── embeddinggemma-300m/
    └── llms/
        └── meta/
            └── Llama-3.1-8B-Instruct/
```

---

## Main Files

| File | Purpose |
|---|---|
| `src/tokenizers.py` | Contains simple tokenization and chunking methods. |
| `src/embedders_no_ai.py` | Contains base-Python Bag-of-Words and TF-IDF embedding methods. |
| `src/embedders_aimodel.py` | Contains optional SentenceTransformers-based embedding helper. |
| `src/similarity_metrics.py` | Contains cosine similarity methods for sparse and dense embeddings. |
| `src/embeddinggemma.py` | Runs Google EmbeddingGemma locally using `sentence-transformers`; includes CLI and HTTP server modes without FastAPI/Uvicorn. |
| `make_test_data.py` | Creates a small test dataset for checking tokenization, embeddings, and retrieval behavior. |
| `main_test.py` | Runs tests across the project files using the generated test data. |
| `downloader.py` | Downloads Hugging Face models locally using a token stored in `.env`. |

---

## Environment Setup

### Python Version

Recommended:

```bash
python3 --version
```

Use Python **3.10, 3.11, or 3.12** if possible. Some machine learning packages may not fully support very new Python versions.

### Create a Virtual Environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install Base Dependencies

For the base-Python tokenization, embedding, and similarity experiments, no major third-party package is required.

For EmbeddingGemma and Hugging Face downloads:

```bash
pip install -U sentence-transformers transformers torch huggingface_hub python-dotenv
```

---

## Downloading Hugging Face Models with `downloader.py`

This project can use a helper script named `downloader.py` to download Hugging Face models into a local `models/` directory.

This is useful when you want to run models locally instead of downloading them every time through `sentence-transformers` or `transformers`.

---

### Install Downloader Dependencies

From the project root:

```bash
pip install -U python-dotenv huggingface_hub sentence-transformers transformers torch
```

---

### Create a `.env` File

Create a file named `.env` in the project root:

```text
as01_v02/.env
```

Add your Hugging Face token:

```bash
HF_TOKEN=hf_your_token_here
```

Do **not** commit this file to GitHub or share it publicly.

Recommended `.gitignore` entry:

```text
.env
models/
```

---

### Example `downloader.py` for EmbeddingGemma

Save the following as:

```text
as01_v02/downloader.py
```

```python
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login

# Load environment variables from .env file
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError(
        "HF_TOKEN not found in .env file. "
        "Add HF_TOKEN=hf_xxxxxxxx to your .env"
    )

# Authenticate with Hugging Face
login(token=hf_token)

# Download EmbeddingGemma model to local device
snapshot_download(
    repo_id="google/embeddinggemma-300m",
    local_dir="./models/embedders/google/embeddinggemma-300m",
    token=hf_token,
)

print("Downloaded google/embeddinggemma-300m to:")
print("./models/embedders/google/embeddinggemma-300m")
```

Run it with:

```bash
python3 downloader.py
```

After it finishes, your local model should be stored at:

```text
models/embedders/google/embeddinggemma-300m
```

You can then run:

```bash
python3 src/embeddinggemma.py encode \
  --model models/embedders/google/embeddinggemma-300m \
  --input-file data/embedding_inputs.json \
  --output-file data/embeddings_output.json
```

---

### Downloading Llama Instead

The code below downloads Meta Llama 3.1 8B Instruct instead of EmbeddingGemma.

> Note: Some Hugging Face models require accepting license terms on the model page before downloading. You may also need approval from Hugging Face or the model publisher.

```python
import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import huggingface_hub as hf
from huggingface_hub import (
    snapshot_download,
    HfApi,
    login,
)

# Load environment variables from .env file
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError(
        "HF_TOKEN not found in .env file. "
        "Add HF_TOKEN=hf_xxxxxxxx to your .env"
    )

# Authenticate with Hugging Face
login(token=hf_token)

# Download AI Model to local device
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="./models/llms/meta/Llama-3.1-8B-Instruct",
    token=hf_token,
)
```

Run:

```bash
python3 downloader.py
```

Expected local model folder:

```text
models/llms/meta/Llama-3.1-8B-Instruct
```

---

### Recommended Flexible `downloader.py`

Instead of editing the script each time, you can use this argument-based version:

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download


DEFAULT_REPO_ID = "google/embeddinggemma-300m"
DEFAULT_LOCAL_DIR = "./models/embedders/google/embeddinggemma-300m"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model to a local directory."
    )

    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repo ID. Default: {DEFAULT_REPO_ID}",
    )

    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"Local download directory. Default: {DEFAULT_LOCAL_DIR}",
    )

    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file containing HF_TOKEN. Default: .env",
    )

    args = parser.parse_args()

    load_dotenv(args.env_file)

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            f"HF_TOKEN not found in {args.env_file}. "
            "Add HF_TOKEN=hf_xxxxxxxx to your .env file."
        )

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print("Logging in to Hugging Face...")
    login(token=hf_token)

    print(f"Downloading repo: {args.repo_id}")
    print(f"Local directory: {local_dir}")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        token=hf_token,
    )

    print()
    print("Download complete.")
    print(f"Repo: {args.repo_id}")
    print(f"Path: {local_dir.resolve()}")


if __name__ == "__main__":
    main()
```

Download EmbeddingGemma:

```bash
python3 downloader.py
```

Or explicitly:

```bash
python3 downloader.py \
  --repo-id google/embeddinggemma-300m \
  --local-dir models/embedders/google/embeddinggemma-300m
```

Download Llama 3.1 8B Instruct:

```bash
python3 downloader.py \
  --repo-id meta-llama/Llama-3.1-8B-Instruct \
  --local-dir models/llms/meta/Llama-3.1-8B-Instruct
```

---

### Verify the Downloaded Model Path

After downloading EmbeddingGemma, check:

```bash
ls models/embedders/google/embeddinggemma-300m
```

You should see model files such as configuration files, tokenizer files, and model weight files.

Then test:

```bash
python3 src/embeddinggemma.py encode \
  --model models/embedders/google/embeddinggemma-300m \
  --input-file data/embedding_inputs.json \
  --output-file data/embeddings_output.json
```

---

### Course Policy Reminder for Model Downloads

Before downloading or using external Hugging Face models or datasets, make sure your use follows:

- The course technology FAQ
- CMU computing and data policies
- Hugging Face model license terms
- Any model-specific access restrictions

If AI tools or models were used to generate code, download models, or support the project, disclose that use according to the assignment policy.

---

## Create Test Data

Generate the small test dataset:

```bash
python3 make_test_data.py
```

This creates files under `data/`, including:

```text
data/embedding_inputs.json
data/embedding_lines.txt
data/openai_embeddings_request.json
data/embed_request.json
data/retrieval_eval.json
```

The test dataset includes short example documents and queries about topics such as:

- Python
- Large language models
- Semantic search
- Food
- Space
- Music

This dataset is only for testing the code pipeline. For the assignment report, use the dataset selected for your organizational scenario.

---

## Run the Main Test Script

Run:

```bash
python3 main_test.py
```

This script tests:

- Importing all source files
- Tokenization
- Fixed-size chunking
- Bag-of-Words embeddings
- TF-IDF embeddings
- Sparse cosine similarity
- Retrieval-style matching between queries and documents
- EmbeddingGemma helper functions

The test report is written to:

```text
data/main_test_report.json
```

---

### Optional Real EmbeddingGemma Test

To test the actual Google EmbeddingGemma model:

```bash
python3 main_test.py --run-gemma --gemma-model google/embeddinggemma-300m
```

Or use a local model directory:

```bash
python3 main_test.py \
  --run-gemma \
  --gemma-model models/embedders/google/embeddinggemma-300m
```

This may download the model from Hugging Face if it is not already cached and if a Hugging Face model ID is used.

---

## Run EmbeddingGemma from the Command Line

The file `src/embeddinggemma.py` supports a CLI mode.

Example using Hugging Face model ID:

```bash
python3 src/embeddinggemma.py encode \
  --model google/embeddinggemma-300m \
  --input-file data/embedding_inputs.json \
  --output-file data/embeddings_output.json
```

This reads text from:

```text
data/embedding_inputs.json
```

and writes embeddings to:

```text
data/embeddings_output.json
```

---

### Use a Local Model Folder

If the model has already been downloaded locally, use:

```bash
python3 src/embeddinggemma.py encode \
  --model models/embedders/google/embeddinggemma-300m \
  --input-file data/embedding_inputs.json \
  --output-file data/embeddings_output.json
```

If downloading manually with Hugging Face CLI:

```bash
huggingface-cli download google/embeddinggemma-300m \
  --local-dir models/embedders/google/embeddinggemma-300m
```

---

## Run EmbeddingGemma as a Local HTTP Server

`src/embeddinggemma.py` can also run a small local HTTP server using Python’s standard library.

It does **not** require FastAPI or Uvicorn.

Start the server:

```bash
python3 src/embeddinggemma.py serve \
  --model google/embeddinggemma-300m \
  --host 127.0.0.1 \
  --port 8000
```

Or with a local model:

```bash
python3 src/embeddinggemma.py serve \
  --model models/embedders/google/embeddinggemma-300m \
  --host 127.0.0.1 \
  --port 8000
```

---

### Test the OpenAI-Compatible Endpoint

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d @data/openai_embeddings_request.json
```

---

### Test the Custom Endpoint

```bash
curl http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d @data/embed_request.json
```

---

### Optional API Key

Set an environment variable:

```bash
export EMBEDDINGS_API_KEY="my-secret-key"
```

Then start the server:

```bash
python3 src/embeddinggemma.py serve \
  --model google/embeddinggemma-300m \
  --host 127.0.0.1 \
  --port 8000
```

Call with:

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d @data/openai_embeddings_request.json
```

---

## Assignment Part A: Dataset Datasheet

For Part A, the assignment requires selecting a dataset related to the team’s organizational scenario and final project.

The written memo should include a dataset datasheet/codebook with the sections below.

---

### Organizational Scenario

Describe:

- The organization or setting
- The AI model development problem
- Who may use the model
- What decisions or workflows the model may influence
- Why this dataset is relevant to the scenario

---

### Dataset Selection

Document:

- Dataset name
- Dataset source
- Dataset access method
- Any restrictions or safety concerns
- Whether the dataset is from:
  - Class Google Drive
  - Hugging Face
  - UCI Machine Learning Repository
  - Synthetic generation
  - Another approved source

---

### Existing Feature Descriptions

Create a table with:

| Feature Name | Data Type | Description | Utility for AI Model Development |
|---|---|---|---|
| Example feature | String/text/numeric/etc. | Description in your own words | Why it matters for the project |

---

### Traffic Light Protocol Labels

Using the Traffic Light Protocol, create classification labels for the examples in the dataset.

Suggested table:

| TLP Label | Meaning | Example from Dataset |
|---|---|---|
| TLP:CLEAR | Public information | Example |
| TLP:GREEN | Limited sharing | Example |
| TLP:AMBER | Restricted sharing | Example |
| TLP:RED | Highly restricted | Example |

Reference:  
<https://www.cisa.gov/news-events/news/traffic-light-protocol-tlp-definitions-and-usage>

---

### Descriptive Statistics

At minimum, report:

- Number of observations
- Number of features
- Missing values per feature
- Data types of all features

For numeric features:

- Minimum
- Maximum
- Mean
- Median
- Standard deviation
- Quartiles
- Range

For categorical or text features:

- Frequency counts
- Unique values
- Class distributions

Also include appropriate visualizations, such as:

- Histograms
- Bar charts
- Boxplots
- Frequency plots
- Density plots

---

### Distribution Analysis

In your own words, discuss:

- Which features appear normally distributed
- Which features appear non-normally distributed
- What evidence supports this
- How these distributions could affect model training or fine-tuning

---

### Feature Engineering

Create at least two new features.

For each new feature, describe:

| New Feature | Source Feature(s) | How Created | Data Type | Why Useful |
|---|---|---|---|---|

Examples of possible engineered text features:

- Text length in characters
- Text length in words
- Sentence count
- Average words per sentence
- Label simplification
- Topic grouping
- Missing-value indicator
- Keyword indicator
- TLP label

Then compare the new features to the original features using descriptive statistics.

---

## Assignment Part B: Tokenization and Embedding Experiments

Part B requires experimenting with knowledge representation.

This repository supports the following experiment pipeline:

```text
Raw text
   ↓
Tokenization or chunking
   ↓
Embedding
   ↓
Similarity measurement
   ↓
Interpretation
```

---

### Tokenization Strategies

The file `src/tokenizers.py` includes simple tokenization options:

| Strategy | Description |
|---|---|
| `word` | Lowercase text, strip punctuation, split into words. |
| `sentence` | Split text into sentences, then tokenize each sentence. |
| `fixed_chunk` | Split text into fixed-size word chunks. |

Example concept:

```python
tokenizer.tokenize_words("Embedding models turn text into vectors.")
```

---

### Embedding Strategies

The file `src/embedders_no_ai.py` includes:

| Strategy | Description |
|---|---|
| `bow` | Bag-of-Words sparse term-frequency embedding. |
| `tfidf` | TF-IDF sparse embedding, weighting distinctive words more heavily. |

The file `src/embeddinggemma.py` supports:

| Strategy | Description |
|---|---|
| `google/embeddinggemma-300m` | Dense semantic embeddings from Google EmbeddingGemma. |

---

### Similarity Metrics

The file `src/similarity_metrics.py` includes:

| Metric | Description |
|---|---|
| Sparse cosine similarity | Used for Bag-of-Words and TF-IDF vectors. |
| Dense cosine similarity | Used for dense model embeddings. |

---

### Experiment Interpretation

In the memo, discuss how the results affect:

- Model performance
- Generalization
- Bias and representation of concepts
- Dataset quality
- Future model training or fine-tuning decisions

Use evidence from the experiment results, such as:

- Similarity scores
- Most similar document pairs
- Least similar document pairs
- Retrieval accuracy
- Differences between Bag-of-Words, TF-IDF, and dense embeddings

---

## Assignment Part C: Research Plan and Analysis Memo

The final memo should include one to three research questions related to the organizational scenario and future model development project.

Each research question should be answerable through:

- The selected dataset
- LLM training or fine-tuning work
- Evaluation experiments
- Error analysis
- Model behavior analysis

Example structure:

```text
Research Question 1:
How does [dataset property or representation choice] affect [model behavior or outcome] in [organizational scenario]?

Research Question 2:
Can [feature engineering/tokenization/embedding strategy] improve [model performance, fairness, reliability, or usefulness]?

Research Question 3:
What risks or limitations arise when using this dataset to train or fine-tune an LLM for [organizational use case]?
```

The report must be written in the student’s own words.

---

## Reproducibility Checklist

Before submitting, verify that the code can be rerun:

```bash
python3 make_test_data.py
python3 main_test.py
```

Optional EmbeddingGemma CLI test:

```bash
python3 src/embeddinggemma.py encode \
  --model google/embeddinggemma-300m \
  --input-file data/embedding_inputs.json \
  --output-file data/embeddings_output.json
```

Optional local model test:

```bash
python3 src/embeddinggemma.py encode \
  --model models/embedders/google/embeddinggemma-300m \
  --input-file data/embedding_inputs.json \
  --output-file data/embeddings_output.json
```

Check that the following files exist:

```text
data/embedding_inputs.json
data/retrieval_eval.json
data/main_test_report.json
```

If running EmbeddingGemma:

```text
data/embeddings_output.json
```

---

## Troubleshooting

### Problem: `Path /models/... not found`

If you see an error like:

```text
FileNotFoundError: Path /models/embedders/google/embeddinggemma300m not found
```

then the model path is incorrect.

Use the Hugging Face model ID directly:

```bash
python3 src/embeddinggemma.py encode \
  --model google/embeddinggemma-300m \
  --input-file data/embedding_inputs.json \
  --output-file data/embeddings_output.json
```

Or use a valid local path:

```bash
python3 src/embeddinggemma.py encode \
  --model models/embedders/google/embeddinggemma-300m \
  --input-file data/embedding_inputs.json \
  --output-file data/embeddings_output.json
```

---

### Problem: `sentence-transformers` Missing

Install dependencies:

```bash
pip install -U sentence-transformers transformers torch
```

---

### Problem: `dotenv` Missing

Install:

```bash
pip install -U python-dotenv
```

---

### Problem: Hugging Face Authentication

If the model requires authentication or terms acceptance:

```bash
huggingface-cli login
```

Or use the provided `.env` and `downloader.py` workflow:

```bash
HF_TOKEN=hf_your_token_here
python3 downloader.py
```

---

### Problem: Python Package Support

If using Python 3.14 causes package issues, create an environment with Python 3.10, 3.11, or 3.12.

---

### Problem: `embed_corpus() missing 1 required positional argument`

This indicates that some starter helper functions may be structured as functions with a `self` argument rather than proper class instance methods.

The provided `main_test.py` includes runtime patches for this issue so that testing can continue. If desired, later refactor the helper files so methods consistently use either:

```python
self.method_name(...)
```

inside classes, or normal module-level functions without `self`.

---

### Problem: JSON Serialization Error in `main_test.py`

If the test report fails with:

```text
TypeError: Object of type module is not JSON serializable
```

make sure `main_test.py` includes a JSON-safe conversion helper before writing the report.

The report should store summaries of modules and functions rather than raw Python module objects.

---

## Submission Reminder

The final Canvas submission should include a link or PDF version of the assignment memo.

The memo should include:

- Part A: Organizational scenario and dataset datasheet
- Part B: Tokenization and embedding experiment results
- Part C: Research plan and analysis memo
- Citations for external sources
- Disclosure of any AI assistance used for code or other allowed tasks

Do **not** use AI to write, format, or edit the assignment report if prohibited by the course instructions.

---

## References

Traffic Light Protocol definitions:  
<https://www.cisa.gov/news-events/news/traffic-light-protocol-tlp-definitions-and-usage>

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021).  
Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92.  
<https://doi.org/10.1145/3458723>

Google EmbeddingGemma model:  
<https://huggingface.co/google/embeddinggemma-300m>

SentenceTransformers documentation:  
<https://www.sbert.net/>

Hugging Face Hub documentation:  
<https://huggingface.co/docs/huggingface_hub/index>
