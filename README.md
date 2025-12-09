# 6600 Final Project — CEFR-Aware Document Simplification
Lin Ai · Zizheng Wang · Wenhao Zhou

## Overview
This repository contains our end-to-end pipeline for simplifying English news articles and educational passages to specific CEFR levels (A2/B1/B2). The system covers:
- data collection from open corpora plus fresh Guardian articles
- sentence/document cleaning and CEFR labeling
- pseudo pair generation by enforcing level-aware rewriting with DeepSeek
- LoRA fine-tuning of Flan-T5 for controllable simplification
- a LangGraph-based inference agent that plans, rewrites, and repairs drafts with automatic CEFR and salience checks
- automatic evaluation that mixes readability scores, semantic similarity, and LLM-as-a-judge scoring

Use this README as a reproducibility guide for the DSAN 6600 final project.

## Repository Layout
- `scripts/` – entry points for downloading corpora, training seq2seq models, and running the LangGraph pipeline (document-level utilities live in `scripts/document-level/`).
- `src/` – core source code
  - `build_finetuning_signals*.py`: dataset construction utilities
  - `document_processing/`: Guardian scraping + raw-to-clean conversions
  - `langgraph_pipeline*.py`: multi-stage simplification graph and helpers
  - `evaluation/`: cosine similarity, readability, and LLM judging scripts
- `configs/` – curated JSONL datasets we ship with the repo
- `data/raw` and `data/processed` – placeholder directories for downloaded corpora
- `outputs/` – drop checkpoints, generations, and evaluation CSVs here


## Environment Setup
1. **Python** – tested with Python 3.10/3.11 on macOS (MPS) and Linux (CUDA/T4).  
2. **Virtual environment**
   ```bash
   cd 6600-Final-Project
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   ```
3. **Dependencies** – install the libraries used across the repo:
   ```bash
   pip install -U torch transformers datasets sentence-transformers \
       langgraph langchain-openai openai peft accelerate textstat pandas numpy \
       scikit-learn tqdm python-dotenv nltk matplotlib
   python -m nltk.downloader punkt
   ```
   Adjust the `torch` install command if you need a CUDA-specific wheel.

## Required Environment Variables
Set the following before running any scripts (e.g., in `.env` or your shell profile):

| Variable | Purpose |
| --- | --- |
| `DEEPSEEK_API_KEY` (required) | Used by `build_finetuning_signals.py` and LangGraph for planning, rewriting, and salience checks via DeepSeek's OpenAI-compatible API. |
| `DEEPSEEK_MODEL` *(optional, default `deepseek-chat`)* | Swap to `deepseek-reasoner` or any compatible checkpoint. |
| `DEEPSEEK_BASE_URL` *(optional, default `https://api.deepseek.com/v1`)* | Override when proxying the API. |
| `FLAN_LOCAL_PATH` | Path to either a full Flan-T5 checkpoint or a merged adapter directory. Set this before invoking the LangGraph pipeline so it can load your fine-tuned weights locally (no HF download at inference time). |
| `OPENAI_API_KEY` | Required by `src/evaluation/llm_evaluator.py` when running GPT-4/4o audits. |
| `GUARDIAN_API_KEY` | Needed if you re-run `src/document_processing/get_news.py` to scrape fresh Guardian articles. Put the key into that script or export it as an env var. |

## Data Acquisition & Cleaning
1. **Download Universal CEFR corpora**  
   ```bash
   cd scripts
   python huggingface_download.py
   ```
   This writes `readme_train.jsonl`, `cambridge_train.jsonl`, and `elg_train.jsonl` to the repo root. Move them under `data/raw/` to keep things tidy.

2. **Guardian articles**  
   - Customize `src/document_processing/get_news.py` with your Guardian API key (or export it).
   - Run the script to create `guardian_articles.json`, then convert to CSV with `data_processing.py`.

3. **Text cleaning**  
   Use `src/langgraph_df_cleaning.py` (exposed via `clean_text`) to normalize punctuation, drop titles, and merge hard-wrapped lines. All data loaders rely on this helper, so keep preprocessing consistent.

## Building Fine-Tuning Signals
We train on a mix of human gold (TSAR) and pseudo pairs built from Universal CEFR sentences.

1. **Format TSAR 2025**  
   ```bash
   cd src/document_processing
   python tsar_signals_cleaning.py  # writes configs/tsar2025_formatted.jsonl
   ```
   This uses our CEFR classifier (`ModernBERT-base-doc_sent_en-Cefr`) to tag the originals and builds prompt-formatted entries.

2. **Generate pseudo pairs**  
   ```bash
   cd src
   python build_finetuning_signals.py
   ```
   Key steps performed automatically:
   - sentence segmentation + CEFR labeling of Universal CEFR datasets
   - DeepSeek batching (`LLM_MAX_WORKERS`/`LLM_GROUP_SIZE`) to complexify sentences one level above the ground truth
   - CEFR gating to discard rewrites that do not satisfy the target level
   - JSONL output at `universalcefr_pseudopairs_8.jsonl`

3. **Assemble the final training file**  
   Concatenate the pseudo pairs with the TSAR gold data into `finetuning_signals.jsonl` (see `configs/finetuning_signals.jsonl` for our reference mix). Every example must expose `input_text`, `target_text`, and optionally `weight`.

## Fine-Tuning Flan-T5 with LoRA
The repo includes two training scripts:
- `scripts/train_rewriting_model.py` – tuned for sentence/paragraph length (256 tokens).
- `scripts/document-level/train.py` – doubled context windows for long Guardian articles.

Both scripts:
1. Load `finetuning_signals.jsonl`.
2. Infer per-example weights (`GOLD_WEIGHT` vs `PSEUDO_WEIGHT`).
3. Tokenize with `AutoTokenizer.from_pretrained(MODEL_NAME)`.
4. Attach a LoRA adapter (defaults to rank 16, alpha 32 in `get_peft_model`).
5. Train with Hugging Face `Seq2SeqTrainer` + `EarlyStoppingCallback`.

Example command (sentence-level):
```bash
cd scripts
python train_rewriting_model.py \
  --output_dir ../outputs/flan_t5_base_lora \
  --logging_steps 25
```
Trainer arguments are defined inside the script; edit `train_cfg` or pass overrides via HF environment variables. After training, copy the adapter folder path into `FLAN_LOCAL_PATH` so the LangGraph runtime picks it up immediately.

## LangGraph Simplification Pipeline
`src/langgraph_pipeline.py` composes the full inference agent:
1. **Planning node** – DeepSeek creates a high-level rewrite plan tailored to the requested CEFR level.
2. **Rewrite node** – runs your fine-tuned Flan-T5 (loaded locally via `FLAN_LOCAL_PATH`) on each sentence.
3. **Build + salience check** – stitches sentences back into a draft and uses a question-answering loop to ensure key facts are preserved.
4. **CEFR check** – ModernBERT classifier validates every sentence and the whole passage.
5. **Repair loops** – LangGraph routes to either salience repair (LLM) or CEFR repair (force simpler rephrasing) until both constraints pass or loop budgets are exhausted.
6. **Stitch node** – returns the final simplified document.

Run a quick test:
```bash
cd scripts
python langgraph_pipeline_running.py
```
Edit the script to point at your CSV (`CSV_PATH`), target column, slice, and CEFR level. The example processes Guardian chunks 100–250 into `my_output_simplified(A2_100_250).csv`.

To simplify a single chunk from the CLI:
```bash
CEFR_LEVEL=A2 \
CHUNK_TEXT="$(cat sample.txt)" \
python src/langgraph_pipeline.py
```

## Evaluation
`src/evaluation/eval.py` orchestrates three signals:
1. **Cosine similarity (`cosine_sim.py`)** – SentenceTransformer embeddings (`paraphrase-multilingual-MiniLM-L12-v2` by default).
2. **Level appropriateness (`combine.py`)** – Flesch Reading Ease against CEFR-specific targets plus a weighted readability/level match score.
3. **LLM adjudication (`llm_evaluator.py`)** – GPT-4 (or configured model) scores fluency, faithfulness, and level match on a 0–5 scale; results are normalized to 0–1.

Usage:
```bash
cd src/evaluation
python eval.py \
  --input_path ../../outputs/my_output_simplified(A2_100_250).csv \
  --output_path ../../outputs/my_output_simplified(A2_100_250)_evaluated.csv
```
The script auto-detects column names (ID/original/simplified) and appends `_cosine_score`, `_llm_score`, `_flesch_reading_ease`, `_level_match_score`, `_appropriateness_score`, and `_total_avg`.

## Reproducing Our Results
1. Install dependencies and set env vars (`DEEPSEEK_*`, `OPENAI_API_KEY`, `FLAN_LOCAL_PATH`).
2. Download Universal CEFR datasets and collect Guardian data (optional but recommended for domain adaptation).
3. Run `tsar_signals_cleaning.py` and `build_finetuning_signals.py`, then merge JSONL files into `finetuning_signals.jsonl`.
4. Fine-tune Flan-T5 + LoRA using `scripts/train_rewriting_model.py` (or the document-level variant).
5. Update `FLAN_LOCAL_PATH` to your new adapter, then run `scripts/langgraph_pipeline_running.py` to simplify held-out Guardian articles.
6. Evaluate the generated CSV with `src/evaluation/eval.py` and inspect the `_total_avg` scores across CEFR levels.

## Troubleshooting
- **DeepSeek / OpenAI errors** – double-check API keys and rate limits. Both the dataset builder and LangGraph pipeline retry automatically, but persistent failures usually mean keys are missing.
- **Tokenizer or CEFR model OOM** – reduce `LLM_MAX_WORKERS`, `LLM_GROUP_SIZE`, or run on CPU (set `CUDA_VISIBLE_DEVICES=""`). `build_finetuning_signals.py` also supports truncation via `DEBUG_N`.
- **LangGraph cannot find weights** – ensure `FLAN_LOCAL_PATH` points to either a PEFT adapter directory (contains `adapter_config.json`) or a merged checkpoint folder with `config.json`. The loader prints helpful diagnostics.
- **Evaluation stalls** – GPT-based scoring is throttled to avoid rate limits; tweak `delay` inside `LLMEvaluator.evaluate_batch` if you have higher quotas.

Feel free to file issues or reach out if you extend the pipeline to additional CEFR levels or languages. Happy simplifying!
