## Project Goal

* **Description:** AMLK is a Hebrew text summarization research project. The goal is to fine-tune the Qwen 3.5 2B language model on Hebrew summarization datasets, evaluate it with ROUGE, BERTScore, and LLM-based metrics, and produce a research paper and presentation. The project runs locally for development and on HuggingFace infrastructure for training jobs when local compute is insufficient; all scripts are executed as command-line Python scripts.

---

## Project Structure - remember to update it when you make changes

* **Architecture:** The project is divided into three sequential pipelines:
  1. **Training pipeline** — downloads Hebrew summarization datasets (IAHLT summarization_he, HeSum), loads the Qwen 3.5 2B base model, and fine-tunes it using the HuggingFace `transformers`/`trl` stack. If local GPU is insufficient, the job is submitted to HuggingFace as a remote training job.
  2. **Evaluation pipeline** — takes the fine-tuned checkpoint and runs it against a held-out test set, computing ROUGE scores, BERTScore, and an LLM-as-judge evaluation (via the Gemini API).
  3. **Results & reporting** — aggregated metrics feed into the final paper and presentation.

* **Code Flow:**
  1. Dataset download & preprocessing → tokenised dataset saved to disk
  2. Model fine-tuning → checkpoint saved to disk / HF Hub
  3. Inference on test set → predictions saved to disk
  4. Evaluation scripts consume predictions → produce metric reports
  5. Reports feed the paper / presentation

---

## File Structure - remember to update it with the latest project information

```
/AMLK
├── .claude/
│   └── skills/
│       └── coding-principles/SKILL.md   # Project-local coding standards
├── data/
│   ├── __init__.py
│   └── download.py                      # Pipeline step 1: downloads & normalizes IAHLT+HeSum datasets
├── tests/
│   ├── __init__.py
│   └── test_download.py                 # Unit tests for download.py normalization functions
├── training/                            # (planned) Fine-tuning scripts
├── evaluation/                          # (planned) ROUGE / BERTScore / LLM eval scripts
├── outputs/
│   └── data/
│       └── raw/
│           └── combined.jsonl           # Merged normalized dataset (gitignored)
├── .env                                 # HF_TOKEN, GEMINI_API_KEY — never commit
├── .gitignore
├── AGENTS.md
├── CLAUDE.md                            # Claude Code guidance
├── README.md
├── requirements.txt
└── TODO.md                              # Milestone tracker
```

* `.claude/skills/coding-principles/`: Defines project coding standards — all contributors (human and AI) must follow this skill.
* `data/download.py`: Downloads Hebrew summarization datasets (biunlp/HeSum, and IAHLT/summarization_he when accessible), normalises to `{text, summary, source}` schema, and writes to `outputs/data/raw/combined.jsonl`.
* `tests/test_download.py`: Unit tests for the normalization functions in `data/download.py`.
* `training/`: Fine-tuning entry points and configuration.
* `evaluation/`: Scripts for each evaluation method (ROUGE, BERTScore, LLM judge).
* `outputs/`: Checkpoints, generated summaries, metric JSON/CSV files. Gitignored once large artifacts appear.

---

## Building and Running

**Prerequisites:**
* Python 3.10+
* `pip install transformers trl datasets accelerate evaluate bert-score rouge-score google-generativeai`
* Copy `.env.example` → `.env` and fill in:
  * `HF_TOKEN` — HuggingFace access token (needed for model download and remote jobs)
  * `GEMINI_API_KEY` — Gemini API key (needed for LLM-based evaluation)
* Source the env before running any script: `source .env`

**Build Steps (if applicable):**
1. Download datasets: scripts in `data/` (planned)
2. Preprocess / tokenise: output written to `data/processed/`

**Running the Application:**
1. Fine-tune: `python training/train.py` (or submit as HF job — see training README when added)
2. Evaluate: `python evaluation/evaluate.py --checkpoint outputs/<run-name>`

---

## Status - remember to update it

Dataset download (Task 2) is complete as of 2026-05-26. `data/download.py` downloads and normalizes Hebrew summarization datasets (biunlp/HeSum successfully; IAHLT/summarization_he is inaccessible — requires gating approval or different credentials). The combined dataset has 10,000 records written to `outputs/data/raw/combined.jsonl`. Next step is preprocessing/tokenization (Task 3), followed by model fine-tuning (pipeline A). The presentation deadline is 2026-06-14 and final project submission is 2026-07-31.

---

## Code Writing Rules
Do not create new documentation files (unless explicitly requested). Only update documentation via the `README` if necessary.

### File Header (Mandatory)
In the header of every code file, you **must** describe how that file relates to the **overall project architecture** and **code flow**.

Each code file **must** include a short description (no more than 4–5 sentences) that explains the following:
- Its role in the **big picture** (as defined in the **Project Structure** section).
- Its connection to the main **code flow** of the project.
- The intended **execution environment** (where this code will run, as defined in the **Project Goal** section).
- The skills, memory, shared docs are very important to continue working on the project. You have all these as live files and currently updating them is very very important. Remember to do it!
