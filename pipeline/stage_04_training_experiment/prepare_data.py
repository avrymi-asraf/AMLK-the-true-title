"""Build the matched Hugging Face splits used by the final E4 experiment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import datasets as hf_datasets

from pipeline.common.paths import TRAINING_WORK_DIR
from pipeline.stage_04_training_experiment.config import ARTICLE_TOKEN_BUDGET, MODEL_ID
from pipeline.stage_04_training_experiment.prompts import build_prompt


OUTPUT_ROOT = TRAINING_WORK_DIR / "processed"
TRAIN_COLUMNS = ("text", "summary", "source", "prompt", "completion")


def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def split_dataset(
    dataset: hf_datasets.Dataset,
    seed: int = 42,
) -> tuple[hf_datasets.Dataset, hf_datasets.Dataset, hf_datasets.Dataset]:
    """Split 80/10/10 with the exact final E4 seed convention."""
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    validation_test = split["test"].train_test_split(test_size=0.5, seed=seed)
    return split["train"], validation_test["train"], validation_test["test"]


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(row)
        return rows
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return rows


def build_train_dataset(records: list[dict], tokenizer) -> hf_datasets.Dataset:
    if not records:
        raise ValueError("no E4 records were supplied")
    texts: list[str] = []
    summaries: list[str] = []
    sources: list[str] = []
    for index, record in enumerate(records):
        text = (record.get("text") or "").strip()
        summary = (record.get("summary") or record.get("headline") or "").strip()
        source = (record.get("source") or "").strip()
        if not text or not summary or not source:
            raise ValueError(f"record {index} has an empty text, summary, or source")
        texts.append(truncate_to_tokens(text, tokenizer, ARTICLE_TOKEN_BUDGET))
        summaries.append(summary)
        sources.append(source)
    return hf_datasets.Dataset.from_dict({
        "text": texts,
        "summary": summaries,
        "source": sources,
        "prompt": [build_prompt(text) for text in texts],
        "completion": list(summaries),
    })


def load_test_split(path: Path) -> hf_datasets.Dataset:
    test_path = path / "test" if (path / "test").exists() else path
    if not test_path.exists():
        raise FileNotFoundError(f"test split not found under {path}")
    return hf_datasets.load_from_disk(str(test_path))


def validate_train_dataset(
    train: hf_datasets.Dataset,
    validation: hf_datasets.Dataset,
    test: hf_datasets.Dataset,
) -> None:
    """Validate the final E4 schema, pairing, and leakage safeguards."""
    for split_name, dataset in (("train", train), ("validation", validation), ("test", test)):
        if len(dataset) == 0:
            raise ValueError(f"{split_name} split is empty")
        missing = [column for column in TRAIN_COLUMNS if column not in dataset.column_names]
        if missing:
            raise ValueError(f"{split_name} lacks required columns: {missing}")
        for column in TRAIN_COLUMNS:
            for index, value in enumerate(dataset[column]):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{split_name}.{column}[{index}] is empty or non-string")
        for index, (summary, completion) in enumerate(zip(dataset["summary"], dataset["completion"])):
            if summary != completion:
                raise ValueError(f"{split_name}[{index}] completion differs from summary")
        for index, (text, prompt) in enumerate(zip(dataset["text"], dataset["prompt"])):
            if text not in prompt:
                raise ValueError(f"{split_name}[{index}] prompt does not contain the article")
    train_texts, validation_texts, test_texts = set(train["text"]), set(validation["text"]), set(test["text"])
    if train_texts & validation_texts or train_texts & test_texts or validation_texts & test_texts:
        raise ValueError("train/validation/test article texts overlap")


def save_splits(
    train: hf_datasets.Dataset,
    validation: hf_datasets.Dataset,
    test: hf_datasets.Dataset,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(output_dir / "train"))
    validation.save_to_disk(str(output_dir / "val"))
    test.save_to_disk(str(output_dir / "test"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("uncleaned", "curated"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--test-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output or OUTPUT_ROOT / args.arm
    if output_dir.exists() and not args.force:
        print(f"Output already exists at {output_dir}. Pass --force to rebuild.")
        return 0

    from transformers import AutoTokenizer

    records = load_records(args.input)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=False)
    dataset = build_train_dataset(records, tokenizer)
    train, validation, test = split_dataset(dataset, seed=args.seed)
    validate_train_dataset(train, validation, test)
    if args.test_from is not None:
        test = load_test_split(args.test_from)
        validate_train_dataset(train, validation, test)
    save_splits(train, validation, test, output_dir)
    print(f"Saved E4 {args.arm} splits: train={len(train)}, val={len(validation)}, test={len(test)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
