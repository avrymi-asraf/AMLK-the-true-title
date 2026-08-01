"""Tests for data/download.py and data/download_raw.py normalization + E4 pool.

Behavioral checks: normalizers map rows to {text, summary, source, hesum_id} and
skip empties; the raw E4 pool excludes curated held-out ids, is text-deduplicated,
and is reproducible under its sample seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import datasets as hf_datasets

from data.download import normalize_curated_record
from data.download_raw import (
    TARGET_SAMPLE_SIZE,
    curated_held_out_ids,
    dedupe_by_text,
    exclude_held_out,
    normalize_raw_record,
    sample_records,
)
from data.preprocess import split_dataset


def test_normalize_curated_record_happy_path():
    record = {
        "hesum_id": "abc",
        "text": "תוכן המאמר המלא",
        "headline": "כותרת",
    }
    result = normalize_curated_record(record)
    assert result == {
        "text": "תוכן המאמר המלא",
        "summary": "כותרת",
        "source": "hesum-curated",
        "hesum_id": "abc",
    }


def test_normalize_curated_skips_empty_text():
    assert normalize_curated_record({"hesum_id": "1", "text": "", "headline": "כ"}) is None


def test_normalize_curated_skips_empty_headline():
    assert normalize_curated_record({"hesum_id": "1", "text": "ט", "headline": ""}) is None


def test_normalize_raw_record_happy_path():
    record = {"id": "r1", "text": "מאמר", "headline": "כותרת גולמית"}
    result = normalize_raw_record(record)
    assert result == {
        "text": "מאמר",
        "summary": "כותרת גולמית",
        "source": "hesum-raw",
        "hesum_id": "r1",
    }


def test_normalize_raw_skips_empty():
    assert normalize_raw_record({"id": "1", "text": "", "headline": "x"}) is None
    assert normalize_raw_record({"id": "1", "text": "x", "headline": ""}) is None


def test_exclude_held_out_drops_val_test_ids():
    records = [
        {"hesum_id": "a", "text": "t1", "summary": "s1", "source": "hesum-raw"},
        {"hesum_id": "b", "text": "t2", "summary": "s2", "source": "hesum-raw"},
        {"hesum_id": "c", "text": "t3", "summary": "s3", "source": "hesum-raw"},
    ]
    held = {"b", "c"}
    kept = exclude_held_out(records, held)
    assert [r["hesum_id"] for r in kept] == ["a"]


def test_dedupe_by_text_keeps_first():
    records = [
        {"hesum_id": "1", "text": "same", "summary": "s1", "source": "hesum-raw"},
        {"hesum_id": "2", "text": "same", "summary": "s2", "source": "hesum-raw"},
        {"hesum_id": "3", "text": "other", "summary": "s3", "source": "hesum-raw"},
    ]
    out = dedupe_by_text(records)
    assert len(out) == 2
    assert out[0]["hesum_id"] == "1"
    assert out[1]["hesum_id"] == "3"


def test_sample_records_reproducible_under_seed():
    records = [
        {"hesum_id": str(i), "text": f"t{i}", "summary": f"s{i}", "source": "hesum-raw"}
        for i in range(100)
    ]
    a = sample_records(records, n=20, seed=7)
    b = sample_records(records, n=20, seed=7)
    c = sample_records(records, n=20, seed=8)
    assert [r["hesum_id"] for r in a] == [r["hesum_id"] for r in b]
    assert [r["hesum_id"] for r in a] != [r["hesum_id"] for r in c]


def test_sample_disjoint_from_held_out():
    """Id exclusion is applied before sample; sample must not reintroduce held-out ids."""
    held = {str(i) for i in range(10, 20)}
    pool = [
        {"hesum_id": str(i), "text": f"t{i}", "summary": f"s{i}", "source": "hesum-raw"}
        for i in range(50)
        if str(i) not in held
    ]
    sample = sample_records(pool, n=15, seed=42)
    assert not {r["hesum_id"] for r in sample} & held


def test_curated_held_out_ids_matches_split_sizes(tmp_path: Path):
    """Reproduce curated split on synthetic ids: held-out = val ∪ test."""
    n = 100
    rows = [
        {"hesum_id": f"id{i}", "text": f"article {i} body", "headline": f"head {i}"}
        for i in range(n)
    ]
    curated_path = tmp_path / "final_clean_hesum.json"
    curated_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    held = curated_held_out_ids(curated_path, seed=42)
    ds = hf_datasets.Dataset.from_dict({"hesum_id": [r["hesum_id"] for r in rows]})
    train, val, test = split_dataset(ds, seed=42)
    assert held == set(val["hesum_id"]) | set(test["hesum_id"])
    assert len(held) == len(val) + len(test)
    assert not held & set(train["hesum_id"])


def test_target_sample_size_constant():
    # Guard: 5854 → split_dataset yields 4683/585/586 (matched to curated).
    assert TARGET_SAMPLE_SIZE == 5854
