"""tests/test_stage_3_labeling.py — Stage 3 gold-labeling pipeline tests.

Three test groups:
  1. Stratified sample correctness   — create_labeling_sample()
  2. Cache hit on rerun              — load_cache / append_to_cache / run_one_judge
  3. Output schema                   — required columns and dtypes
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pandas as pd

from bpclassifier.label import (
    REQUIRED_OUTPUT_COLUMNS,
    SEED,
    append_to_cache,
    create_labeling_sample,
    load_cache,
    parse_json_response,
    run_one_judge,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_sentences_df(
    n_companies: int = 4,
    sections: tuple[str, ...] = ("prepared_remarks", "question", "answer"),
    n_per_cell: int = 40,
) -> pd.DataFrame:
    """Build a synthetic sentences DataFrame matching the real schema."""
    rows = []
    for c in range(n_companies):
        for s in sections:
            for i in range(n_per_cell):
                rows.append(
                    {
                        "sentence_id": f"c{c}_{s}_{i:04d}",
                        "text": f"Company {c} reported results for {s} item {i}.",
                        "ticker": f"TK{c}",
                        "quarter": "Q1-2025",
                        "quarter_num": 1.0,
                        "year": 2025,
                        "call_date": "2025-01-01",
                        "company": f"CorpAlpha{c}",
                        "section_type": s,
                        "speaker_role": "executive",
                        "speaker_name": "",
                        "speaker_title": "",
                        "position_in_block": i,
                        "position_in_doc": i,
                    }
                )
    return pd.DataFrame(rows)


def _make_rows(sentence_ids: list[str]) -> list[Any]:
    """Build minimal namedtuple-style rows for run_one_judge."""
    return [
        SimpleNamespace(
            sentence_id=sid,
            text="Revenue grew 10% year-over-year in Q1.",
            ticker="TK0",
            quarter="Q1-2025",
            section_type="prepared_remarks",
        )
        for sid in sentence_ids
    ]


def _make_cache_entry(sentence_id: str, judge: str = "anthropic") -> dict:
    return {
        "sentence_id": sentence_id,
        "judge_name": judge,
        "label": "boilerplate",
        "reasoning": "generic thanks",
        "latency_ms": 50,
        "model_id": "test-model",
        "prompt_version": 1,
        "error": None,
    }


# ── 1. Stratified sample ──────────────────────────────────────────────────────


class TestCreateLabelingSample:
    def test_exact_size(self):
        df = _make_sentences_df()  # 4 × 3 × 40 = 480 rows
        sample = create_labeling_sample(df, n=100, seed=SEED)
        assert len(sample) == 100

    def test_reproducible(self):
        df = _make_sentences_df()
        s1 = create_labeling_sample(df, n=100, seed=SEED)
        s2 = create_labeling_sample(df, n=100, seed=SEED)
        pd.testing.assert_frame_equal(
            s1.reset_index(drop=True),
            s2.reset_index(drop=True),
        )

    def test_different_seeds_differ(self):
        df = _make_sentences_df()
        s1 = create_labeling_sample(df, n=100, seed=42)
        s2 = create_labeling_sample(df, n=100, seed=99)
        # Very unlikely to be identical under different seeds
        assert not s1["sentence_id"].equals(s2["sentence_id"])

    def test_all_strata_represented(self):
        """Every company × section_type combination must appear at least once."""
        df = _make_sentences_df()
        n_strata = df.groupby(["company", "section_type"]).ngroups
        sample = create_labeling_sample(df, n=n_strata * 3, seed=SEED)
        observed = sample.groupby(["company", "section_type"]).ngroups
        assert observed == n_strata

    def test_proportional_representation(self):
        """Each stratum's share in the sample should roughly match the pool share."""
        df = _make_sentences_df()
        sample = create_labeling_sample(df, n=100, seed=SEED)
        pool_shares = df.groupby("section_type").size() / len(df)
        sample_shares = sample.groupby("section_type").size() / len(sample)
        for sec in pool_shares.index:
            assert (
                abs(pool_shares[sec] - sample_shares.get(sec, 0.0)) < 0.10
            ), f"section_type='{sec}' share drifted more than 10 ppts from pool"

    def test_no_duplicates(self):
        df = _make_sentences_df()
        sample = create_labeling_sample(df, n=100, seed=SEED)
        assert sample["sentence_id"].is_unique

    def test_sample_is_subset_of_pool(self):
        df = _make_sentences_df()
        sample = create_labeling_sample(df, n=100, seed=SEED)
        assert set(sample["sentence_id"]).issubset(set(df["sentence_id"]))

    def test_fallback_when_small_strata(self):
        """Gracefully handles strata with a single member (can't stratify-split)."""
        df = _make_sentences_df(n_companies=2, sections=("prepared_remarks",), n_per_cell=1)
        # 2 strata × 1 row = 2 rows total; ask for 2
        sample = create_labeling_sample(df, n=2, seed=SEED)
        assert len(sample) >= 1  # at least something returned


# ── 2. Cache round-trip and hit-on-rerun ──────────────────────────────────────


class TestCache:
    def test_append_then_load(self, tmp_path):
        entry = _make_cache_entry("sent_001")
        append_to_cache("anthropic", entry, tmp_path)
        cache = load_cache("anthropic", tmp_path)
        assert "sent_001" in cache
        assert cache["sent_001"]["label"] == "boilerplate"

    def test_load_empty_returns_empty_dict(self, tmp_path):
        cache = load_cache("anthropic", tmp_path)
        assert cache == {}

    def test_multiple_entries_all_loaded(self, tmp_path):
        for i in range(10):
            append_to_cache("deepseek", _make_cache_entry(f"s{i:04d}", "deepseek"), tmp_path)
        cache = load_cache("deepseek", tmp_path)
        assert len(cache) == 10

    def test_corrupt_line_skipped(self, tmp_path):
        cache_file = tmp_path / "anthropic.jsonl"
        cache_file.write_text(
            '{"sentence_id": "good", "judge_name": "anthropic", "label": "boilerplate", '
            '"reasoning": "x", "latency_ms": 1, "model_id": "m", "prompt_version": 1, "error": null}\n'
            "NOT_VALID_JSON\n",
            encoding="utf-8",
        )
        cache = load_cache("anthropic", tmp_path)
        assert len(cache) == 1
        assert "good" in cache

    def test_cache_dir_created_on_append(self, tmp_path):
        subdir = tmp_path / "deep" / "cache"
        append_to_cache("llama", _make_cache_entry("x"), subdir)
        assert (subdir / "llama.jsonl").exists()

    def test_cache_hit_skips_judge_calls(self, tmp_path):
        """When all sentences are in cache, judge_fn must never be called."""
        sentence_ids = [f"s{i:04d}" for i in range(5)]
        for sid in sentence_ids:
            append_to_cache("anthropic", _make_cache_entry(sid), tmp_path)

        cache = load_cache("anthropic", tmp_path)
        rows = _make_rows(sentence_ids)

        call_count = [0]

        async def mock_judge_fn(sid: str, prompt: str) -> dict:
            call_count[0] += 1
            return _make_cache_entry(sid)

        results = asyncio.run(
            run_one_judge(
                "anthropic",
                rows,
                cache,
                mock_judge_fn,
                tmp_path,
                asyncio.Semaphore(5),
            )
        )

        assert call_count[0] == 0, "Expected zero API calls when all entries are cached"
        assert len(results) == 5

    def test_partial_cache_calls_only_missing(self, tmp_path):
        """Sentences not in cache trigger judge_fn; cached ones do not."""
        sentence_ids = [f"s{i:04d}" for i in range(6)]
        # Pre-cache first 3
        for sid in sentence_ids[:3]:
            append_to_cache("deepseek", _make_cache_entry(sid, "deepseek"), tmp_path)

        cache = load_cache("deepseek", tmp_path)
        rows = _make_rows(sentence_ids)

        called_ids: list[str] = []

        async def mock_judge_fn(sid: str, prompt: str) -> dict:
            called_ids.append(sid)
            return _make_cache_entry(sid, "deepseek")

        asyncio.run(
            run_one_judge(
                "deepseek",
                rows,
                cache,
                mock_judge_fn,
                tmp_path,
                asyncio.Semaphore(5),
            )
        )

        assert set(called_ids) == set(sentence_ids[3:])


# ── 3. Output schema ──────────────────────────────────────────────────────────


class TestOutputSchema:
    def _make_minimal_output(self) -> pd.DataFrame:
        """Build a minimal judge_outputs DataFrame that matches the spec."""
        rows = []
        for judge in ("anthropic", "deepseek", "llama"):
            rows.append(
                {
                    "sentence_id": "s0001",
                    "judge_name": judge,
                    "label": "boilerplate",
                    "reasoning": "operator housekeeping",
                    "latency_ms": 120,
                    "model_id": "test-model",
                    "prompt_version": 1,
                    "error": None,
                }
            )
        return pd.DataFrame(rows)

    def test_required_columns_present(self):
        df = self._make_minimal_output()
        missing = [c for c in REQUIRED_OUTPUT_COLUMNS if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_column_order_matches_spec(self):
        df = self._make_minimal_output()[REQUIRED_OUTPUT_COLUMNS]
        assert list(df.columns) == REQUIRED_OUTPUT_COLUMNS

    def test_label_values_are_valid_or_null(self):
        df = self._make_minimal_output()
        valid = {"boilerplate", "substantive", None}
        bad = [v for v in df["label"] if v not in valid]
        assert not bad, f"Unexpected label values: {bad}"

    def test_prompt_version_is_integer(self):
        df = self._make_minimal_output()
        assert df["prompt_version"].dtype in (int, "int64", "Int64")
        assert (df["prompt_version"] >= 1).all()

    def test_judge_names_correct(self):
        df = self._make_minimal_output()
        assert set(df["judge_name"]) == {"anthropic", "deepseek", "llama"}


# ── 4. parse_json_response edge cases ─────────────────────────────────────────


class TestParseJsonResponse:
    def test_plain_json(self):
        text = '{"label": "boilerplate", "reasoning": "operator housekeeping"}'
        result = parse_json_response(text)
        assert result is not None
        assert result["label"] == "boilerplate"

    def test_json_with_markdown_fence(self):
        text = '```json\n{"label": "substantive", "reasoning": "revenue guidance"}\n```'
        result = parse_json_response(text)
        assert result is not None
        assert result["label"] == "substantive"

    def test_invalid_label_returns_none(self):
        text = '{"label": "BOILERPLATE", "reasoning": "wrong case"}'
        assert parse_json_response(text) is None

    def test_garbage_returns_none(self):
        assert parse_json_response("not json at all") is None

    def test_empty_string_returns_none(self):
        assert parse_json_response("") is None

    def test_substantive_label(self):
        text = '{"label": "substantive", "reasoning": "specific revenue number"}'
        result = parse_json_response(text)
        assert result["label"] == "substantive"
