"""tests/test_stage_3_audit.py — Stage 3 audit pipeline tests.

Three test groups:
  1. Disagreement detection   — pivot_judgments / find_disagreements
  2. Stratified sample        — stratified_sample distribution and size
  3. Decision persistence     — save_decision / load_decisions / resume logic
"""

from __future__ import annotations

import pandas as pd

from bpclassifier.audit import (
    find_disagreements,
    load_decisions,
    pivot_judgments,
    save_decision,
    stratified_sample,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_long_df(rows: list[dict]) -> pd.DataFrame:
    """Build a long-form judge_outputs DataFrame from a compact spec."""
    records = []
    for r in rows:
        for judge in ("anthropic", "deepseek", "llama"):
            records.append(
                {
                    "sentence_id": r["sentence_id"],
                    "judge_name": judge,
                    "label": r[judge],
                    "reasoning": f"{judge} says {r[judge]}",
                    "latency_ms": 50,
                    "model_id": "test-model",
                    "prompt_version": 1,
                    "error": None,
                }
            )
    return pd.DataFrame(records)


# 6-row synthetic fixture:
# rows 0-1: unanimous boilerplate
# rows 2-3: unanimous substantive
# rows 4: anthropic disagrees (minority=anthropic)
# rows 5: deepseek disagrees (minority=deepseek)
_SPEC = [
    {
        "sentence_id": "s0001",
        "anthropic": "boilerplate",
        "deepseek": "boilerplate",
        "llama": "boilerplate",
    },
    {
        "sentence_id": "s0002",
        "anthropic": "boilerplate",
        "deepseek": "boilerplate",
        "llama": "boilerplate",
    },
    {
        "sentence_id": "s0003",
        "anthropic": "substantive",
        "deepseek": "substantive",
        "llama": "substantive",
    },
    {
        "sentence_id": "s0004",
        "anthropic": "substantive",
        "deepseek": "substantive",
        "llama": "substantive",
    },
    {
        "sentence_id": "s0005",
        "anthropic": "substantive",
        "deepseek": "boilerplate",
        "llama": "boilerplate",
    },
    {
        "sentence_id": "s0006",
        "anthropic": "boilerplate",
        "deepseek": "substantive",
        "llama": "boilerplate",
    },
]


# ── 1. Disagreement detection ─────────────────────────────────────────────────


class TestDisagreementDetection:
    def _wide(self) -> pd.DataFrame:
        return pivot_judgments(_make_long_df(_SPEC))

    def test_pivot_row_count(self):
        wide = self._wide()
        assert len(wide) == len(_SPEC)

    def test_pivot_columns_present(self):
        wide = self._wide()
        for col in (
            "sentence_id",
            "anthropic_label",
            "deepseek_label",
            "llama_label",
            "anthropic_reasoning",
            "deepseek_reasoning",
            "llama_reasoning",
        ):
            assert col in wide.columns, f"Missing column: {col}"

    def test_unanimous_rows_excluded(self):
        wide = self._wide()
        dis = find_disagreements(wide)
        # Only s0005 and s0006 disagree
        assert set(dis["sentence_id"]) == {"s0005", "s0006"}

    def test_disagreement_count(self):
        wide = self._wide()
        dis = find_disagreements(wide)
        assert len(dis) == 2

    def test_minority_judge_correct(self):
        wide = self._wide()
        dis = find_disagreements(wide).set_index("sentence_id")
        assert dis.loc["s0005", "minority_judge"] == "anthropic"
        assert dis.loc["s0006", "minority_judge"] == "deepseek"

    def test_no_disagreements_returns_empty(self):
        spec_unanimous = [
            {
                "sentence_id": f"u{i:04d}",
                "anthropic": "boilerplate",
                "deepseek": "boilerplate",
                "llama": "boilerplate",
            }
            for i in range(5)
        ]
        wide = pivot_judgments(_make_long_df(spec_unanimous))
        dis = find_disagreements(wide)
        assert len(dis) == 0


# ── 2. Stratified sample ──────────────────────────────────────────────────────


class TestStratifiedSample:
    def _make_disagreements(self, per_judge: int = 20) -> pd.DataFrame:
        """Build a synthetic disagreements DataFrame with equal groups."""
        records = []
        for judge in ("anthropic", "deepseek", "llama"):
            for i in range(per_judge):
                records.append(
                    {
                        "sentence_id": f"{judge}_{i:04d}",
                        "anthropic_label": "boilerplate",
                        "deepseek_label": "boilerplate",
                        "llama_label": "boilerplate",
                        "minority_judge": judge,
                    }
                )
        return pd.DataFrame(records)

    def test_size_at_most_n(self):
        dis = self._make_disagreements(per_judge=20)
        sample = stratified_sample(dis, n=15, seed=42)
        assert len(sample) <= 15

    def test_no_duplicates(self):
        dis = self._make_disagreements(per_judge=20)
        sample = stratified_sample(dis, n=30, seed=42)
        assert sample["sentence_id"].is_unique

    def test_all_judges_represented(self):
        dis = self._make_disagreements(per_judge=20)
        sample = stratified_sample(dis, n=30, seed=42)
        represented = set(sample["minority_judge"].unique())
        assert represented == {"anthropic", "deepseek", "llama"}

    def test_roughly_equal_groups(self):
        dis = self._make_disagreements(per_judge=20)
        sample = stratified_sample(dis, n=30, seed=42)
        counts = sample["minority_judge"].value_counts()
        # Each group should be within 1 of target (10)
        for judge in ("anthropic", "deepseek", "llama"):
            assert 9 <= counts.get(judge, 0) <= 11

    def test_reproducible(self):
        dis = self._make_disagreements(per_judge=20)
        s1 = stratified_sample(dis, n=30, seed=42)
        s2 = stratified_sample(dis, n=30, seed=42)
        assert list(s1["sentence_id"]) == list(s2["sentence_id"])

    def test_small_group_fallback(self):
        """When one group has fewer rows than target, still returns ≤ n rows."""
        records = []
        for i in range(3):
            records.append({"sentence_id": f"ant_{i}", "minority_judge": "anthropic"})
        for i in range(20):
            records.append({"sentence_id": f"ds_{i}", "minority_judge": "deepseek"})
        for i in range(20):
            records.append({"sentence_id": f"ll_{i}", "minority_judge": "llama"})
        dis = pd.DataFrame(records)
        sample = stratified_sample(dis, n=15, seed=42)
        assert len(sample) <= 15
        assert sample["sentence_id"].is_unique


# ── 3. Decision persistence and resume logic ──────────────────────────────────


class TestDecisionPersistence:
    def test_save_then_load(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        save_decision("s0001", "boilerplate", {"anthropic": "boilerplate"}, path)
        decisions = load_decisions(path)
        assert "s0001" in decisions
        assert decisions["s0001"]["david_label"] == "boilerplate"

    def test_load_missing_file_returns_empty(self, tmp_path):
        decisions = load_decisions(tmp_path / "nonexistent.jsonl")
        assert decisions == {}

    def test_multiple_saves_all_loaded(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        for i in range(10):
            save_decision(f"s{i:04d}", "boilerplate", {}, path)
        decisions = load_decisions(path)
        assert len(decisions) == 10

    def test_corrupt_line_skipped(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        path.write_text(
            '{"sentence_id": "good", "david_label": "boilerplate", '
            '"timestamp_iso": "2025-01-01T00:00:00+00:00", "original_judge_labels": {}}\n'
            "NOT_VALID_JSON\n",
            encoding="utf-8",
        )
        decisions = load_decisions(path)
        assert len(decisions) == 1
        assert "good" in decisions

    def test_resume_skips_already_decided(self, tmp_path):
        """
        Simulate 25 pre-existing decisions; verify that only the
        remaining 25 sentence_ids are not in the loaded decisions dict.
        """
        path = tmp_path / "decisions.jsonl"
        all_ids = [f"s{i:04d}" for i in range(50)]
        decided_ids = all_ids[:25]
        for sid in decided_ids:
            save_decision(sid, "boilerplate", {}, path)

        decisions = load_decisions(path)
        done = set(decisions.keys())
        remaining = [sid for sid in all_ids if sid not in done]

        assert len(done) == 25
        assert len(remaining) == 25
        assert set(decided_ids).isdisjoint(set(remaining))

    def test_decision_has_required_fields(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        save_decision("x001", "substantive", {"anthropic": "boilerplate"}, path)
        decisions = load_decisions(path)
        entry = decisions["x001"]
        for field in ("sentence_id", "david_label", "timestamp_iso", "original_judge_labels"):
            assert field in entry, f"Missing field: {field}"
        assert entry["david_label"] == "substantive"

    def test_parent_dir_created(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        path = deep / "decisions.jsonl"
        save_decision("y001", "boilerplate", {}, path)
        assert path.exists()
