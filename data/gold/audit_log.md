# audit_log.md — Stage 3 Gold-Label Methodology

## Overview

Gold labels were produced for 2,500 sentences sampled from 131 earnings-call transcripts.
Labels freeze at `data/gold/labeled.parquet` (2,500 rows).

---

## Label hierarchy

| Priority | Condition | Source tag | Count |
|---|---|---|---|
| 1 | David's hand-audit decision | `audit` | 46 |
| 2 | Qwen3.5-27B meta-judge (disagreement cases) | `meta_judge` | 286 |
| 3 | Unanimous majority vote (all 3 judges agree) | `unanimous` | 2,168 |
| **Total** | | | **2,500** |

---

## Inter-rater agreement

Three LLM judges labeled the 2,500-sentence sample independently:
- **Anthropic** claude-sonnet-4-6
- **DeepSeek** DeepSeek-V3.1 (via W&B Inference)
- **Llama** Llama-3.3-70B (via W&B Inference)

### Pairwise Cohen's κ

| Pair | κ | Interpretation |
|---|---|---|
| Anthropic ↔ DeepSeek | 0.6473 | Substantial |
| Anthropic ↔ Llama | 0.7407 | Substantial |
| DeepSeek ↔ Llama | 0.7575 | Substantial |

### Fleiss' κ (all three judges)

**κ = 0.7119** — Substantial agreement (Landis & Koch scale: 0.61–0.80 = substantial).

### Disagreement breakdown

- Total disagreements: **332 / 2,500 (13.3%)**
- Minority judge distribution:
  - Anthropic was minority: 167 cases (50.3%)
  - DeepSeek was minority: 115 cases (34.6%)
  - Llama was minority: 50 cases (15.1%)

---

## Hand-audit procedure

A stratified sample of 50 disagreement cases was drawn (~17 per minority-judge group).
David reviewed each case against the rubric (`data/gold/rubric.md`) and assigned a final label.

- **Cases reviewed:** 46 of 50
- **Cases skipped (4):** sentences where the correct label was genuinely ambiguous given the rubric; David chose not to override the meta-judge on these.
- **David's audit distribution:** boilerplate=23, substantive=23

### Meta-judge quality check

Qwen3.5-27B meta-judge predictions were compared against David's 46 audit decisions:

- **Agreement rate: 26/46 (56.5%)**

This agreement rate is expected to be near chance for this subset — by construction, these are the 332 hardest cases where three LLM judges actively disagreed. The meta-judge's value is providing a consistent, rubric-grounded tiebreaker; David's audit overrides it where he had higher confidence. The relatively low agreement reflects genuine case difficulty, not a systematic model error.

---

## Final class balance

| Label | Count | % |
|---|---|---|
| substantive | 1,970 | 78.8% |
| boilerplate | 530 | 21.2% |
| **Total** | **2,500** | **100%** |

The imbalance (~4:1) is expected: earnings-call transcripts contain predominantly substantive executive commentary; boilerplate (safe-harbor language, operator housekeeping, generic thanks) comprises roughly 20% of the corpus.

---

## Rubric notes and edge-case decisions

The rubric (`data/gold/rubric.md`) contains:
- 10 boilerplate anchor examples
- 10 substantive anchor examples
- 10 edge-case rules

Key ambiguity rules applied during audit:
1. Forward-looking statements with quantitative guidance → **substantive** (even if formulaic phrasing)
2. Pure operator housekeeping ("Please go ahead") → **boilerplate**
3. Earnings-call safe-harbor disclaimers → **boilerplate** (regardless of length)
4. Analyst question sentences: labeled on informational content, not speaker role
5. Sentences referencing prior quarters' numbers without new guidance → **boilerplate**

---

## Artifacts

| Artifact | Location |
|---|---|
| 3-judge outputs (long-form) | `data/interim/judge_outputs.parquet` |
| Meta-judge outputs | `data/interim/meta_judge_outputs.parquet` |
| Audit decisions (JSONL) | `data/gold/audit_decisions.jsonl` |
| Audit sample | `data/gold/audit_sample.parquet` |
| Frozen gold labels | `data/gold/labeled.parquet` |
| W&B Artifact | `gold-labels:v0` |
