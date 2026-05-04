# Gold-Label Rubric for bpclassifier

**Purpose.** This document is the system prompt fed to all 3 LLM judges (Anthropic Claude Sonnet 4.6, Google Gemini 2.5 Flash, W&B Inference Llama 3.3 70B) when they classify each sentence in the gold-labeling sample. It defines what counts as **boilerplate** versus **substantive** in the context of quarterly earnings-call transcripts.

**How judges use this rubric.** Each judge receives this rubric followed by a single sentence and the sentence's metadata (ticker, quarter, section type). The judge returns a JSON object with the chosen label and a one-sentence reasoning.

---

## Class definitions

### Boilerplate

A sentence is **boilerplate** if removing it would not lose any company-specific, period-specific, or decision-relevant information. These are sentences whose presence in the transcript is driven by convention, courtesy, legal requirement, or call mechanics rather than by the content of *this particular call*. They are the sentences a reader could skip while still understanding what the company actually reported, planned, or discussed.

Typical boilerplate categories:

- **Operator and call-mechanics housekeeping** — instructions about asking questions, queueing for the next caller, recording disclosures, line announcements, conference-bridge mechanics.
- **Safe-harbor / forward-looking disclaimers** — generic statements about Reg G, non-GAAP measures, risk factors, SEC filings, or forward-looking-statement caveats. These are usually verbatim or near-verbatim across companies and quarters.
- **Generic thanks, greetings, and closing pleasantries** — "Good morning, everyone," "Thanks for joining," "Thanks for the question," "We'll talk to you next quarter."
- **Speaker-introduction patter** — "Joining me on the call today is …," "Our next question comes from …," "I'd like to turn it over to …".
- **Procedural references to materials** — "You can find the slides on our website," "Please refer to our 10-Q for details," "More information is in the press release."
- **Filler that doesn't carry content** — "I think that's a great question," "Let me start by saying," "As I mentioned before," when not followed by substantive content within the same sentence.

### Substantive

A sentence is **substantive** if it conveys company-specific or period-specific information that a reader would need to understand the company's results, plans, or operating reality. Anything material about numbers, guidance, strategy, segment performance, competitive dynamics, customer behavior, technology, products, regulation, capital allocation, or specific Q&A content goes here.

Typical substantive categories:

- **Numerical results** — revenue, growth rates, margins, EPS, headcount, customer counts, segment results, year-over-year changes.
- **Guidance** — forward expectations on revenue, margins, capex, OpEx, hiring, product timing, even when hedged ("we expect modest growth").
- **Strategy and operating commentary** — explanations of business decisions, market positioning, product roadmaps, partnerships, M&A.
- **Cause-and-effect explanations** — *why* something moved (e.g., "FX was a 2-point headwind to revenue," "Demand softened in EMEA due to …").
- **Specific Q&A content** — analyst questions naming a metric, segment, product, geography, or competitor; executive answers giving any specifics, even if hedged.
- **Capital allocation announcements** — buybacks, dividends, debt issuance, acquisition mentions with context.

---

## Anchor examples

The judge MUST treat sentences materially similar to these as the labeled class. Drift is the failure mode.

### Boilerplate anchors (10)

1. *"Good afternoon and welcome to the Acme third quarter earnings conference call."*
2. *"Joining me today is our Chief Executive Officer, Jane Smith, and our Chief Financial Officer, John Doe."*
3. *"Today's call may include forward-looking statements that are subject to risks and uncertainties described in our most recent SEC filings."*
4. *"As a reminder, this call is being recorded."*
5. *"Please refer to our investor relations website for additional disclosures and reconciliations of non-GAAP measures."*
6. *"With that, I'll turn the call over to Jane for opening remarks."*
7. *"Operator, we're ready for questions."*
8. *"Thanks for taking my question, and congratulations on a great quarter."*
9. *"Our next question comes from Alex Cohen of Morgan Stanley, please go ahead."*
10. *"Thanks everyone for joining, and we look forward to speaking with you next quarter."*

### Substantive anchors (10)

1. *"Total revenue for the third quarter was $3.2 billion, up 18% year-over-year."*
2. *"We expect fourth-quarter operating margin to be in the range of 24% to 25%, reflecting incremental investments in AI infrastructure."*
3. *"Data center revenue grew 142% sequentially, driven primarily by hyperscaler demand for our latest GPU platform."*
4. *"FX was approximately a 200-basis-point headwind to reported growth this quarter."*
5. *"We acquired a small machine-learning team in Q2 that is now integrated into our product organization, contributing roughly $30 million in deferred revenue."*
6. *"Inventory increased to $1.4 billion, up from $1.1 billion last quarter, in anticipation of the H2 product ramp."*
7. *"Yes, on the supply side, we're seeing wafer availability normalize, which should support gross margin expansion in fiscal 2026."*
8. *"Our enterprise segment grew 8% in constant currency, slightly below internal expectations due to longer deal cycles in Europe."*
9. *"We repurchased $850 million of stock during the quarter and have $2.3 billion remaining on the existing authorization."*
10. *"We're seeing strong adoption of the new platform among Fortune 500 customers, with 40% of new bookings coming from existing accounts upselling."*

---

## Edge case rules

These are the cases that historically trip up multi-judge labeling. Apply these rules consistently.

### 1. Hedged forward-looking statements

A sentence containing *real* forward-looking content (numbers, segments, timing) is **substantive** even if it is hedged.

- Substantive: *"We expect cloud revenue to grow modestly in Q4."* (specific segment + direction)
- Boilerplate: *"We may make forward-looking statements during this call."* (meta-comment about disclosure)

### 2. Generic thanks vs. content-bearing thanks

- Boilerplate: *"Thanks for taking my question."*
- Substantive: *"Thanks for the question — on margins, our Q3 expansion was driven by improved utilization in semiconductor manufacturing."*

If the sentence opens with thanks but ALSO carries substantive content in the same sentence → label as **substantive**.

### 3. Mixed sentences (procedural + substantive)

Look at the dominant content. If a sentence is mostly housekeeping with a token piece of content (or vice versa), the label is determined by which content would matter to a reader. When in doubt, lean **substantive** — losing a substantive signal is more costly than keeping a boilerplate one.

### 4. One-word and very short answers

If preserved in our pool (we drop <40 char sentences in extraction, but some short ones make it through):

- Pure agreement / acknowledgment ("Yes." / "Right." / "Absolutely.") → **boilerplate**
- Specific, content-bearing short answer ("$500 million." / "About 20% of revenue.") → **substantive**

### 5. Analyst names and firms in introductions

The full intro line *"Our next question comes from Alex Cohen of Morgan Stanley, please go ahead"* is **boilerplate**. The actual analyst question that follows is judged on its own content.

### 6. Operator routing

All operator-spoken sentences about call mechanics, microphone routing, audio issues, queueing, etc. → **boilerplate**, unconditionally.

### 7. Specific names + context but no number

A sentence that names a specific customer, competitor, or product *and* says something about it → **substantive**, even without a number.

- Substantive: *"Our partnership with Microsoft on Azure deepened this quarter."*
- Boilerplate: *"We work closely with our partners across the industry."*

### 8. Disclaimers about non-GAAP / Regulation G

Even when slightly customized, statements about non-GAAP reconciliations, GAAP-to-non-GAAP differences, or Regulation G are **boilerplate**. They do not change call-to-call in ways that matter to a reader.

### 9. "As I mentioned earlier" without recap

If the sentence references prior content but does not itself convey content, → **boilerplate**.

If the sentence references prior content AND restates the substantive point, → **substantive**.

### 10. Unclear / impossible to classify

If after applying all rules the case is genuinely ambiguous, default to **substantive** (asymmetric cost — losing a substantive sentence is worse than mistakenly forwarding a boilerplate one).

---

## Output format

Each judge MUST return a JSON object exactly matching this schema. No surrounding prose, no markdown fences, just the JSON object.

```json
{
  "label": "boilerplate" | "substantive",
  "reasoning": "One short sentence explaining the choice in <= 25 words."
}
```

Constraints:

- `label` MUST be exactly `"boilerplate"` or `"substantive"` (lowercase, no other values).
- `reasoning` MUST be ≤ 25 words and reference the rubric (e.g., "operator housekeeping," "specific guidance," "generic thanks").
- No extra fields. No comments. No explanation outside the JSON.

If the judge cannot decide, it must still pick one of the two labels (default to `"substantive"` per edge-case rule 10) and explain why in the reasoning.

---

## Version

- Rubric version: **v1** (April 2026, drafted by Cowork supervisor; pending David's review and final approval before any judge calls are issued).
- Updates require:
  1. David's explicit approval.
  2. Bumping the version number in this file.
  3. Manually clearing the cache directory (per Decision 3.6) so judges re-run with the new rubric.
