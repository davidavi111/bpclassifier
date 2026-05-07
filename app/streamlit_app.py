"""app/streamlit_app.py — Boilerplate Classifier GUI (Stage 7).

Inputs:  ECT/ dropdown, file upload (.txt), or paste textarea
Pipeline: parse → sentence segmentation → FinBERT inference → threshold
Output:  inline transcript (boilerplate = red background) + stats panel
"""

from __future__ import annotations

import html
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ─── Path setup ───────────────────────────────────────────────────────────────
_APP = Path(__file__).parent
_ROOT = _APP.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

from bpclassifier.extract import (  # noqa: E402
    TranscriptParseError,
    _tokenize_sentences,
    parse_transcript,
    transcript_to_sentences,
)

_ECT_DIR = _ROOT / "ECT"
_EVAL_DIR = _ROOT / "data" / "eval"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Boilerplate Classifier",
    page_icon="📊",
    layout="wide",
)


# ─── Model loading (cached across reruns) ─────────────────────────────────────


@st.cache_resource(show_spinner="Loading winner model from W&B (first run only)…")
def load_winner():
    """Download winner FinBERT from W&B, load into memory. Cached."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sys.path.insert(0, str(_ROOT / "scripts"))
    from gpu_runner import get_wandb_key  # noqa: PLC0415

    get_wandb_key()
    import wandb  # noqa: PLC0415

    with open(_EVAL_DIR / "confusion_matrix_test.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    winner_name = cfg["winner"]
    threshold = float(cfg["threshold"])

    api = wandb.Api()
    art = api.artifact(
        f"david-avichzer-hebrew-university-of-jerusalem/"
        f"Boilerplate_Classifier/model-{winner_name}:v0"
    )
    model_dir = Path(art.download())

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    return model, tokenizer, threshold, winner_name


# ─── Inference ────────────────────────────────────────────────────────────────


def predict_proba(model, tokenizer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    import torch
    import torch.nn.functional as F  # noqa: N812

    all_probs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, truncation=True, padding=True, max_length=128, return_tensors="pt"
            )
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
    return np.clip(np.array(all_probs, dtype=float), 0.0, 1.0)


def run_pipeline(txt_path: Path, model, tokenizer, threshold: float) -> list[dict]:
    """Parse, segment, and label one transcript file. Returns list of sentence dicts."""
    transcript = parse_transcript(txt_path)
    sentences = transcript_to_sentences(transcript)
    if not sentences:
        return []

    texts = [s.text for s in sentences]
    probs = predict_proba(model, tokenizer, texts)

    return [
        {
            "text": s.text,
            "section_type": s.section_type,
            "prob": float(p),
            "label": "substantive" if p >= threshold else "boilerplate",
        }
        for s, p in zip(sentences, probs, strict=False)
    ]


def run_pipeline_raw(raw_text: str, model, tokenizer, threshold: float) -> list[dict]:
    """Tokenize unstructured pasted text and label it. Falls back when ECT parse fails."""
    sentences = [s for s in _tokenize_sentences(raw_text) if len(s) >= 40]
    if not sentences:
        return []
    probs = predict_proba(model, tokenizer, sentences)
    return [
        {
            "text": s,
            "section_type": "prepared_remarks",
            "prob": float(p),
            "label": "substantive" if p >= threshold else "boilerplate",
        }
        for s, p in zip(sentences, probs, strict=False)
    ]


# ─── Rendering ────────────────────────────────────────────────────────────────

_BP_STYLE = (
    "background-color:#ffcccc; padding:1px 4px; border-radius:3px;" " margin:1px 0; display:inline;"
)
_SECTION_ICONS = {
    "prepared_remarks": "📋 Prepared Remarks",
    "question": "❓ Questions",
    "answer": "💬 Answers",
}


def render_inline(results: list[dict]) -> None:
    """Render the full transcript inline with colour-coded sentences."""
    from collections import defaultdict

    sections: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        sections[r["section_type"]].append(r)

    for sec_key in ["prepared_remarks", "question", "answer"]:
        if sec_key not in sections:
            continue
        st.subheader(_SECTION_ICONS.get(sec_key, sec_key))
        parts: list[str] = []
        for r in sections[sec_key]:
            escaped = html.escape(r["text"])
            if r["label"] == "boilerplate":
                parts.append(f'<span style="{_BP_STYLE}">{escaped}</span>')
            else:
                parts.append(escaped)
        st.markdown(" ".join(parts), unsafe_allow_html=True)
        st.write("")  # spacing


def render_stats(results: list[dict]) -> None:
    """Render summary statistics panel."""
    total = len(results)
    n_bp = sum(1 for r in results if r["label"] == "boilerplate")
    n_sub = total - n_bp

    c1, c2, c3 = st.columns(3)
    c1.metric("Total sentences", total)
    c2.metric("Boilerplate", f"{n_bp} ({100*n_bp/total:.1f}%)")
    c3.metric("Substantive", f"{n_sub} ({100*n_sub/total:.1f}%)")

    st.divider()
    st.subheader("Breakdown by section")

    from collections import defaultdict

    sec_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"boilerplate": 0, "substantive": 0}
    )
    for r in results:
        sec_counts[r["section_type"]][r["label"]] += 1

    rows = []
    for sec in ["prepared_remarks", "question", "answer"]:
        if sec not in sec_counts:
            continue
        bp = sec_counts[sec]["boilerplate"]
        sub = sec_counts[sec]["substantive"]
        tot = bp + sub
        rows.append(
            {
                "Section": _SECTION_ICONS.get(sec, sec),
                "Total": tot,
                "Boilerplate": bp,
                "Substantive": sub,
                "BP %": f"{100*bp/tot:.1f}%" if tot else "—",
                "Sub %": f"{100*sub/tot:.1f}%" if tot else "—",
            }
        )

    if rows:
        import pandas as pd

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ─── Main app ─────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("📊 Boilerplate Classifier")
    st.caption("Inline tagging of earnings-call transcripts: boilerplate (red) vs. substantive.")

    # Sidebar — model info
    with st.sidebar:
        st.header("Model info")
        with open(_EVAL_DIR / "confusion_matrix_test.json", encoding="utf-8") as fh:
            cfg = json.load(fh)
        st.write(f"**Winner:** {cfg['winner']}")
        st.write(f"**Threshold:** {cfg['threshold']:.2f}")
        st.write(f"**Test macro-F1:** {cfg['test_macro_f1']:.3f}")
        st.write(f"**Test sub-recall:** {cfg['test_substantive_recall']:.3f}")
        st.divider()
        st.caption(
            "Red background = boilerplate. Plain = substantive. "
            "Recall floor ≥ 0.96 enforced on test set."
        )

    # Input method
    input_method = st.radio(
        "Input method",
        ["ECT Library", "Upload .txt file", "Paste text"],
        horizontal=True,
    )

    txt_path: Path | None = None
    raw_text: str | None = None

    if input_method == "ECT Library":
        ect_files = sorted(_ECT_DIR.glob("*.txt")) if _ECT_DIR.exists() else []
        if not ect_files:
            st.error(f"No .txt files found in {_ECT_DIR}")
            return
        file_names = [f.name for f in ect_files]
        selected = st.selectbox("Select transcript", file_names)
        txt_path = _ECT_DIR / selected

    elif input_method == "Upload .txt file":
        uploaded = st.file_uploader("Upload transcript (.txt)", type=["txt"])
        if uploaded is not None:
            content = uploaded.read().decode("utf-8-sig", errors="ignore")
            tmp = tempfile.NamedTemporaryFile(
                suffix=".txt", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(content)
            tmp.flush()
            tmp.close()
            txt_path = Path(tmp.name)

    else:  # Paste text
        raw_text = st.text_area(
            "Paste transcript text",
            height=200,
            placeholder="Paste earnings-call transcript content here…",
        )

    run_btn = st.button("Analyze", type="primary")

    if not run_btn:
        return

    if txt_path is None and not raw_text:
        st.warning("Please select or paste a transcript first.")
        return

    model, tokenizer, threshold, model_name = load_winner()

    with st.spinner("Running inference…"):
        t0 = time.perf_counter()

        if txt_path is not None:
            try:
                results = run_pipeline(txt_path, model, tokenizer, threshold)
            except TranscriptParseError:
                raw_text = txt_path.read_text(encoding="utf-8-sig", errors="ignore")
                results = run_pipeline_raw(raw_text, model, tokenizer, threshold)
        else:
            # Try ECT parse first (user may paste the full file contents)
            with tempfile.NamedTemporaryFile(
                suffix=".txt", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                tmp.write(raw_text or "")
                tmp_path = Path(tmp.name)
            try:
                results = run_pipeline(tmp_path, model, tokenizer, threshold)
            except (TranscriptParseError, Exception):
                results = run_pipeline_raw(raw_text or "", model, tokenizer, threshold)

        elapsed = time.perf_counter() - t0

    if not results:
        st.warning("No sentences extracted (transcript may be too short or malformed).")
        return

    st.success(f"Done — {len(results)} sentences in {elapsed:.1f}s")

    render_stats(results)
    st.divider()
    render_inline(results)


if __name__ == "__main__":
    main()
