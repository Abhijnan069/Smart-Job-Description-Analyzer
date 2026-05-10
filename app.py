"""
Smart Job Description Analyzer — Classical NLP (Streamlit single-file app).
NLTK + VADER + scikit-learn TF-IDF. No transformer models.
"""

from __future__ import annotations

import contextlib
import io
import re
import threading
from collections import Counter, defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk import ne_chunk, pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Curated skills (keyword detection). Longer phrases first for greedy matching.
# ---------------------------------------------------------------------------
_TECH_SKILLS_ORDERED: tuple[str, ...] = (
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Data Engineering",
    "Data Science",
    "Computer Vision",
    "A/B Testing",
    "Power BI",
    "Tableau",
    "TensorFlow",
    "PyTorch",
    "Scikit-learn",
    "Scikit Learn",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "PostgreSQL",
    "MongoDB",
    "Kubernetes",
    "Terraform",
    "JavaScript",
    "TypeScript",
    "React",
    "Angular",
    "Node.js",
    "Django",
    "Flask",
    "FastAPI",
    "Spring Boot",
    "Apache Spark",
    "Databricks",
    "Snowflake",
    "BigQuery",
    "Redshift",
    "Airflow",
    "dbt",
    "Git",
    "Linux",
    "Docker",
    "AWS",
    "Azure",
    "GCP",
    "Statistics",
    "SQL",
    "NoSQL",
    "Python",
    "Java",
    "Scala",
    "Go",
    "C++",
    "R",
)

TECH_SKILLS: frozenset[str] = frozenset(_TECH_SKILLS_ORDERED)

_URGENCY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE)
    for p in (
        "urgent",
        "asap",
        "as soon as possible",
        "immediately",
        "deadline",
        "rolling basis",
        "start date",
        "immediate start",
        "hiring now",
        "apply today",
        "fast-paced",
        "fast paced",
    )
)


# =============================================================================
# Resource bootstrap (cached)
# =============================================================================

_NLTK_BOOTSTRAP_LOCK = threading.Lock()


@st.cache_resource(show_spinner=False)
def _ensure_nltk_resources() -> None:
    """Download required NLTK data silently (stdout/stderr suppressed)."""
    # NLTK 3.9+ splits some corpora: `*_eng` / `*_tab` IDs are required in addition to
    # legacy names (`pos_tag` → averaged_perceptron_tagger_eng; `ne_chunk` → maxent_ne_chunker_tab).
    with _NLTK_BOOTSTRAP_LOCK:
        packages = (
            "punkt",
            "punkt_tab",
            "wordnet",
            "stopwords",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
            "maxent_ne_chunker",
            "maxent_ne_chunker_tab",
            "words",
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            for pkg in packages:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    continue

        # Force-load WordNet after download to avoid LazyCorpusLoader class-mutation
        # races (observed as: "WordNetCorpusReader object has no attribute 'subdir'").
        with contextlib.suppress(Exception):
            if hasattr(wordnet, "ensure_loaded"):
                wordnet.ensure_loaded()
        with contextlib.suppress(Exception):
            if hasattr(stopwords, "ensure_loaded"):
                stopwords.ensure_loaded()


def _penn_to_wordnet(tag: str) -> str:
    if tag.startswith("J"):
        # WordNet POS tags: a=ADJ, v=VERB, n=NOUN, r=ADV
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return "n"


def _skill_pattern(skill: str) -> re.Pattern[str]:
    """Word-boundary aware pattern; allows punctuation around tokens."""
    parts = skill.split()
    if len(parts) == 1:
        return re.compile(rf"(?<![A-Za-z0-9]){re.escape(parts[0])}(?![A-Za-z0-9])", re.I)
    return re.compile(rf"\b{re.escape(skill)}\b", re.I)


def _parse_ne_tree(tree: Tree) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    if isinstance(tree, Tree):
        for child in tree:
            if isinstance(child, Tree):
                label = child.label()
                leaves = [t[0] for t in child.leaves()]
                phrase = " ".join(leaves)
                out[label].append(phrase)
            else:
                continue
    return dict(out)


def _normalize_entity_categories(
    raw: dict[str, list[str]],
) -> dict[str, list[str]]:
    persons: list[str] = []
    organizations: list[str] = []
    locations: list[str] = []
    for label, phrases in raw.items():
        u = label.upper()
        if u == "PERSON":
            persons.extend(phrases)
        elif u == "ORGANIZATION" or u == "ORG":
            organizations.extend(phrases)
        elif u in ("GPE", "LOCATION", "FACILITY"):
            locations.extend(phrases)
    def _uniq(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            k = x.strip()
            if not k or k.lower() in seen:
                continue
            seen.add(k.lower())
            out.append(k)
        return out
    return {
        "persons": _uniq(persons),
        "organizations": _uniq(organizations),
        "locations": _uniq(locations),
    }


# =============================================================================
# Module 1 — Preprocessing
# =============================================================================


def preprocess_text(text: str) -> dict[str, Any]:
    if not text or not str(text).strip():
        return {
            "sentences": [],
            "raw_tokens": [],
            "filtered_tokens": [],
            "lemmas": [],
            "pos_tags": [],
            "top_10_terms": [],
        }
    _ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stops = set(stopwords.words("english"))

    sentences = sent_tokenize(text)
    raw_tokens: list[str] = []
    for sent in sentences:
        raw_tokens.extend(word_tokenize(sent.lower()))

    alpha_tokens = [t for t in raw_tokens if t.isalpha()]
    tagged = pos_tag(alpha_tokens)
    filtered_tokens = [w for w, _ in tagged if w not in stops]
    filtered_tags = [(w, t) for w, t in tagged if w not in stops]

    lemmas: list[str] = []
    pos_tags: list[tuple[str, str]] = []
    for w, t in filtered_tags:
        pos_tags.append((w, t))
        wn_pos = _penn_to_wordnet(t)
        try:
            lemmas.append(lemmatizer.lemmatize(w, wn_pos))
        except (LookupError, AttributeError):
            # If WordNet is unavailable/corrupt, keep the token as-is instead of crashing the app.
            lemmas.append(w)

    term_freq = Counter(filtered_tokens)
    top_10 = term_freq.most_common(10)

    return {
        "sentences": sentences,
        "raw_tokens": raw_tokens,
        "filtered_tokens": filtered_tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "top_10_terms": top_10,
    }


# =============================================================================
# Module 2 — Entities & skills
# =============================================================================


def extract_entities_and_skills(text: str) -> dict[str, Any]:
    if not text or not str(text).strip():
        return {
            "skills": [],
            "entities": {"persons": [], "organizations": [], "locations": []},
            "raw_ne_labels": {},
        }
    _ensure_nltk_resources()

    detected_skills: list[str] = []
    for skill in _TECH_SKILLS_ORDERED:
        if _skill_pattern(skill).search(text):
            detected_skills.append(skill)
    # Dedupe while preserving curated order
    seen_s: set[str] = set()
    skills_ordered = []
    for s in detected_skills:
        if s.lower() in seen_s:
            continue
        seen_s.add(s.lower())
        skills_ordered.append(s)

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tree = ne_chunk(tagged, binary=False)
    raw_ne = _parse_ne_tree(tree)
    entities = _normalize_entity_categories(raw_ne)

    return {
        "skills": skills_ordered,
        "entities": entities,
        "raw_ne_labels": raw_ne,
    }


# =============================================================================
# Module 3 — Tone / sentiment (VADER)
# =============================================================================


def analyze_tone(text: str) -> dict[str, Any]:
    if not text or not str(text).strip():
        return {
            "compound": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 1.0,
            "tone_label": "Neutral/Demanding",
            "urgency_hits": [],
        }
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = float(scores["compound"])
    tone_label = "Positive/Welcoming" if compound >= 0.05 else "Neutral/Demanding"

    urgency_hits: list[str] = []
    for pat in _URGENCY_PATTERNS:
        for m in pat.finditer(text):
            urgency_hits.append(m.group(0))
    # Dedupe case-insensitively, keep first casing
    seen_u: set[str] = set()
    uniq_urgency: list[str] = []
    for u in urgency_hits:
        k = u.lower()
        if k in seen_u:
            continue
        seen_u.add(k)
        uniq_urgency.append(u)

    return {
        "compound": compound,
        "pos": float(scores["pos"]),
        "neg": float(scores["neg"]),
        "neu": float(scores["neu"]),
        "tone_label": tone_label,
        "urgency_hits": uniq_urgency,
    }


# =============================================================================
# Module 4 — TF-IDF similarity
# =============================================================================


def compute_similarity(jd_text: str, resume_text: str) -> dict[str, Any]:
    jd_text = (jd_text or "").strip()
    resume_text = (resume_text or "").strip()
    if not jd_text or not resume_text:
        return {
            "similarity_score": 0.0,
            "similarity_percent": 0.0,
            "matched_keywords": [],
            "gap_keywords": [],
            "feature_names": np.array([]),
            "jd_vector": None,
            "resume_vector": None,
        }

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    mat = vectorizer.fit_transform([jd_text, resume_text])
    jd_vec = mat[0]
    res_vec = mat[1]
    sim = float(cosine_similarity(jd_vec, res_vec)[0, 0])

    names = vectorizer.get_feature_names_out()
    jd_d = jd_vec.toarray().ravel()
    res_d = res_vec.toarray().ravel()

    eps = 1e-9
    jd_threshold = max(0.02, np.percentile(jd_d[jd_d > 0], 75) if np.any(jd_d > 0) else 0.02)

    matched: list[dict[str, Any]] = []
    gap: list[dict[str, Any]] = []
    for i, term in enumerate(names):
        jw, rw = float(jd_d[i]), float(res_d[i])
        if jw >= jd_threshold and rw > eps:
            matched.append({"term": term, "jd_weight": jw, "resume_weight": rw})
        elif jw >= jd_threshold and rw <= eps:
            gap.append({"term": term, "jd_weight": jw})

    matched.sort(key=lambda x: min(x["jd_weight"], x["resume_weight"]), reverse=True)
    gap.sort(key=lambda x: x["jd_weight"], reverse=True)

    return {
        "similarity_score": sim,
        "similarity_percent": round(sim * 100.0, 2),
        "matched_keywords": matched[:40],
        "gap_keywords": gap[:40],
        "feature_names": names,
        "jd_vector": jd_d,
        "resume_vector": res_d,
    }


# =============================================================================
# Visualization helpers (pure matplotlib; no NLP)
# =============================================================================


def _fig_pos_distribution(pos_tags: list[tuple[str, str]]) -> plt.Figure:
    coarse = Counter()
    for _, t in pos_tags:
        if t.startswith("J"):
            coarse["ADJ"] += 1
        elif t.startswith("V"):
            coarse["VERB"] += 1
        elif t.startswith("N"):
            coarse["NOUN"] += 1
        elif t.startswith("R"):
            coarse["ADV"] += 1
        else:
            coarse["OTHER"] += 1
    labels = list(coarse.keys())
    vals = [coarse[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(labels, vals, color="#7dd3fc", edgecolor="#2e3147")
    ax.set_facecolor("#1a1d27")
    fig.patch.set_facecolor("#0f1117")
    ax.tick_params(colors="#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")
    ax.xaxis.label.set_color("#e5e7eb")
    ax.set_title("POS tag distribution (coarse)", color="#fde68a", fontsize=11)
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def _fig_top_terms(top_10: list[tuple[str, int]]) -> plt.Figure:
    if not top_10:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.text(0.5, 0.5, "No terms", ha="center", va="center", color="#e5e7eb")
        ax.axis("off")
        fig.patch.set_facecolor("#0f1117")
        return fig
    terms = [t for t, _ in reversed(top_10)]
    counts = [c for _, c in reversed(top_10)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(terms, counts, color="#86efac", edgecolor="#2e3147")
    ax.set_facecolor("#1a1d27")
    fig.patch.set_facecolor("#0f1117")
    ax.tick_params(colors="#e5e7eb")
    ax.set_title("Top 10 term frequencies (JD)", color="#fde68a", fontsize=11)
    plt.tight_layout()
    return fig


def _fig_tfidf_compare(sim_result: dict[str, Any], max_terms: int = 14) -> plt.Figure:
    matched = sim_result.get("matched_keywords") or []
    gap = sim_result.get("gap_keywords") or []
    take_m = matched[: max_terms // 2]
    take_g = gap[: max(4, max_terms - len(take_m))]

    rows: list[tuple[str, float, float, str]] = []
    for m in take_m:
        rows.append((m["term"], m["jd_weight"], m["resume_weight"], "matched"))
    for g in take_g:
        rows.append((g["term"], g["jd_weight"], 0.0, "gap"))

    rows = rows[:max_terms]
    if not rows:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.text(0.5, 0.5, "Not enough TF-IDF signal", ha="center", va="center", color="#e5e7eb")
        ax.axis("off")
        fig.patch.set_facecolor("#0f1117")
        return fig

    terms = [r[0] for r in rows]
    jd_w = [r[1] for r in rows]
    res_w = [r[2] for r in rows]
    kinds = [r[3] for r in rows]

    y = np.arange(len(terms))
    height = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(y - height / 2, jd_w, height, label="JD TF-IDF", color="#7dd3fc", edgecolor="#2e3147")
    ax.barh(y + height / 2, res_w, height, label="Resume TF-IDF", color="#86efac", edgecolor="#2e3147")
    for i, k in enumerate(kinds):
        if k == "gap":
            ax.axhspan(y[i] - 0.5, y[i] + 0.5, facecolor="#fca5a5", alpha=0.08, zorder=0)
    ax.set_yticks(y, labels=terms, fontsize=9)
    ax.set_xlabel("Weight")
    ax.set_facecolor("#1a1d27")
    fig.patch.set_facecolor("#0f1117")
    ax.tick_params(colors="#e5e7eb")
    ax.set_title("TF-IDF: JD vs resume (matched + gaps)", color="#fde68a", fontsize=11)
    ax.legend(facecolor="#1a1d27", edgecolor="#2e3147", labelcolor="#e5e7eb")
    plt.tight_layout()
    return fig


# =============================================================================
# Streamlit UI (presentation only)
# =============================================================================

_DEFAULT_JD = """\
Data Scientist — Analytics & Modeling

We are seeking a Data Scientist to join our growing analytics team. You will work with
stakeholders to frame problems, build predictive models, and communicate insights clearly.

Responsibilities:
- Analyze large datasets with Python, SQL, and pandas; deliver dashboards in collaboration
  with product partners.
- Apply statistics and machine learning to improve forecasting and experimentation (A/B tests).
- Partner with data engineering to productionize models and ensure data quality.

Requirements:
- Strong Python skills and experience with scikit-learn; familiarity with TensorFlow or
  PyTorch is a plus.
- Solid SQL and experience with cloud platforms (AWS preferred); Docker experience helpful.
- Excellent communication; comfortable in a fast-paced, collaborative environment.

This is an urgent hire — please apply by the deadline listed on our careers page. We are an
equal opportunity employer and welcome diverse candidates.
"""

_DEFAULT_RESUME = """\
Alex Rivera
Data Scientist

Summary:
Data scientist with 4+ years building predictive models and analytics pipelines in Python and SQL.
Experienced with machine learning, statistics, and experimentation on AWS.

Skills:
Python, SQL, pandas, NumPy, scikit-learn, TensorFlow (basics), Docker, AWS (S3, Lambda),
Git, statistics, A/B testing, data visualization (matplotlib).

Experience:
- Built gradient boosting models for demand forecasting; improved accuracy and reduced inventory costs.
- Owned feature engineering and model monitoring pipelines deployed on AWS with Docker.
- Partnered with engineering to productionize scoring services and SQL-based reporting.

Education:
M.S. Statistics; coursework in machine learning and experimental design.

Note: Excited about collaborative teams and clear communication with stakeholders.
"""


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'IBM Plex Sans', sans-serif;
            color: #e5e7eb;
        }
        .stApp {
            background-color: #0f1117;
        }
        h1, h2, h3, h4 {
            font-family: 'IBM Plex Mono', monospace !important;
            letter-spacing: -0.02em;
        }
        div[data-testid="stVerticalBlock"] > div {
            gap: 0.75rem;
        }
        .metric-card {
            background: #1a1d27;
            border: 1px solid #2e3147;
            border-radius: 10px;
            padding: 14px 16px;
            min-height: 92px;
        }
        .metric-card .label {
            font-size: 0.78rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .metric-card .value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.45rem;
            margin-top: 6px;
        }
        .metric-card .accent-blue { color: #7dd3fc; }
        .metric-card .accent-green { color: #86efac; }
        .metric-card .accent-red { color: #fca5a5; }
        .metric-card .accent-yellow { color: #fde68a; }
        .block-container { padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, accent: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value accent-{accent}">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _build_conclusion(
    sim_pct: float,
    tone_label: str,
    jd_skills: list[str],
    resume_skills: list[str],
    gap_terms: list[dict[str, Any]],
) -> str:
    jd_set = {s.lower() for s in jd_skills}
    res_set = {s.lower() for s in resume_skills}
    overlap = sorted(jd_set & res_set)
    missing = sorted(jd_set - res_set)

    if sim_pct >= 70:
        fit = "strong textual overlap with the job description"
    elif sim_pct >= 45:
        fit = "moderate overlap; the resume addresses several JD themes but not all"
    else:
        fit = "limited overlap; consider aligning wording and highlighted accomplishments with the JD"

    parts = [
        f"**Match score:** {sim_pct:.1f}% — {fit}.",
        f"**Tone:** The posting reads as *{tone_label}* per VADER compound sentiment.",
    ]
    if overlap:
        parts.append(
            "**Skill alignment (keyword hits):** "
            + ", ".join(f"`{s}`" for s in overlap[:12])
            + ("…" if len(overlap) > 12 else "")
            + "."
        )
    if missing:
        parts.append(
            "**Skills mentioned in the JD but not keyword-detected in the resume:** "
            + ", ".join(f"`{s}`" for s in missing[:10])
            + ("…" if len(missing) > 10 else "")
            + ". Consider adding evidence if you have it."
        )
    if gap_terms:
        top_gaps = ", ".join(f"`{g['term']}`" for g in gap_terms[:8])
        parts.append(
            f"**TF-IDF gaps (emphasized in the JD, absent in the resume):** {top_gaps}."
        )
    parts.append(
        "**Verdict:** Use the gap lists to tune your resume bullets; similarity is lexical (TF-IDF), "
        "not a guarantee of qualification."
    )
    return "\n\n".join(parts)


def main() -> None:
    st.set_page_config(
        page_title="Smart Job Description Analyzer",
        page_icon="📋",
        layout="wide",
    )
    _inject_css()

    st.title("Smart Job Description Analyzer")
    st.caption("Classical NLP — NLTK · VADER · scikit-learn TF-IDF (no transformers).")

    c1, c2 = st.columns(2)
    with c1:
        jd_input = st.text_area("Job description", value=_DEFAULT_JD, height=280, key="jd")
    with c2:
        resume_input = st.text_area("Resume", value=_DEFAULT_RESUME, height=280, key="resume")

    pre_jd = preprocess_text(jd_input)
    pre_res = preprocess_text(resume_input)
    ent_jd = extract_entities_and_skills(jd_input)
    ent_res = extract_entities_and_skills(resume_input)
    tone_jd = analyze_tone(jd_input)
    sim = compute_similarity(jd_input, resume_input)

    st.subheader("Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        _metric_card("Sentences (JD)", str(len(pre_jd["sentences"])), "blue")
    with m2:
        _metric_card("Tokens (JD, raw)", str(len(pre_jd["raw_tokens"])), "blue")
    with m3:
        _metric_card("Skills found (JD)", str(len(ent_jd["skills"])), "green")
    with m4:
        _metric_card("Sentiment (compound)", f"{tone_jd['compound']:.3f}", "yellow")
    with m5:
        _metric_card("Match % (TF-IDF)", f"{sim['similarity_percent']:.1f}%", "green")

    st.subheader("Visualizations")
    v1, v2 = st.columns(2)
    with v1:
        st.pyplot(
            _fig_pos_distribution(pre_jd["pos_tags"]),
            use_container_width=True,
            clear_figure=True,
        )
    with v2:
        st.pyplot(
            _fig_top_terms(pre_jd["top_10_terms"]),
            use_container_width=True,
            clear_figure=True,
        )
    st.pyplot(_fig_tfidf_compare(sim), use_container_width=True, clear_figure=True)

    st.subheader("Details")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Named entities (JD)**")
        st.json(ent_jd["entities"])
        st.markdown("**Skills — JD**")
        st.write(ent_jd["skills"] or ["—"])
        st.markdown("**Skills — Resume**")
        st.write(ent_res["skills"] or ["—"])
    with d2:
        st.markdown("**VADER scores**")
        st.json(
            {
                k: tone_jd[k]
                for k in ("compound", "pos", "neg", "neu", "tone_label", "urgency_hits")
            }
        )
        st.markdown("**Matched keywords (TF-IDF)**")
        st.dataframe(pd.DataFrame(sim["matched_keywords"]).head(20), use_container_width=True)
        st.markdown("**Gap keywords (TF-IDF)**")
        st.dataframe(pd.DataFrame(sim["gap_keywords"]).head(20), use_container_width=True)

    st.subheader("Conclusion")
    st.markdown(
        _build_conclusion(
            float(sim["similarity_percent"]),
            str(tone_jd["tone_label"]),
            list(ent_jd["skills"]),
            list(ent_res["skills"]),
            list(sim["gap_keywords"]),
        )
    )


if __name__ == "__main__":
    main()
