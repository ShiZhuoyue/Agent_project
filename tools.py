import concurrent.futures
import html
import json
import math
import os
import re
from copy import deepcopy
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import arxiv
import requests
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from database import get_vector_db
from rag_pipeline import (
    append_sparse_records,
    build_fulltext_documents,
    download_pdf,
    format_retrieved_chunk,
    hybrid_retrieve,
    load_sparse_records,
    paper_already_indexed,
)
from research_harness import CURRENT_YEAR, DEFAULT_MIN_RELEVANCE, MAX_COUNT

from langchain_core.tools import StructuredTool

CITATION_LINE_PATTERN = re.compile(
    r"^\d+\.\s+(?P<title>.*?)\s+\|\s+source=[^|]+\s+\|\s+year=[^|]+\s+\|\s+"
    r"score=[0-9.]+\s+\|\s+semantic=[0-9.]+\s+\|\s+coverage=[0-9.]+\s+\|\s+"
    r"title_overlap=[0-9.]+\s+\|\s+focus_phrase=[0-9.]+\s+\|\s+"
    r"latent_clue=[0-9.]+\s+\|\s+citations=(?P<citations>[0-9.]+|n/a)",
    re.MULTILINE,
)

SEMANTIC_SCHOLAR_TIMEOUT_SECONDS = 8
OPENALEX_TIMEOUT_SECONDS = 10
CROSSREF_TIMEOUT_SECONDS = 8
WEB_SEARCH_TIMEOUT_SECONDS = 5
MAX_REWRITE_LIMIT = 2
MIN_CANDIDATE_POOL = 12
MAX_CANDIDATE_POOL = 30
MAX_SOURCE_QUERIES = 4
MAX_ARXIV_QUERIES = 6
MAX_WEB_RESULTS = 6
SEARCH_TERM_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+/\-]*")
SEARCH_TERM_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "any",
    "for",
    "from",
    "how",
    "into",
    "like",
    "method",
    "methods",
    "of",
    "on",
    "paper",
    "papers",
    "research",
    "result",
    "results",
    "study",
    "studies",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "would",
    "could",
    "can",
    "please",
    "find",
    "search",
    "recommend",
    "show",
    "tell",
    "first",
    "utilized",
    "utilize",
    "used",
    "include",
    "includes",
    "purpose",
    "purposes",
    "using",
    "with",
}
PHRASE_SPLIT_PATTERN = re.compile(r"\b(?:using|with|via|based on|through|under|like|such as)\b", re.IGNORECASE)
RETRIEVAL_WRAPPER_PATTERNS = (
    r"^(?:which|what)\s+paper\s+(?:first\s+)?(?:proved|showed|used|utilized|introduced|studied|demonstrated)?\s*(?:that|to|for)?\s*",
    r"^(?:where can i find|can you direct me to|could you direct me to)\s+",
    r"^(?:could you|can you|would you|please)\s+(?:recommend|find|search|look up|show|identify|direct me to)?\s*(?:research|papers?|studies|resources?|work)?\s*(?:that|which|on|for|about)?\s*",
    r"^(?:could you|can you|would you|please)\s+(?:refer me to|suggest)\s+(?:research|papers?|studies|resources?|work)?\s*(?:that|which|on|for|about)?\s*",
    r"^(?:what are some|which|what)\s+(?:research|papers?|studies|resources?|work)?\s*(?:that|which|on|for|about)?\s*",
    r"^(?:is there (?:a|any)\s+(?:paper|study|research|work)\s+)(?:that|which|exploring|about)?\s*",
    r"^(?:are there any|is there any)\s+",
)
GENERIC_FOCUS_BOUNDARY_TOKENS = {
    "data",
    "enough",
    "find",
    "improve",
    "improves",
    "improved",
    "learn",
    "method",
    "methods",
    "methodologies",
    "paper",
    "papers",
    "process",
    "purpose",
    "purposes",
    "research",
    "resource",
    "resources",
    "solve",
    "studies",
    "study",
    "task",
    "tasks",
    "trained",
    "would",
}
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO") or "eval@example.com"
TITLE_SEEKING_PATTERNS = (
    "which paper",
    "what paper",
    "where can i find",
    "could you recommend",
    "can you recommend",
    "could you suggest",
    "can you suggest",
    "refer me to research",
    "what are some studies",
    "what are some papers",
    "what are some resources",
    "recommend research",
    "recommend papers",
    "recommend studies",
    "is there a paper",
    "is there any paper",
    "first proved",
    "utilized",
    "used",
)
FOUNDATIONAL_PATTERNS = (
    "first",
    "original",
    "earliest",
    "seminal",
    "proved",
    "prove",
    "proof",
    "introduced",
)
DDG_RESULT_ANCHOR_PATTERN = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
SITE_TITLE_SUFFIX_PATTERN = re.compile(
    r"\s+-\s+(?:OpenReview|OpenReview\.net|arXiv(?:\.org)?|ACL Anthology|NASA/ADS|"
    r"ResearchGate|Google Scholar|Springer(?:\s+Nature)?|ScienceDirect|IEEE Xplore|"
    r"NSF Public Access|Paper to HTML|Paper page|Crossref|OpenAlex|OpenReview Forum)\b.*$",
    re.IGNORECASE,
)
SITE_TITLE_PREFIX_PATTERN = re.compile(r"^(?:PDF|HTML|Paper page)\s+", re.IGNORECASE)
SCHOLARLY_WEB_DOMAINS = (
    "arxiv.org",
    "openreview.net",
    "aclanthology.org",
    "semanticscholar.org",
    "doi.org",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "springer.com",
    "link.springer.com",
    "sciencedirect.com",
    "proceedings.neurips.cc",
    "papers.nips.cc",
    "proceedings.mlr.press",
    "crossref.org",
    "openalex.org",
    "researchgate.net",
    "adsabs.harvard.edu",
    "scholar.google.com",
)
AGENT_FOCUS_PATTERN = re.compile(r"\bagent(?:s|ic)?\b|智能体|agent方面|research agent", re.IGNORECASE)
AGENT_AI_CONTEXT_PATTERN = re.compile(
    r"\b(ai|artificial intelligence|llm|language model|multi-agent|autonomous agent|"
    r"intelligent agent|agentic|bdi|consensus|robot|reasoning|planning)\b|智能体|多智能体",
    re.IGNORECASE,
)
AGENT_MODERN_CONTEXT_PATTERN = re.compile(
    r"\b(llm|large language model|language model|foundation model|agentic|tool use|tool-using|"
    r"reasoning|planner|planning|autonomous agent|multi-agent|memory|retrieval-augmented|rag)\b|"
    r"大语言模型|智能体|多智能体|检索增强",
    re.IGNORECASE,
)
AGENT_NON_AI_CONTEXT_PATTERN = re.compile(
    r"\b(anticancer|hepatitis|drug|drugs|pharmac|chemical|chemotherapy|food spoilage|virus|"
    r"bacteria|disease|therapy|protein|molecule|molecules)\b|抗癌|药物|病毒|细菌",
    re.IGNORECASE,
)
RESOURCE_LOOKUP_PATTERN = re.compile(
    r"\b(dataset|datasets|resource|resources|corpus|corpora|benchmark|benchmarks|tool|tools|"
    r"lexicon|dictionary|reviews|dvd|music|books)\b",
    re.IGNORECASE,
)
BENCHMARK_LOOKUP_PATTERN = re.compile(
    r"\b(benchmark|benchmarks|evaluation|evaluate|metric|metrics|leaderboard|bleu|meteor|chrf|mmlu)\b",
    re.IGNORECASE,
)
ANALYSIS_LOOKUP_PATTERN = re.compile(
    r"\b(survey|review|analysis|analyze|study|studies|detection|detect|identify|identification)\b",
    re.IGNORECASE,
)


def _contains_non_ascii(value: str | None) -> bool:
    return any(ord(char) > 127 for char in str(value or ""))


def _dedupe_preserve(values: list[str], limit: int | None = None) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _sanitize_query_term(value)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(cleaned)
        if limit and len(unique) >= limit:
            break
    return unique


def _looks_like_broken_placeholder(value: str | None) -> bool:
    cleaned = str(value or "").strip()
    if not cleaned:
        return False
    if "?" not in cleaned:
        return False
    return not any(char.isalnum() for char in cleaned.replace("?", ""))


def _candidate_pool_size(count: int) -> int:
    return min(MAX_CANDIDATE_POOL, max(MIN_CANDIDATE_POOL, count * 5))


def _looks_like_category(value: str | None) -> bool:
    if not value:
        return False
    return "." in value or value.lower() in {"cs", "math", "physics", "stat", "econ", "q-bio", "q-fin", "eess", "astro-ph", "cond-mat"}


def _sanitize_query_term(value: str) -> str:
    return " ".join(str(value).replace('"', " ").replace("\n", " ").split())


def _strip_query_wrappers(value: str | None) -> str:
    cleaned = _sanitize_query_term(value or "")
    if not cleaned:
        return ""
    for pattern in RETRIEVAL_WRAPPER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfor the purposes of\b", " for ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:the context of|context of)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:refer me to|suggest|suggest that|is a|is an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:?!")
    return cleaned


def _should_compact_search_term(cleaned: str) -> bool:
    if not cleaned:
        return False
    lowered = cleaned.lower()
    word_count = len(lowered.split())
    if word_count >= 8:
        return True
    return any(
        marker in lowered
        for marker in (
            "?",
            "which paper",
            "what are some",
            "could you",
            "can you",
            "where can i find",
            "are there any",
            "studies that",
            "research that",
        )
    )


def _is_title_seeking_question(question: str | None) -> bool:
    lowered = str(question or "").lower()
    if not lowered:
        return False
    return any(pattern in lowered for pattern in TITLE_SEEKING_PATTERNS)


def _prefers_foundational_work(question: str | None) -> bool:
    lowered = str(question or "").lower()
    if not lowered:
        return False
    return any(pattern in lowered for pattern in FOUNDATIONAL_PATTERNS)


def _build_compact_query(value: str | None, max_terms: int = 10) -> str:
    cleaned = _strip_query_wrappers(value or "")
    if not cleaned:
        return ""
    translated = _translate_search_term(cleaned) or cleaned
    terms = _extract_search_terms(translated, max_terms=max_terms)
    return " ".join(terms) if terms else translated


def _translate_search_term(value: str | None) -> str:
    cleaned = _sanitize_query_term(value or "")
    if not cleaned:
        return ""
    if not _contains_non_ascii(cleaned) and not _should_compact_search_term(cleaned):
        return cleaned

    rewriter = _make_rewriter()
    if rewriter is None:
        return cleaned if not _contains_non_ascii(cleaned) else ""

    prompt = (
        "Rewrite the following research search phrase into concise English keywords for arXiv search.\n"
        "Return JSON only with key: query.\n"
        "Keep it short, factual, and searchable.\n"
        "Preserve rare technical terms, acronyms, kernels, metrics, and benchmark names.\n"
        "Do not explain.\n\n"
        f"phrase: {cleaned}\n"
    )
    try:
        response = rewriter.invoke(prompt)
        payload = _parse_json_object(getattr(response, "content", response))
    except Exception:
        payload = {}

    translated = _sanitize_query_term(payload.get("query") or "")
    if translated and not _contains_non_ascii(translated) and not _looks_like_broken_placeholder(translated):
        return translated
    return ""


def _extract_search_terms(value: str | None, max_terms: int = 6) -> list[str]:
    ordered_terms: list[str] = []
    seen: set[str] = set()
    for token in SEARCH_TERM_TOKEN_PATTERN.findall(str(value or "")):
        cleaned = token.strip().strip(".,;:!?()[]{}\"'")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in SEARCH_TERM_STOPWORDS:
            continue
        if len(cleaned) <= 1 and cleaned.upper() != cleaned:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered_terms.append(cleaned)
        if len(ordered_terms) >= max_terms:
            break
    return ordered_terms


def _build_keyword_clauses(field: str, value: str | None, max_terms: int = 6) -> list[str]:
    terms = _extract_search_terms(value, max_terms=max_terms)
    if not terms and value:
        fallback = _sanitize_query_term(value)
        return [f'{field}:"{fallback}"'] if fallback else []
    return [f'{field}:"{term}"' for term in terms]


def _combine_query_clauses(*clauses: str | None) -> str:
    return " AND ".join([clause for clause in clauses if clause])


def _category_query_clause(filters: dict[str, Any]) -> str:
    category = filters.get("category")
    if not category:
        return ""
    if _looks_like_category(category):
        return f"cat:{category}"
    translated_category = _translate_search_term(category)
    return f'all:"{translated_category}"' if translated_category else ""


def _build_anchor_term_queries(filters: dict[str, Any]) -> list[str]:
    combined_seed = " ".join(
        str(value or "")
        for value in (
            filters.get("paper_title"),
            filters.get("topic"),
            *(filters.get("latent_clues") or []),
            *(filters.get("query_variants") or []),
            filters.get("question"),
        )
    )
    anchor_terms = _extract_search_terms(combined_seed, max_terms=10)
    if not anchor_terms:
        return []

    subsets: list[list[str]] = []
    if len(anchor_terms) >= 3:
        subsets.append(anchor_terms[:3])
    if len(anchor_terms) >= 4:
        subsets.append(anchor_terms[1:4])
    if len(anchor_terms) >= 5:
        subsets.append(anchor_terms[:2] + anchor_terms[3:4])
    if len(anchor_terms) >= 2:
        subsets.append(anchor_terms[:2])

    queries: list[str] = []
    category_clause = _category_query_clause(filters)
    for subset in subsets:
        term_clause = _combine_query_clauses(*[f'all:"{term}"' for term in subset[:3]])
        if term_clause:
            queries.append(_combine_query_clauses(term_clause, category_clause))
    return _dedupe_preserve(queries, limit=3)


def _build_arxiv_queries(filters: dict[str, Any]) -> list[str]:
    category_clause = _category_query_clause(filters)
    queries: list[str] = []
    translated_author = _translate_search_term(filters.get("author"))
    translated_title = _translate_search_term(filters.get("paper_title"))
    translated_topic = _translate_search_term(filters.get("topic"))
    compact_topic = _build_compact_query(filters.get("topic"), max_terms=12)
    compact_question = _build_compact_query(filters.get("question"), max_terms=12)

    if translated_title:
        queries.append(_combine_query_clauses(f'ti:"{translated_title}"', category_clause))
    if translated_author:
        queries.append(_combine_query_clauses(f'au:"{translated_author}"', category_clause))

    focus_sources = [
        filters.get("paper_title"),
        *(filters.get("latent_clues") or []),
        filters.get("topic"),
        filters.get("question"),
    ]
    focus_phrases: list[str] = []
    for value in focus_sources:
        focus_phrases.extend(_extract_focus_phrases(value, max_phrases=3))
    for phrase in _dedupe_preserve(focus_phrases, limit=3):
        queries.append(_combine_query_clauses(f'all:"{phrase}"', category_clause))

    for value in (translated_topic, compact_topic, compact_question):
        if value:
            queries.append(_combine_query_clauses(f'all:"{value}"', category_clause))

    for value in filters.get("query_variants") or []:
        compact_variant = _build_compact_query(value, max_terms=12)
        if compact_variant:
            queries.append(_combine_query_clauses(f'all:"{compact_variant}"', category_clause))

    queries.extend(_build_anchor_term_queries(filters))

    if not queries:
        fallback = compact_question or translated_topic or "research papers"
        queries.append(_combine_query_clauses(f'all:"{fallback}"', category_clause))

    return _dedupe_preserve(queries, limit=MAX_ARXIV_QUERIES)


def _build_arxiv_query(filters: dict[str, Any]) -> str:
    queries = _build_arxiv_queries(filters)
    return queries[0] if queries else 'all:"research papers"'


def _build_openalex_query(filters: dict[str, Any]) -> str:
    translated_topic = _translate_search_term(filters.get("topic"))
    translated_title = _translate_search_term(filters.get("paper_title"))
    translated_author = _translate_search_term(filters.get("author"))
    translated_fallback = _translate_search_term(filters.get("question"))

    terms: list[str] = []
    for value in (translated_title, translated_topic, translated_author, translated_fallback):
        for term in _extract_search_terms(value, max_terms=8):
            if term not in terms:
                terms.append(term)
            if len(terms) >= 10:
                break
        if len(terms) >= 10:
            break
    return " ".join(terms) if terms else "research papers"


def _build_openalex_queries(filters: dict[str, Any]) -> list[str]:
    queries = [_build_openalex_query(filters)]
    if _is_title_seeking_question(filters.get("question")):
        queries.append(_build_compact_query(filters.get("question"), max_terms=12))
    for value in filters.get("query_variants") or []:
        queries.append(_build_compact_query(value, max_terms=12))
    return _dedupe_preserve(queries, limit=MAX_SOURCE_QUERIES)


def _build_crossref_queries(filters: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    for value in (
        filters.get("paper_title"),
        *(filters.get("query_variants") or []),
        filters.get("topic"),
        filters.get("question"),
    ):
        queries.append(_build_compact_query(value, max_terms=12))
    return _dedupe_preserve(queries, limit=MAX_SOURCE_QUERIES)


def _build_web_queries(filters: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    question = _sanitize_query_term(filters.get("question") or "")
    topic = _sanitize_query_term(filters.get("topic") or "")
    if question:
        queries.append(question)
    if topic:
        queries.append(topic)
    for value in filters.get("query_variants") or []:
        queries.append(_sanitize_query_term(value))
    return _dedupe_preserve(queries, limit=MAX_SOURCE_QUERIES)


def _question_capability_profile(question: str | None) -> str:
    text = str(question or "")
    if RESOURCE_LOOKUP_PATTERN.search(text):
        return "resource_lookup"
    if BENCHMARK_LOOKUP_PATTERN.search(text):
        return "benchmark_eval_lookup"
    if ANALYSIS_LOOKUP_PATTERN.search(text):
        return "analysis_or_detection_lookup"
    if _is_title_seeking_question(text):
        return "constraint_heavy_lookup"
    return "method_topic_lookup"


def _pattern_presence_score(pattern: re.Pattern[str], text: str) -> float:
    matches = pattern.findall(str(text or ""))
    if not matches:
        return 0.0
    return min(len(matches) / 3.0, 1.0)


def _agent_domain_bonus(
    question: str,
    candidate_text: str,
    categories: list[str],
    year: int | None = None,
) -> float:
    if not AGENT_FOCUS_PATTERN.search(str(question or "")):
        return 0.0
    category_text = " ".join(str(category or "") for category in categories)
    combined = f"{candidate_text}\n{category_text}"
    positive = _pattern_presence_score(AGENT_AI_CONTEXT_PATTERN, combined)
    modern = _pattern_presence_score(AGENT_MODERN_CONTEXT_PATTERN, combined)
    negative = _pattern_presence_score(AGENT_NON_AI_CONTEXT_PATTERN, combined)
    if positive <= 0 and modern <= 0 and negative > 0:
        return -0.18

    bonus = positive * 0.10 + modern * 0.10 - negative * 0.10
    if not _prefers_foundational_work(question) and year is not None:
        if year >= 2023:
            bonus += 0.05
        elif year <= 2012 and modern <= 0:
            bonus -= 0.05
    return bonus


def _llm_bootstrap_query_expansion(filters: dict[str, Any]) -> tuple[list[str], list[str]]:
    reasoner = _make_retrieval_reasoner()
    if reasoner is None:
        return [], []

    question = str(filters.get("question") or "").strip()
    if not question:
        return [], []

    prompt = (
        "You are preparing retrieval expansions for a literature-search agent before the first search.\n"
        "Return JSON only with keys clue_phrases and query_variants.\n"
        "clue_phrases: up to 4 short technical phrases likely to appear in the target paper title or abstract.\n"
        "query_variants: up to 4 concise English search queries.\n"
        "Be conservative: preserve the user's meaning, do not invent exact titles, authors, years, or venues.\n"
        "Useful transformations include aliases, paper-title style paraphrases, objective names, and neighboring terminology.\n"
        "Prefer technical wording over conversational wording.\n\n"
        f"question: {question}\n"
        f"topic: {filters.get('topic')}\n"
        f"paper_title: {filters.get('paper_title')}\n"
        f"author: {filters.get('author')}\n"
    )
    try:
        response = reasoner.invoke(prompt)
        payload = _parse_json_object(getattr(response, "content", response))
    except Exception:
        payload = {}

    clue_phrases = [
        _sanitize_query_term(value)
        for value in payload.get("clue_phrases", [])
        if isinstance(value, str) and value.strip()
    ]
    query_variants = [
        _sanitize_query_term(value)
        for value in payload.get("query_variants", [])
        if isinstance(value, str) and value.strip()
    ]
    return _dedupe_preserve(clue_phrases, limit=4), _dedupe_preserve(query_variants, limit=4)


def _bootstrap_search_filters(filters: dict[str, Any]) -> dict[str, Any]:
    bootstrapped = deepcopy(filters)
    latent_clues: list[str] = []
    for value in (filters.get("paper_title"), filters.get("topic"), filters.get("question")):
        latent_clues.extend(_extract_focus_phrases(value, max_phrases=3))

    query_variants: list[str] = []
    for value in (filters.get("paper_title"), filters.get("topic"), filters.get("question")):
        compact = _build_compact_query(value, max_terms=12)
        if compact:
            query_variants.append(compact)

    anchor_terms = _salient_query_terms(str(filters.get("question") or ""), filters, limit=8)
    for clue in _dedupe_preserve(latent_clues, limit=3):
        clue_terms = _extract_search_terms(clue, max_terms=4)
        if clue_terms and anchor_terms:
            query_variants.append(" ".join(_dedupe_preserve(clue_terms + anchor_terms[:5], limit=10)))

    combined_text = " ".join(
        str(value or "") for value in (filters.get("question"), filters.get("topic"), filters.get("paper_title"))
    )
    if AGENT_FOCUS_PATTERN.search(combined_text):
        query_variants.extend(
            [
                "AI agents",
                "LLM agents",
                "autonomous agents",
                "multi-agent systems",
            ]
        )

    llm_clues, llm_variants = _llm_bootstrap_query_expansion(filters)

    bootstrapped["latent_clues"] = _dedupe_preserve(
        list(filters.get("latent_clues") or []) + latent_clues + llm_clues,
        limit=4,
    )
    bootstrapped["query_variants"] = _dedupe_preserve(
        list(filters.get("query_variants") or []) + query_variants + llm_variants,
        limit=MAX_SOURCE_QUERIES,
    )
    return bootstrapped


def _normalize_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(title or "").lower())


def _term_in_text(term: str, text: str) -> bool:
    lowered_term = str(term or "").lower().strip()
    lowered_text = str(text or "").lower()
    if not lowered_term or not lowered_text:
        return False
    if any(marker in lowered_term for marker in ("+", "-", "/", ".")):
        return lowered_term in lowered_text
    if len(lowered_term) <= 3:
        return bool(re.search(rf"\b{re.escape(lowered_term)}\b", lowered_text))
    if " " in lowered_term:
        return lowered_term in lowered_text
    return bool(re.search(rf"\b{re.escape(lowered_term)}\b", lowered_text))


def _salient_query_terms(question: str, filters: dict[str, Any], limit: int = 10) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in (filters.get("paper_title"), filters.get("topic"), question):
        for term in _extract_search_terms(value, max_terms=limit * 2):
            lowered = term.lower()
            if lowered in seen:
                continue
            if lowered in SEARCH_TERM_STOPWORDS:
                continue
            if len(term) < 3 and not any(char.isdigit() for char in term):
                continue
            seen.add(lowered)
            terms.append(term)
            if len(terms) >= limit:
                return terms
    return terms


def _term_coverage_score(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    matches = sum(1 for term in terms if _term_in_text(term, text))
    return matches / len(terms)


def _extract_focus_phrases(text: str | None, max_phrases: int = 6) -> list[str]:
    cleaned = _strip_query_wrappers(text or "")
    if not cleaned:
        return []
    core_segment = _strip_query_wrappers(PHRASE_SPLIT_PATTERN.split(cleaned, maxsplit=1)[0])
    phrase_candidates: list[str] = []
    quoted_phrases = re.findall(r"['\"]([^'\"]{6,80})['\"]", str(text or ""))
    phrase_candidates.extend(quoted_phrases)
    for source_text in [core_segment, cleaned]:
        terms = _extract_search_terms(source_text, max_terms=18)
        for length in (4, 3, 2):
            for index in range(0, len(terms) - length + 1):
                chunk = terms[index : index + length]
                phrase = " ".join(chunk)
                lowered = phrase.lower()
                if sum(1 for token in chunk if token.lower() not in SEARCH_TERM_STOPWORDS) < 2:
                    continue
                if chunk[0].lower() in GENERIC_FOCUS_BOUNDARY_TOKENS:
                    continue
                if chunk[-1].lower() in GENERIC_FOCUS_BOUNDARY_TOKENS:
                    continue
                if len(lowered) < 10:
                    continue
                phrase_candidates.append(phrase)
    return _dedupe_preserve(phrase_candidates, limit=max_phrases)


def _focus_phrase_score(phrases: list[str], text: str) -> float:
    if not phrases:
        return 0.0
    score = 0.0
    for phrase in phrases:
        if phrase.lower() in str(text or "").lower():
            score += min(len(phrase.split()) / 4.0, 1.0)
    return min(score, 1.0)


def _reconstruct_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    max_position = max((position for positions in inverted_index.values() for position in positions), default=-1)
    if max_position < 0:
        return ""
    tokens = [""] * (max_position + 1)
    for token, positions in inverted_index.items():
        for position in positions:
            if 0 <= position < len(tokens):
                tokens[position] = token
    return " ".join(token for token in tokens if token).strip()


def _openalex_result_to_candidate(result: dict[str, Any]) -> dict[str, Any]:
    authors = [
        authorship.get("author", {}).get("display_name", "")
        for authorship in result.get("authorships", []) or []
        if authorship.get("author", {}).get("display_name")
    ]
    concepts = [
        concept.get("display_name", "")
        for concept in result.get("concepts", []) or []
        if concept.get("display_name")
    ]
    openalex_id = str(result.get("id") or "").rstrip("/").split("/")[-1]
    external_ids = result.get("ids") or {}
    pdf_url = (
        ((result.get("best_oa_location") or {}).get("pdf_url"))
        or ((result.get("open_access") or {}).get("oa_url"))
        or None
    )
    landing_page_url = (
        ((result.get("primary_location") or {}).get("landing_page_url"))
        or ((result.get("best_oa_location") or {}).get("landing_page_url"))
        or str(result.get("id") or "")
    )
    return {
        "source": "openalex",
        "result": None,
        "title": str(result.get("display_name") or "").strip(),
        "summary": _reconstruct_openalex_abstract(result.get("abstract_inverted_index")),
        "authors": authors,
        "year": result.get("publication_year"),
        "categories": concepts,
        "primary_category": concepts[0] if concepts else None,
        "arxiv_id": external_ids.get("arxiv") or f"openalex:{openalex_id}",
        "openalex_id": openalex_id,
        "paper_url": landing_page_url,
        "pdf_url": pdf_url,
        "citation_count": result.get("cited_by_count"),
        "semantic_score": 0.0,
        "time_score": 0.0,
        "citation_score": 0.0,
        "overall_score": 0.0,
        "source_rank": 999,
        "title_match": False,
        "author_match": True,
        "category_match": True,
        "year_match": True,
        "citation_match": True,
        "passes_hard_filters": True,
    }


def _search_openalex_candidates(filters: dict[str, Any], pool_size: int) -> tuple[str, list[dict[str, Any]]]:
    queries = _build_openalex_queries(filters)
    candidates: list[dict[str, Any]] = []
    for query in queries:
        params = {
            "search": query,
            "per-page": min(max(pool_size, 10), 20),
            "mailto": OPENALEX_MAILTO,
            "select": ",".join(
                [
                    "id",
                    "display_name",
                    "publication_year",
                    "authorships",
                    "concepts",
                    "abstract_inverted_index",
                    "cited_by_count",
                    "primary_location",
                    "best_oa_location",
                    "open_access",
                    "ids",
                ]
            ),
        }
        try:
            response = requests.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=OPENALEX_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            continue

        for index, item in enumerate(payload.get("results", []) or [], start=1):
            candidate = _openalex_result_to_candidate(item)
            if candidate["title"]:
                candidate["source_rank"] = index
                candidates.append(candidate)
    return " || ".join(queries), _merge_candidate_sources(candidates)


def _crossref_result_to_candidate(result: dict[str, Any]) -> dict[str, Any]:
    title_values = result.get("title") or []
    title = str(title_values[0] if title_values else "").strip()
    authors = [
        " ".join(part for part in [author.get("given"), author.get("family")] if part)
        for author in (result.get("author") or [])
        if author.get("family") or author.get("given")
    ]
    subjects = [str(subject).strip() for subject in (result.get("subject") or []) if str(subject).strip()]
    year = None
    for key in ("published-print", "published-online", "issued"):
        date_parts = ((result.get(key) or {}).get("date-parts") or [])
        if date_parts and date_parts[0]:
            maybe_year = date_parts[0][0]
            if isinstance(maybe_year, int):
                year = maybe_year
                break
    links = result.get("link") or []
    pdf_url = next(
        (link.get("URL") for link in links if "pdf" in str(link.get("content-type") or "").lower()),
        None,
    )
    doi = str(result.get("DOI") or "").strip()
    paper_url = str(result.get("URL") or "").strip() or (f"https://doi.org/{doi}" if doi else "")
    return {
        "source": "crossref",
        "result": None,
        "title": title,
        "summary": str(result.get("abstract") or "").strip(),
        "authors": authors,
        "year": year,
        "categories": subjects,
        "primary_category": subjects[0] if subjects else None,
        "arxiv_id": f"crossref:{doi}" if doi else f"crossref:{_normalize_title_key(title)}",
        "openalex_id": None,
        "paper_url": paper_url,
        "pdf_url": pdf_url,
        "citation_count": result.get("is-referenced-by-count"),
        "semantic_score": 0.0,
        "time_score": 0.0,
        "citation_score": 0.0,
        "overall_score": 0.0,
        "source_rank": 999,
        "title_match": False,
        "author_match": True,
        "category_match": True,
        "year_match": True,
        "citation_match": True,
        "passes_hard_filters": True,
    }


def _search_crossref_candidates(filters: dict[str, Any], pool_size: int) -> tuple[str, list[dict[str, Any]]]:
    queries = _build_crossref_queries(filters)
    session = requests.Session()
    candidates: list[dict[str, Any]] = []
    headers = {"User-Agent": f"agent-eval/1.0 (mailto:{OPENALEX_MAILTO})"}
    for query in queries:
        params = {
            "query.bibliographic": query,
            "rows": min(max(pool_size, 8), 12),
            "mailto": OPENALEX_MAILTO,
        }
        try:
            response = session.get(
                "https://api.crossref.org/works",
                params=params,
                headers=headers,
                timeout=CROSSREF_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json().get("message", {})
        except requests.RequestException:
            continue

        for index, item in enumerate(payload.get("items", []) or [], start=1):
            candidate = _crossref_result_to_candidate(item)
            if candidate["title"]:
                candidate["source_rank"] = index
                candidates.append(candidate)
    return " || ".join(queries), _merge_candidate_sources(candidates)


def _strip_html_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _clean_web_result_title(title: str) -> str:
    cleaned = _strip_html_tags(title)
    cleaned = SITE_TITLE_PREFIX_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned)
    cleaned = SITE_TITLE_SUFFIX_PATTERN.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" -|")


def _resolve_ddg_href(href: str) -> str:
    parsed = urlparse(str(href or ""))
    if parsed.path.startswith("/l/"):
        encoded = parse_qs(parsed.query).get("uddg", [])
        if encoded:
            return unquote(encoded[0])
    return str(href or "")


def _looks_scholarly_url(url: str) -> bool:
    parsed = urlparse(str(url or ""))
    netloc = parsed.netloc.lower()
    path = parsed.path.lower()
    if path.endswith(".pdf"):
        return True
    return any(domain in netloc for domain in SCHOLARLY_WEB_DOMAINS)


def _web_result_to_candidate(title: str, href: str) -> dict[str, Any]:
    cleaned_title = _clean_web_result_title(title)
    resolved_href = _resolve_ddg_href(href)
    pdf_url = resolved_href if resolved_href.lower().endswith(".pdf") else None
    return {
        "source": "web",
        "result": None,
        "title": cleaned_title,
        "summary": "",
        "authors": [],
        "year": None,
        "categories": [],
        "primary_category": None,
        "arxiv_id": f"web:{_normalize_title_key(cleaned_title)}",
        "openalex_id": None,
        "paper_url": resolved_href,
        "pdf_url": pdf_url,
        "citation_count": None,
        "semantic_score": 0.0,
        "time_score": 0.0,
        "citation_score": 0.0,
        "overall_score": 0.0,
        "source_rank": 999,
        "title_match": False,
        "author_match": True,
        "category_match": True,
        "year_match": True,
        "citation_match": True,
        "passes_hard_filters": True,
    }


def _search_web_candidates(filters: dict[str, Any], pool_size: int) -> tuple[str, list[dict[str, Any]]]:
    if not (_is_title_seeking_question(filters.get("question")) or filters.get("query_variants")):
        return "", []

    queries = _build_web_queries(filters)
    headers = {"User-Agent": "Mozilla/5.0"}
    candidates: list[dict[str, Any]] = []
    for query in queries:
        try:
            response = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=headers,
                timeout=WEB_SEARCH_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException:
            continue

        matches = list(DDG_RESULT_ANCHOR_PATTERN.finditer(response.text))
        for index, match in enumerate(matches[: min(MAX_WEB_RESULTS, max(pool_size, 6))], start=1):
            candidate = _web_result_to_candidate(match.group("title"), match.group("href"))
            if (
                candidate["title"]
                and len(candidate["title"].split()) >= 3
                and _looks_scholarly_url(candidate["paper_url"])
            ):
                candidate["source_rank"] = index
                candidates.append(candidate)
    return " || ".join(queries), _merge_candidate_sources(candidates)


def _merge_candidate_sources(*candidate_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for candidates in candidate_lists:
        for candidate in candidates:
            title_key = _normalize_title_key(candidate.get("title", ""))
            if not title_key or title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            merged.append(candidate)
    return merged


def _result_to_candidate(result: arxiv.Result) -> dict[str, Any]:
    authors = [author.name for author in result.authors]
    categories = list(result.categories or [])
    return {
        "source": "arxiv",
        "result": result,
        "title": result.title.strip(),
        "summary": " ".join(result.summary.split()),
        "authors": authors,
        "year": result.published.year if result.published else None,
        "categories": categories,
        "primary_category": getattr(result, "primary_category", None),
        "arxiv_id": result.entry_id.rstrip("/").split("/")[-1],
        "openalex_id": None,
        "paper_url": result.entry_id,
        "pdf_url": getattr(result, "pdf_url", None),
        "citation_count": None,
        "semantic_score": 0.0,
        "time_score": 0.0,
        "citation_score": 0.0,
        "overall_score": 0.0,
        "source_rank": 999,
        "title_match": False,
        "author_match": True,
        "category_match": True,
        "year_match": True,
        "citation_match": True,
        "passes_hard_filters": True,
    }


def _search_candidates(filters: dict[str, Any], pool_size: int) -> tuple[str, list[dict[str, Any]]]:
    openalex_query, openalex_candidates = _search_openalex_candidates(filters, pool_size)
    crossref_query, crossref_candidates = _search_crossref_candidates(filters, pool_size)
    web_query, web_candidates = _search_web_candidates(filters, pool_size)
    arxiv_candidates: list[dict[str, Any]] = []
    arxiv_query = ""
    should_query_arxiv = (
        len(openalex_candidates) < max(6, pool_size // 2)
        or bool(filters.get("paper_title"))
        or bool(filters.get("category"))
        or bool(filters.get("author"))
        or _is_title_seeking_question(filters.get("question"))
    )
    if should_query_arxiv:
        arxiv_queries = _build_arxiv_queries(filters)
        arxiv_query = " || ".join(arxiv_queries)
        client = arxiv.Client(page_size=min(pool_size, 100), delay_seconds=2.0, num_retries=3)
        per_query_results = min(max(6, math.ceil(pool_size / max(len(arxiv_queries), 1)) + 4), pool_size)
        arxiv_candidates = []
        for query_index, candidate_query in enumerate(arxiv_queries, start=1):
            search = arxiv.Search(
                query=candidate_query,
                max_results=per_query_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            try:
                results = list(client.results(search))
            except Exception:
                continue
            for rank_index, result in enumerate(results, start=1):
                candidate = _result_to_candidate(result)
                candidate["source_rank"] = min(candidate.get("source_rank", 999), rank_index + (query_index - 1) * 2)
                arxiv_candidates.append(candidate)

    merged_candidates = _merge_candidate_sources(
        openalex_candidates,
        crossref_candidates,
        web_candidates,
        arxiv_candidates,
    )
    return (
        f"openalex={openalex_query} || crossref={crossref_query} || web={web_query} || arxiv={arxiv_query}",
        merged_candidates[: max(pool_size * 3, pool_size)],
    )


def _candidate_text(candidate: dict[str, Any]) -> str:
    authors = ", ".join(candidate["authors"])
    categories = ", ".join(candidate["categories"])
    return (
        f"title: {candidate['title']}\n"
        f"summary: {candidate['summary']}\n"
        f"authors: {authors}\n"
        f"categories: {categories}\n"
        f"year: {candidate['year'] or 'unknown'}"
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _token_overlap_score(query: str, text: str) -> float:
    query_tokens = {token for token in query.lower().split() if len(token) > 1}
    text_tokens = {token for token in text.lower().split() if len(token) > 1}
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _fetch_semantic_scholar_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate.get("citation_count"), int):
        return {}

    fields = "title,year,citationCount,authors,fieldsOfStudy"
    arxiv_id = str(candidate.get("arxiv_id") or "")
    if arxiv_id and not arxiv_id.startswith("openalex:"):
        try:
            response = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}",
                params={"fields": fields},
                timeout=SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
            )
            if response.ok:
                return response.json()
        except requests.RequestException:
            pass

    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search/match",
            params={"query": candidate["title"], "fields": fields},
            timeout=SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
        )
        if response.ok:
            return response.json()
    except requests.RequestException:
        pass

    return {}


def _enrich_candidates_with_citations(candidates: list[dict[str, Any]], limit: int) -> None:
    subset = candidates[:limit]
    if not subset:
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(subset))) as executor:
        future_map = {executor.submit(_fetch_semantic_scholar_metadata, candidate): candidate for candidate in subset}
        for future in concurrent.futures.as_completed(future_map):
            candidate = future_map[future]
            try:
                payload = future.result() or {}
            except Exception:
                payload = {}

            citation_count = payload.get("citationCount")
            if isinstance(citation_count, int):
                candidate["citation_count"] = citation_count


def _text_matches(expected: str | None, actual: str) -> bool:
    if not expected:
        return True
    normalized_expected = "".join(expected.lower().split())
    normalized_actual = "".join(actual.lower().split())
    return (
        normalized_expected == normalized_actual
        or normalized_expected in normalized_actual
        or normalized_actual in normalized_expected
    )


def _author_matches(expected: str | None, authors: list[str]) -> bool:
    if not expected:
        return True
    return any(_text_matches(expected, author) for author in authors)


def _category_matches(expected: str | None, categories: list[str]) -> bool:
    if not expected:
        return True
    lowered_expected = expected.lower()
    lowered_categories = [category.lower() for category in categories]
    return any(
        lowered_expected == category
        or lowered_expected in category
        or category in lowered_expected
        for category in lowered_categories
    )


def _time_score(requested_year: int | None, candidate_year: int | None) -> tuple[float, bool]:
    if candidate_year is None:
        return 0.0, requested_year is None
    if requested_year is not None:
        delta = abs(candidate_year - requested_year)
        return max(0.0, 1.0 - (delta / 3.0)), delta == 0
    age = max(CURRENT_YEAR - candidate_year, 0)
    return max(0.0, 1.0 - min(age, 10) / 10.0), True


def _foundational_time_score(candidate_year: int | None) -> float:
    if candidate_year is None:
        return 0.0
    age = max(CURRENT_YEAR - candidate_year, 0)
    return min(age, 15) / 15.0


def _citation_score(requested_min_citations: int, actual_citations: int | None) -> tuple[float, bool]:
    if requested_min_citations <= 0:
        return 0.0, True
    if actual_citations is None:
        return 0.0, True
    return min(1.0, actual_citations / requested_min_citations), actual_citations >= requested_min_citations


def _source_rank_score(source_rank: int | None) -> float:
    if not isinstance(source_rank, int) or source_rank <= 0:
        return 0.0
    return max(0.0, 1.0 - min(source_rank - 1, 9) / 9.0)


def _score_candidates(question: str, filters: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []

    scoring_query = question or filters.get("topic") or filters.get("paper_title") or "research papers"
    _, embeddings = get_vector_db()
    candidate_texts = [_candidate_text(candidate) for candidate in candidates]

    try:
        query_vector = embeddings.embed_query(scoring_query)
        candidate_vectors = embeddings.embed_documents(candidate_texts)
        semantic_scores = [_cosine_similarity(query_vector, vector) for vector in candidate_vectors]
    except Exception:
        semantic_scores = [_token_overlap_score(scoring_query, text) for text in candidate_texts]

    min_citations = int(filters.get("citation_count") or 0)
    salient_terms = _salient_query_terms(question, filters, limit=10)
    focus_phrases = _extract_focus_phrases(filters.get("topic") or question, max_phrases=6)
    latent_clues = _dedupe_preserve([str(value) for value in filters.get("latent_clues") or []], limit=4)
    title_seeking = _is_title_seeking_question(question)
    prefers_foundational = _prefers_foundational_work(question)
    capability_profile = _question_capability_profile(question)

    for candidate, candidate_text, semantic_score in zip(candidates, candidate_texts, semantic_scores):
        time_score, year_match = _time_score(filters.get("year"), candidate["year"])
        if filters.get("year") is None and prefers_foundational:
            time_score = _foundational_time_score(candidate["year"])
        citation_score, citation_match = _citation_score(min_citations, candidate.get("citation_count"))
        source_rank_score = _source_rank_score(candidate.get("source_rank"))
        title_match = _text_matches(filters.get("paper_title"), candidate["title"])
        author_match = _author_matches(filters.get("author"), candidate["authors"])
        category_match = _category_matches(filters.get("category"), candidate["categories"])
        coverage_score = _term_coverage_score(salient_terms, candidate_text)
        title_overlap_score = _term_coverage_score(salient_terms, candidate["title"])
        focus_phrase_score = max(
            _focus_phrase_score(focus_phrases, candidate["title"]),
            _focus_phrase_score(focus_phrases, candidate_text),
        )
        latent_clue_score = max(
            _focus_phrase_score(latent_clues, candidate["title"]),
            _focus_phrase_score(latent_clues, candidate_text),
        )
        resource_signal = _pattern_presence_score(RESOURCE_LOOKUP_PATTERN, candidate_text)
        benchmark_signal = _pattern_presence_score(BENCHMARK_LOOKUP_PATTERN, candidate_text)
        analysis_signal = _pattern_presence_score(ANALYSIS_LOOKUP_PATTERN, candidate_text)
        agent_domain_bonus = _agent_domain_bonus(
            question,
            candidate_text,
            candidate["categories"],
            candidate.get("year"),
        )
        capability_bonus = 0.0
        if capability_profile == "resource_lookup":
            capability_bonus = resource_signal * 0.08 + benchmark_signal * 0.03
        elif capability_profile == "benchmark_eval_lookup":
            capability_bonus = benchmark_signal * 0.08
        elif capability_profile == "analysis_or_detection_lookup":
            capability_bonus = analysis_signal * 0.08
        elif capability_profile == "constraint_heavy_lookup":
            capability_bonus = latent_clue_score * 0.05 + title_overlap_score * 0.03

        if title_seeking:
            score = (
                semantic_score * 0.30
                + coverage_score * 0.20
                + title_overlap_score * 0.14
                + focus_phrase_score * 0.10
                + latent_clue_score * 0.18
                + time_score * 0.04
                + citation_score * 0.04
                + source_rank_score * 0.04
                + capability_bonus
                + agent_domain_bonus
            )
        else:
            score = (
                semantic_score * 0.44
                + coverage_score * 0.18
                + title_overlap_score * 0.10
                + focus_phrase_score * 0.06
                + latent_clue_score * 0.08
                + time_score * 0.12
                + citation_score * 0.04
                + source_rank_score * 0.02
                + capability_bonus
                + agent_domain_bonus
            )
        if filters.get("paper_title"):
            score += 0.22 if title_match else -0.20
        if filters.get("author"):
            score += 0.08 if author_match else -0.08
        if filters.get("category"):
            score += 0.06 if category_match else -0.06

        candidate["semantic_score"] = round(semantic_score, 4)
        candidate["coverage_score"] = round(coverage_score, 4)
        candidate["title_overlap_score"] = round(title_overlap_score, 4)
        candidate["focus_phrase_score"] = round(focus_phrase_score, 4)
        candidate["latent_clue_score"] = round(latent_clue_score, 4)
        candidate["resource_signal"] = round(resource_signal, 4)
        candidate["benchmark_signal"] = round(benchmark_signal, 4)
        candidate["analysis_signal"] = round(analysis_signal, 4)
        candidate["agent_domain_bonus"] = round(agent_domain_bonus, 4)
        candidate["capability_bonus"] = round(capability_bonus, 4)
        candidate["time_score"] = round(time_score, 4)
        candidate["citation_score"] = round(citation_score, 4)
        candidate["overall_score"] = round(score, 4)
        candidate["title_match"] = title_match
        candidate["author_match"] = author_match
        candidate["category_match"] = category_match
        candidate["year_match"] = year_match
        candidate["citation_match"] = citation_match
        candidate["passes_hard_filters"] = all(
            [
                title_match or not filters.get("paper_title"),
                author_match or not filters.get("author"),
                category_match or not filters.get("category"),
                year_match or not filters.get("year"),
                citation_match or min_citations <= 0,
            ]
        )

    return sorted(
        candidates,
        key=lambda item: (
            item["overall_score"],
            item["semantic_score"],
            item["year"] or 0,
            item.get("citation_count") or 0,
        ),
        reverse=True,
    )


def _should_rewrite(ranked_candidates: list[dict[str, Any]], filters: dict[str, Any], min_relevance_score: float) -> bool:
    if not ranked_candidates:
        return True
    top_candidate = ranked_candidates[0]
    if filters.get("paper_title") and top_candidate["title_match"]:
        return False
    if _is_title_seeking_question(filters.get("question")):
        top_overall = float(top_candidate.get("overall_score") or 0.0)
        top_coverage = float(top_candidate.get("coverage_score") or 0.0)
        top_title_overlap = float(top_candidate.get("title_overlap_score") or 0.0)
        top_focus = float(top_candidate.get("focus_phrase_score") or 0.0)
        top_latent = float(top_candidate.get("latent_clue_score") or 0.0)
        best_focus = max(
            (
                float(candidate.get("coverage_score") or 0.0)
                + float(candidate.get("title_overlap_score") or 0.0) * 0.6
            )
            for candidate in ranked_candidates[:5]
        )
        best_clue_alignment = max(
            max(
                float(candidate.get("focus_phrase_score") or 0.0),
                float(candidate.get("latent_clue_score") or 0.0),
            )
            for candidate in ranked_candidates[:5]
        )
        if best_focus < 0.55:
            return True
        if best_clue_alignment < 0.12 and top_title_overlap < 0.42:
            return True
        if top_overall < 0.52:
            return True
        if max(top_focus, top_latent) < 0.08 and top_coverage < 0.55 and top_title_overlap < 0.42:
            return True
        if top_coverage < 0.45 or top_title_overlap < 0.35:
            return True
    return top_candidate["semantic_score"] < min_relevance_score


def _make_rewriter() -> ChatOpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    if not api_key or not base_url:
        return None
    model = os.getenv("OPENAI_REWRITE_MODEL") or os.getenv("OPENAI_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
    return ChatOpenAI(model=model, temperature=0, openai_api_key=api_key, base_url=base_url, request_timeout=45)


def _make_retrieval_reasoner() -> ChatOpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    if not api_key or not base_url:
        return None
    model = os.getenv("OPENAI_RETRIEVAL_MODEL") or "Qwen/Qwen2.5-72B-Instruct"
    return ChatOpenAI(model=model, temperature=0, openai_api_key=api_key, base_url=base_url, request_timeout=45)


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text).strip()
    fenced = text.replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(fenced)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        start = fenced.find("{")
        end = fenced.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            payload = json.loads(fenced[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


def _extract_candidate_clue_phrases(question: str, ranked_candidates: list[dict[str, Any]], max_phrases: int = 4) -> list[str]:
    question_terms = {term.lower() for term in _extract_search_terms(question, max_terms=14)}
    phrase_scores: dict[str, tuple[str, int, int]] = {}
    for candidate in [item for item in ranked_candidates[:12] if item.get("source") != "web"]:
        title_terms = _extract_search_terms(candidate.get("title"), max_terms=12)
        for window_size in (3, 2, 1):
            for index in range(0, len(title_terms) - window_size + 1):
                phrase_terms = title_terms[index : index + window_size]
                lowered_terms = [term.lower() for term in phrase_terms]
                if window_size == 1 and lowered_terms[0] in question_terms:
                    continue
                if window_size == 2 and all(term in question_terms for term in lowered_terms):
                    continue
                if window_size == 3 and all(term in question_terms for term in lowered_terms):
                    continue
                if window_size == 1 and len(phrase_terms[0]) < 4:
                    continue
                if phrase_terms[0].lower() in GENERIC_FOCUS_BOUNDARY_TOKENS:
                    continue
                if phrase_terms[-1].lower() in GENERIC_FOCUS_BOUNDARY_TOKENS:
                    continue
                phrase = " ".join(phrase_terms)
                lowered_phrase = phrase.lower()
                if lowered_phrase in phrase_scores:
                    original, count, width = phrase_scores[lowered_phrase]
                    phrase_scores[lowered_phrase] = (original, count + 1, width)
                else:
                    phrase_scores[lowered_phrase] = (phrase, 1, window_size)

    ranked_phrases = sorted(
        phrase_scores.values(),
        key=lambda item: (item[1], item[2], len(item[0])),
        reverse=True,
    )
    return [phrase for phrase, _, _ in ranked_phrases[:max_phrases]]


def _build_candidate_clue_queries(
    question: str,
    current_filters: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
) -> list[str]:
    base_terms = _extract_search_terms(_build_compact_query(question, max_terms=10), max_terms=8)
    if not base_terms:
        base_terms = _salient_query_terms(question, current_filters, limit=8)
    clue_phrases = _extract_candidate_clue_phrases(question, ranked_candidates, max_phrases=4)
    if not base_terms or not clue_phrases:
        return []

    queries: list[str] = []
    first = clue_phrases[0].split()
    queries.append(" ".join(_dedupe_preserve(base_terms + first, limit=10)))
    if len(clue_phrases) > 1:
        second = clue_phrases[1].split()
        queries.append(" ".join(_dedupe_preserve(base_terms + first + second, limit=12)))
    if len(clue_phrases) > 2:
        second = clue_phrases[1].split()
        third = clue_phrases[2].split()
        queries.append(" ".join(_dedupe_preserve(base_terms + second + third, limit=12)))
    return _dedupe_preserve(queries, limit=3)


def _build_latent_clue_queries(
    question: str,
    current_filters: dict[str, Any],
    clue_phrases: list[str],
) -> list[str]:
    if not clue_phrases:
        return []

    anchor_terms = _extract_search_terms(_build_compact_query(question, max_terms=10), max_terms=8)
    if not anchor_terms:
        anchor_terms = _salient_query_terms(question, current_filters, limit=8)
    if not anchor_terms:
        return []

    queries: list[str] = []
    for clue_phrase in clue_phrases[:3]:
        clue_terms = _extract_search_terms(clue_phrase, max_terms=4)
        if not clue_terms:
            continue
        focused_anchor = anchor_terms[: max(4, min(len(anchor_terms), 6))]
        queries.append(" ".join(_dedupe_preserve(clue_terms + focused_anchor, limit=10)))
        queries.append(" ".join(_dedupe_preserve(clue_terms + anchor_terms, limit=12)))
    return _dedupe_preserve(queries, limit=3)


def _infer_latent_query_expansion(
    question: str,
    current_filters: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    heuristic_clues = _extract_candidate_clue_phrases(question, ranked_candidates, max_phrases=4)
    heuristic_queries = _build_latent_clue_queries(question, current_filters, heuristic_clues)

    reasoner = _make_retrieval_reasoner()
    if reasoner is None:
        return heuristic_clues, heuristic_queries

    rendered_candidates = [
        {
            "title": candidate.get("title"),
            "summary_hint": _sanitize_query_term(str(candidate.get("summary", ""))[:180]),
            "year": candidate.get("year"),
            "semantic_score": candidate.get("semantic_score", 0.0),
            "coverage_score": candidate.get("coverage_score", 0.0),
            "title_overlap_score": candidate.get("title_overlap_score", 0.0),
        }
        for candidate in ranked_candidates[:6]
    ]
    prompt = (
        "You are recovering hidden retrieval clues for a literature-search agent.\n"
        "The user may describe a paper indirectly by contribution, artifact, theorem, benchmark, or surrounding context.\n"
        "Infer short technical clue phrases that are likely to appear in the target paper title or abstract.\n"
        "You may hypothesize terminology only when it is strongly entailed by the user request or strongly suggested by near-miss candidates.\n"
        "Return JSON only with keys clue_phrases and query_variants.\n"
        "clue_phrases: up to 4 short English phrases, each 1 to 5 words.\n"
        "query_variants: up to 3 short English search queries that combine original request terms with the inferred clues.\n"
        "Do not output full paper titles, author names, venues, or years.\n"
        "Prefer rare technical descriptors over broad topics.\n\n"
        f"user_question: {question}\n"
        f"current_filters: {current_filters}\n"
        f"heuristic_clues: {heuristic_clues}\n"
        f"top_failed_candidates: {rendered_candidates}\n"
    )
    try:
        response = reasoner.invoke(prompt)
        payload = _parse_json_object(getattr(response, "content", response))
    except Exception:
        payload = {}

    llm_clues = [
        _sanitize_query_term(value)
        for value in payload.get("clue_phrases", [])
        if isinstance(value, str) and 1 <= len(value.split()) <= 5
    ]
    llm_queries = [
        _sanitize_query_term(value)
        for value in payload.get("query_variants", [])
        if isinstance(value, str) and value.strip()
    ]
    merged_clues = _dedupe_preserve(heuristic_clues + llm_clues, limit=4)
    merged_queries = _dedupe_preserve(
        heuristic_queries + _build_latent_clue_queries(question, current_filters, merged_clues) + llm_queries,
        limit=3,
    )
    return merged_clues, merged_queries


def _llm_rerank_candidates(
    question: str,
    filters: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    output_count: int,
) -> list[dict[str, Any]]:
    if not ranked_candidates or output_count <= 0:
        return ranked_candidates
    if not (
        _prefers_foundational_work(question)
        or _is_title_seeking_question(question)
        or bool(filters.get("latent_clues"))
        or len(str(question or "").split()) >= 11
        or "where can i find" in str(question or "").lower()
    ):
        return ranked_candidates

    reasoner = _make_retrieval_reasoner()
    if reasoner is None:
        return ranked_candidates

    candidate_window = min(len(ranked_candidates), 45 if output_count >= 10 else 30)
    shortlist = ranked_candidates[:candidate_window]
    prefers_foundational = _prefers_foundational_work(question)
    rendered_candidates = [
        (
            f"{index}. title={candidate['title']} | year={candidate.get('year') or 'n/a'} | "
            f"source={candidate.get('source', 'unknown')} | overall={candidate.get('overall_score', 0.0):.3f} | "
            f"semantic={candidate.get('semantic_score', 0.0):.3f} | coverage={candidate.get('coverage_score', 0.0):.3f} | "
            f"title_overlap={candidate.get('title_overlap_score', 0.0):.3f} | "
            f"latent_clue={candidate.get('latent_clue_score', 0.0):.3f} | "
            f"summary_hint={_sanitize_query_term(str(candidate.get('summary', ''))[:220])}"
        )
        for index, candidate in enumerate(shortlist, start=1)
    ]
    prompt = (
        "You are reranking literature-search candidates for a research agent.\n"
        "Return JSON only with key ordered_indices.\n"
        f"ordered_indices must be a list of up to {min(output_count, candidate_window)} integers chosen only from the candidate numbers.\n"
        "Choose the candidates that most likely satisfy the user's actual request.\n"
        "Prioritize exact or foundational matches over broad surveys and generic benchmark papers.\n"
        "If the query asks which paper, what paper, where to find, or who first proved something, prefer the most specific likely target paper.\n"
        "Prefer candidates that align with rare inferred clue phrases when such clues are present.\n"
        "Use summary_hint to detect matching objectives, tasks, losses, evaluation setup, or training strategy when the title alone is not enough.\n"
        "Use the provided scores as hints, not as hard rules.\n"
        f"Prefer earlier papers when foundational_preference={prefers_foundational}.\n\n"
        f"user_query: {question}\n"
        f"filters: {filters}\n"
        "candidates:\n"
        + "\n".join(rendered_candidates)
    )
    try:
        response = reasoner.invoke(prompt)
        payload = _parse_json_object(getattr(response, "content", response))
    except Exception:
        return ranked_candidates

    raw_indices = payload.get("ordered_indices")
    if not isinstance(raw_indices, list):
        return ranked_candidates

    selected_indices: list[int] = []
    seen: set[int] = set()
    for value in raw_indices:
        if not isinstance(value, int):
            continue
        if value < 1 or value > candidate_window or value in seen:
            continue
        seen.add(value)
        selected_indices.append(value)
        if len(selected_indices) >= min(output_count, candidate_window):
            break

    if not selected_indices:
        return ranked_candidates

    reordered = [shortlist[index - 1] for index in selected_indices]
    remaining = [
        candidate
        for index, candidate in enumerate(shortlist, start=1)
        if index not in seen
    ]
    tail = ranked_candidates[candidate_window:]
    return reordered + remaining + tail


def _finalize_ranked_candidates(
    question: str,
    filters: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if not ranked_candidates:
        return []
    reranked = _llm_rerank_candidates(question, filters, ranked_candidates, output_count=max(count, 1))
    filtered = [candidate for candidate in reranked if candidate["passes_hard_filters"]]
    return (filtered or reranked)[:count]


def _rewrite_search_filters(
    question: str,
    current_filters: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    rewritten = deepcopy(current_filters)
    heuristic_variants = _build_candidate_clue_queries(question, current_filters, ranked_candidates)
    latent_clues, latent_variants = _infer_latent_query_expansion(question, current_filters, ranked_candidates)
    llm_variants: list[str] = []
    reasoner = _make_retrieval_reasoner()
    if reasoner is not None:
        failures = [
            {
                "title": candidate["title"],
                "year": candidate["year"],
                "semantic_score": candidate["semantic_score"],
                "coverage_score": candidate.get("coverage_score", 0.0),
            }
            for candidate in ranked_candidates[:5]
        ]
        prompt = (
            "You are refining a paper search request after weak retrieval.\n"
            "Return JSON only with keys: topic, author, category, query_variants.\n"
            "query_variants must be a list of up to 3 short English search queries.\n"
            "Use near-miss titles only as weak clues about terminology.\n"
            "Do not invent exact paper titles, authors, or years.\n"
            "Preserve rare technical terms from the user question, including acronyms, kernels, metrics, and benchmark names.\n"
            "If you add new clue terms, do so only inside query_variants.\n\n"
            f"user_question: {question}\n"
            f"current_filters: {current_filters}\n"
            f"latent_clues: {latent_clues}\n"
            f"top_failed_candidates: {failures}\n"
        )
        try:
            response = reasoner.invoke(prompt)
            payload = _parse_json_object(getattr(response, "content", response))
        except Exception:
            payload = {}
    else:
        payload = {}

    changed = False
    for key in ("topic", "author", "category"):
        value = payload.get(key)
        if isinstance(value, str):
            cleaned = " ".join(value.split()).strip()
            if cleaned and cleaned != current_filters.get(key):
                rewritten[key] = cleaned
                changed = True
    if isinstance(payload.get("query_variants"), list):
        llm_variants = [str(value).strip() for value in payload["query_variants"] if str(value).strip()]

    merged_clues = _dedupe_preserve(list(current_filters.get("latent_clues") or []) + latent_clues, limit=4)
    if merged_clues != list(current_filters.get("latent_clues") or []):
        rewritten["latent_clues"] = merged_clues
        changed = True

    merged_variants = _dedupe_preserve(
        list(current_filters.get("query_variants") or []) + heuristic_variants + latent_variants + llm_variants,
        limit=MAX_SOURCE_QUERIES,
    )
    if merged_variants != list(current_filters.get("query_variants") or []):
        rewritten["query_variants"] = merged_variants
        changed = True

    return rewritten if changed else current_filters


def _format_candidate(candidate: dict[str, Any], rank: int) -> str:
    authors = ", ".join(candidate["authors"][:3]) or "n/a"
    categories = ", ".join(candidate["categories"][:3]) or "n/a"
    citations = candidate["citation_count"] if candidate["citation_count"] is not None else "n/a"
    summary_hint = _sanitize_query_term(candidate.get("summary") or "")
    if len(summary_hint) > 260:
        summary_hint = summary_hint[:260].rstrip() + " ..."
    line = (
        f"{rank}. {candidate['title']} | source={candidate.get('source', 'unknown')} | year={candidate['year'] or 'n/a'} | "
        f"score={candidate['overall_score']:.3f} | semantic={candidate['semantic_score']:.3f} | "
        f"coverage={candidate.get('coverage_score', 0.0):.3f} | title_overlap={candidate.get('title_overlap_score', 0.0):.3f} | "
        f"focus_phrase={candidate.get('focus_phrase_score', 0.0):.3f} | "
        f"latent_clue={candidate.get('latent_clue_score', 0.0):.3f} | "
        f"citations={citations} | authors={authors} | categories={categories}"
    )
    if summary_hint:
        line += f"\nsummary_hint: {summary_hint}"
    return line



def _enrich_selected_citations(candidates: list[dict[str, Any]], limit: int = 10) -> None:
    """Always enrich selected candidates with Semantic Scholar citation counts (no threshold)."""
    subset = [c for c in candidates if not isinstance(c.get("citation_count"), int)]
    if not subset:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(subset))) as executor:
        future_map = {executor.submit(_fetch_semantic_scholar_metadata, c): c for c in subset[:limit]}
        for future in concurrent.futures.as_completed(future_map):
            candidate = future_map[future]
            try:
                payload = future.result() or {}
            except Exception:
                payload = {}
            citation_count = payload.get("citationCount")
            if isinstance(citation_count, int):
                candidate["citation_count"] = citation_count


def _parse_citation_data(search_results: str) -> list[dict[str, Any]]:
    """Parse arxiv_research_tool output to extract paper titles and citation counts."""
    papers: list[dict[str, Any]] = []
    for match in CITATION_LINE_PATTERN.finditer(str(search_results or "")):
        title = match.group("title").strip()
        citations_raw = match.group("citations").strip()
        try:
            citations = int(citations_raw)
        except (TypeError, ValueError):
            citations = 0
        papers.append({"title": title, "citations": citations})
    return papers


def arxiv_research_tool(
    question: str,
    topic: str | None = None,
    paper_title: str | None = None,
    author: str | None = None,
    year: int | None = None,
    category: str | None = None,
    category_strict: bool = False,
    citation_count: int = 0,
    count: int = 3,
    sort_mode: str = "relevance_then_recency",
    rewrite_limit: int = 1,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE,
    **kwargs,
) -> str:
    """Search and retrieve papers from arXiv/Crossref/OpenAlex/Web with structured filters.

    This tool ONLY retrieves and ranks papers. It enriches each paper with its citation_count
    via Semantic Scholar, but performs NO statistical computation (no averaging, summing, or sorting by citations).
    To compute citation statistics (average, sum, sort), use citation_stat_tool AFTER arxiv_research_tool returns results.
    """

    bounded_count = max(1, min(int(count), MAX_COUNT))
    bounded_rewrite_limit = max(0, min(int(rewrite_limit), MAX_REWRITE_LIMIT))
    min_relevance = float(min_relevance_score or DEFAULT_MIN_RELEVANCE)

    base_filters = {
        "question": (question or "").strip(),
        "topic": (topic or "").strip() or None,
        "paper_title": (paper_title or "").strip() or None,
        "author": (author or "").strip() or None,
        "year": int(year) if year else None,
        "category": (category or "").strip() or None,
        "category_strict": bool(category_strict),
        "citation_count": max(0, int(citation_count or 0)),
        "count": bounded_count,
        "sort_mode": sort_mode or "relevance_then_recency",
        "rewrite_limit": bounded_rewrite_limit,
        "min_relevance_score": min_relevance,
        "query_variants": [],
        "latent_clues": [],
    }
    base_filters = _bootstrap_search_filters(base_filters)

    if not any([base_filters["topic"], base_filters["paper_title"], base_filters["author"], base_filters["category"]]):
        return "No actionable search filters were provided."

    pool_size = _candidate_pool_size(bounded_count)
    if _is_title_seeking_question(base_filters["question"]):
        pool_size = max(pool_size, 24)
    attempt_logs: list[str] = []
    search_filters = deepcopy(base_filters)
    final_ranked: list[dict[str, Any]] = []

    for attempt in range(bounded_rewrite_limit + 1):
        query, candidates = _search_candidates(search_filters, pool_size)
        if not candidates and search_filters["category"] and not search_filters.get("category_strict"):
            relaxed_filters = deepcopy(search_filters)
            relaxed_category = relaxed_filters["category"]
            relaxed_filters["category"] = None
            query, candidates = _search_candidates(relaxed_filters, pool_size)
            if candidates:
                attempt_logs.append(
                    f"attempt={attempt + 1} relaxed_category={relaxed_category}"
                )
                search_filters = relaxed_filters

        if search_filters["citation_count"] >= 0:
            _enrich_candidates_with_citations(candidates, limit=min(len(candidates), 10))

        ranked = _score_candidates(base_filters["question"], search_filters, candidates)
        top_semantic = ranked[0]["semantic_score"] if ranked else 0.0
        attempt_logs.append(
            f"attempt={attempt + 1} query={query} candidates={len(ranked)} top_semantic={top_semantic:.3f}"
        )

        final_ranked = ranked
        if not _should_rewrite(ranked, search_filters, min_relevance):
            break
        if attempt >= bounded_rewrite_limit:
            break

        rewritten_filters = _rewrite_search_filters(base_filters["question"], search_filters, ranked)
        if rewritten_filters == search_filters:
            break
        search_filters = rewritten_filters

    selected = _finalize_ranked_candidates(
        base_filters["question"],
        search_filters,
        final_ranked,
        bounded_count,
    )
    if not selected:
        return "No papers were found after structured retrieval and reranking.\n" + "\n".join(attempt_logs)

    # Always enrich final selected papers with Semantic Scholar citation counts (no threshold)
    _enrich_selected_citations(selected, limit=len(selected))

    db, _ = get_vector_db()
    already_indexed: list[str] = []
    to_download: list[dict[str, Any]] = []
    no_pdf_available: list[str] = []
    for candidate in selected:
        candidate_id = str(candidate.get("arxiv_id") or "")
        if candidate_id and paper_already_indexed(candidate_id):
            already_indexed.append(candidate["title"])
        elif not candidate.get("pdf_url"):
            no_pdf_available.append(candidate["title"])
        else:
            to_download.append(candidate)

    downloaded_payloads: list[dict[str, Any]] = []
    if to_download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(to_download))) as executor:
            futures = [executor.submit(download_pdf, candidate) for candidate in to_download]
            for future in concurrent.futures.as_completed(futures):
                payload = future.result()
                if payload:
                    downloaded_payloads.append(payload)

    existing_chunk_ids = {record["chunk_id"] for record in load_sparse_records()}
    indexed_titles: list[str] = []
    failed_ingestion: list[str] = []
    total_chunks = 0

    for payload in downloaded_payloads:
        documents, records = build_fulltext_documents(
            payload["candidate"],
            payload["path"],
            payload["safe_title"],
        )
        if not documents:
            failed_ingestion.append(payload["candidate"]["title"])
            continue

        docs_to_add = []
        ids_to_add: list[str] = []
        records_to_add: list[dict[str, Any]] = []
        for document, record in zip(documents, records):
            chunk_id = record["chunk_id"]
            if chunk_id in existing_chunk_ids:
                continue
            docs_to_add.append(document)
            ids_to_add.append(chunk_id)
            records_to_add.append(record)
            existing_chunk_ids.add(chunk_id)

        if not docs_to_add:
            already_indexed.append(payload["candidate"]["title"])
            continue

        try:
            db.add_documents(docs_to_add, ids=ids_to_add)
            append_sparse_records(records_to_add)
        except Exception:
            failed_ingestion.append(payload["candidate"]["title"])
            continue

        indexed_titles.append(payload["candidate"]["title"])
        total_chunks += len(docs_to_add)

    lines = [
        "Structured retrieval summary:",
        f"sort_mode={base_filters['sort_mode']}",
        *attempt_logs,
        "ranked_results:",
        *[_format_candidate(candidate, index) for index, candidate in enumerate(selected, start=1)],
    ]
    if indexed_titles:
        lines.append(f"indexed_fulltext_papers={', '.join(indexed_titles)}")
        lines.append(f"indexed_fulltext_chunks={total_chunks}")
    if already_indexed:
        lines.append(f"already_indexed={', '.join(already_indexed)}")
    if failed_ingestion:
        lines.append(f"failed_fulltext_ingestion={', '.join(failed_ingestion)}")
    if no_pdf_available:
        lines.append(f"no_open_access_pdf={', '.join(no_pdf_available)}")
    return "\n".join(lines)
# 用 StructuredTool 包装，支持多参数
arxiv_research_tool = StructuredTool.from_function(
    func=arxiv_research_tool,  # 这里指向上面的函数
    name="arxiv_research_tool",
    description="Search and retrieve papers from arXiv/Crossref/OpenAlex/Web with structured filters. This tool ONLY retrieves and ranks papers, enriching each with its citation_count via Semantic Scholar. It performs NO statistical computation — use citation_stat_tool separately for that.",
)


@tool
def citation_stat_tool(search_results: str, operation: str = "average") -> str:
    """Compute citation statistics (average, sum, sort_by_citations) from arxiv_research_tool results.

    IMPORTANT: This tool MUST be called AFTER arxiv_research_tool has returned its ranked paper results.
    Do NOT call this tool without first calling arxiv_research_tool.

    Args:
        search_results: The full output string from arxiv_research_tool containing ranked paper results.
        operation: One of "average", "sum", "sort_by_citations". Defaults to "average".
    """
    papers = _parse_citation_data(search_results)
    if not papers:
        return (
            "No paper citation data found in the provided search results. "
            "Ensure arxiv_research_tool was called first and returned valid ranked results."
        )

    if operation == "sort_by_citations":
        sorted_papers = sorted(papers, key=lambda p: p["citations"], reverse=True)
        lines = ["Papers sorted by citation count (highest first):"]
        for rank, paper in enumerate(sorted_papers, start=1):
            lines.append(f"{rank}. {paper['title']} — citations={paper['citations']}")
        return "\n".join(lines)

    citation_values = [p["citations"] for p in papers]
    total = sum(citation_values)
    count = len(citation_values)
    avg = total / count if count else 0.0

    if operation == "sum":
        lines = [
            f"Total citation count: {total} across {count} papers.",
            "Per-paper breakdown:",
        ]
        for paper in papers:
            lines.append(f"  {paper['title']} — citations={paper['citations']}")
        return "\n".join(lines)

    # Default: average
    lines = [
        f"Average citation count: {avg:.2f} across {count} papers.",
        f"Total: {total} | Min: {min(citation_values)} | Max: {max(citation_values)}",
        "Per-paper breakdown:",
    ]
    for paper in papers:
        lines.append(f"  {paper['title']} — citations={paper['citations']}")
    return "\n".join(lines)


@tool
def query_research_db(question: str) -> str:
    """Query the local research database with hybrid retrieval and reranking."""
    if not str(question or "").strip():
        return "Please provide a research question for local retrieval."
    if not load_sparse_records():
        return "No indexed full-text chunks were found in the local research database."

    docs = hybrid_retrieve(question)
    if not docs:
        return "No relevant local documents were found."
    return "\n\n".join(["Hybrid retrieval summary:"] + [format_retrieved_chunk(doc, index) for index, doc in enumerate(docs, start=1)])


def _local_chunks_for_paper_title(paper_title: str, limit: int = 4) -> list[dict[str, Any]]:
    matched_records: list[dict[str, Any]] = []
    normalized_title = str(paper_title or "").strip()
    if not normalized_title:
        return []

    for record in load_sparse_records():
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        candidate_title = str(metadata.get("paper_title") or metadata.get("source") or "").strip()
        if not candidate_title:
            continue
        if _text_matches(normalized_title, candidate_title):
            matched_records.append(record)

    matched_records.sort(
        key=lambda item: int((item.get("metadata", {}) or {}).get("chunk_index") or 0)
    )
    return matched_records[:limit]


def _remote_candidate_for_title(paper_title: str) -> dict[str, Any] | None:
    filters = {
        "question": paper_title,
        "topic": None,
        "paper_title": paper_title,
        "author": None,
        "year": None,
        "category": None,
        "category_strict": False,
        "citation_count": 0,
        "count": 1,
        "sort_mode": "relevance_then_recency",
        "rewrite_limit": 0,
        "min_relevance_score": DEFAULT_MIN_RELEVANCE,
        "query_variants": [],
        "latent_clues": [],
    }
    _, candidates = _search_candidates(filters, pool_size=8)
    ranked = _score_candidates(paper_title, filters, candidates)
    if not ranked:
        return None
    top_candidate = ranked[0]
    if _text_matches(paper_title, top_candidate.get("title", "")):
        return top_candidate
    return None


def _web_search_for_paper_abstract(paper_title: str) -> str | None:
    """Try to find a paper abstract via web search when APIs return no abstract."""
    try:
        query = f"{paper_title} abstract"
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=WEB_SEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    # Extract text snippets from search result page
    text = _strip_html_tags(response.text)
    # Clean up and truncate to a usable snippet
    text = re.sub(r"\s+", " ", text).strip()
    # Find the most relevant paragraph near the title match
    title_keywords = " ".join(paper_title.lower().split()[:4])
    lower_text = text.lower()
    idx = lower_text.find(title_keywords) if title_keywords else -1
    if idx >= 0:
        snippet = text[max(0, idx - 50):idx + 1000]
    else:
        snippet = text[:800]
    snippet = snippet.strip()
    if len(snippet) < 60:
        return None
    return f"[Web-sourced snippet — may be incomplete]\n{snippet}"


@tool
def summarize_paper_tool(paper_title: str) -> str:
    """Summarize a paper using indexed full text when available, or metadata when not."""
    normalized_title = str(paper_title or "").strip()
    if not normalized_title:
        return "Please provide the paper title you want summarized."

    local_chunks = _local_chunks_for_paper_title(normalized_title, limit=4)
    reasoner = _make_retrieval_reasoner()

    if local_chunks:
        chunk_blocks: list[str] = []
        for index, record in enumerate(local_chunks, start=1):
            metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
            chunk_blocks.append(
                (
                    f"chunk_{index} | title={metadata.get('paper_title', normalized_title)} | "
                    f"year={metadata.get('year', 'n/a')} | chunk_index={metadata.get('chunk_index', 'n/a')}\n"
                    f"{record.get('page_content', '')[:1600]}"
                )
            )
        evidence = "\n\n".join(chunk_blocks)
        if reasoner is None:
            preview = " ".join(record.get("page_content", "") for record in local_chunks[:2]).strip()
            preview = preview[:900].rstrip() + (" ..." if len(preview) > 900 else "")
            return f"Paper: {normalized_title}\nGrounded summary from indexed chunks:\n{preview}"

        prompt = (
            "You are summarizing a research paper for a literature-search agent.\n"
            "Use only the provided paper chunks.\n"
            "Write a concise grounded summary with three parts:\n"
            "1. What the paper studies\n"
            "2. Main method or contribution\n"
            "3. Why it matters\n"
            "If the evidence is partial, say that the summary is based on indexed excerpts.\n\n"
            f"paper_title: {normalized_title}\n\n"
            f"paper_chunks:\n{evidence}"
        )
        try:
            response = reasoner.invoke(prompt)
            summary = _sanitize_query_term(getattr(response, "content", response))
        except Exception:
            summary = ""
        if summary:
            return summary

    candidate = _remote_candidate_for_title(normalized_title)
    if candidate is not None:
        title = candidate.get("title") or normalized_title
        authors = ", ".join(candidate.get("authors", [])[:4]) or "n/a"
        year = candidate.get("year") or "n/a"
        summary_hint = _sanitize_query_term(candidate.get("summary") or "")
        if summary_hint:
            return (
                f"Paper: {title}\n"
                f"Year: {year}\n"
                f"Authors: {authors}\n"
                f"Grounded summary from metadata/abstract:\n{summary_hint}"
            )

        # --- 远程元数据有标题但无摘要：尝试 Web 搜索补充 ---
        web_summary = _web_search_for_paper_abstract(normalized_title)
        if web_summary:
            return f"Paper: {title}\nYear: {year}\nAuthors: {authors}\nGrounded summary from web search:\n{web_summary}"

        # --- 最后兜底：用 LLM 给出有帮助的反馈 ---
        if reasoner is not None:
            try:
                response = reasoner.invoke(
                    f"The user asked to summarize a paper titled '{title}' (year={year}). "
                    f"No abstract or full-text is available from any database. "
                    f"Write a short, honest message explaining that the paper was found but has no retrievable abstract, "
                    f"and suggest the user try searching for it on Google Scholar or Semantic Scholar directly. "
                    f"Do not fabricate any content about the paper. Keep it under 100 words."
                )
                fallback_msg = _sanitize_query_term(getattr(response, "content", response))
                if fallback_msg:
                    return f"Paper: {title}\nYear: {year}\nAuthors: {authors}\n\n{fallback_msg}"
            except Exception:
                pass
        return f"Paper: {title}\nYear: {year}\nAuthors: {authors}\nI found the paper, but only limited metadata was available for summarization."

    return "I could not find grounded paper content for that title. Try searching the paper first so it can be indexed, then ask for a summary."
