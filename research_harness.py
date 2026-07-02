import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

DEFAULT_COUNT = 3
MAX_COUNT = 10
MAX_CITATION_COUNT = 100000
CURRENT_YEAR = datetime.utcnow().year
DEFAULT_SORT_MODE = "relevance_then_recency"
DEFAULT_REWRITE_LIMIT = 1
DEFAULT_MIN_RELEVANCE = 0.35

SUMMARY_HINTS = ("summarize", "summary", "analyze", "analysis", "overview", "总结", "概述", "分析", "解读")
LOCAL_DB_HINTS = (
    "local db",
    "local database",
    "vector db",
    "vector database",
    "local knowledge base",
    "our database",
    "my database",
    "indexed papers",
    "downloaded papers",
    "downloaded pdfs",
    "本地库",
    "向量库",
    "数据库",
    "已下载",
)
SEARCH_HINTS = (
    "search",
    "find",
    "look up",
    "recommend",
    "study",
    "studies",
    "research",
    "resource",
    "resources",
    "dataset",
    "datasets",
    "tool",
    "tools",
    "where can i find",
    "could you recommend",
    "can you recommend",
    "paper",
    "papers",
    "article",
    "articles",
    "arxiv",
    "literature",
    "survey",
    "搜",
    "搜索",
    "查找",
    "找",
    "检索",
    "论文",
    "文献",
)
RECENT_HINTS = ("recent", "latest", "newest", "最近", "最新")

CATEGORY_ALIASES = {
    "computer vision": "cs.CV",
    "vision": "cs.CV",
    "cv": "cs.CV",
    "nlp": "cs.CL",
    "natural language processing": "cs.CL",
    "computational linguistics": "cs.CL",
    "machine learning": "cs.LG",
    "ml": "cs.LG",
    "reinforcement learning": "cs.LG",
    "robotics": "cs.RO",
    "robot": "cs.RO",
    "multimodal": "cs.MM",
    "multi-modal": "cs.MM",
    "artificial intelligence": "cs.AI",
    "ai": "cs.AI",
    "information retrieval": "cs.IR",
    "security": "cs.CR",
    "cryptography": "cs.CR",
    "drone": "cs.RO",
    "drones": "cs.RO",
    "uav": "cs.RO",
    "autonomous driving": "cs.RO",
    "self-driving": "cs.RO",
    "无人机": "cs.RO",
    "自动驾驶": "cs.RO",
}

ARXIV_CATEGORY_PATTERN = re.compile(
    r"\b(?:astro-ph|cond-mat|cs|econ|eess|math|physics|q-bio|q-fin|stat)(?:\.[A-Za-z\-]+)?\b",
    re.IGNORECASE,
)
TECHNICAL_TOKEN_PATTERN = re.compile(
    r"\b(?:[A-Z]{2,}[A-Za-z0-9+\-]*|[A-Za-z0-9+]+(?:[-/][A-Za-z0-9+]+)+|[A-Za-z]*\d+[A-Za-z0-9+\-]*)\b"
)


@dataclass(frozen=True)
class HarnessDecision:
    should_call_agent: bool
    intent: str
    topic: str | None
    paper_title: str | None
    author: str | None
    year: int | None
    category: str | None
    citation_count: int
    count: int
    reason: str
    normalized_query: str
    signals: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0


def build_structured_request(raw_query: str, planner_payload: dict | None = None) -> dict:
    planner_payload = planner_payload or {}
    cleaned_payload = _normalize_planner_payload(planner_payload)
    query = _normalize_whitespace(raw_query)
    ignored_intents = cleaned_payload["ignored_intents"]
    risk_flags: list[str] = []

    if ignored_intents:
        risk_flags.append("ignored_non_research_intents")

    count = _normalize_count(
        cleaned_payload["count"] or _extract_requested_count(query) or DEFAULT_COUNT,
        risk_flags,
    )
    year = _normalize_year(
        cleaned_payload["year"] or _extract_year(query),
        risk_flags,
    )
    citation_count = _normalize_citation_count(
        cleaned_payload["citation_count"] or _extract_citation_count(query) or 0,
        risk_flags,
    )
    author = _normalize_author(cleaned_payload["author"] or _extract_author(query), risk_flags)
    explicit_category = _extract_category(query)
    category = _normalize_category(explicit_category, risk_flags)
    paper_title = _normalize_text(
        cleaned_payload["paper_title"] or _extract_paper_title(query),
        240,
        risk_flags,
        "paper_title",
    )
    heuristic_topic = _extract_topic(query)
    topic = _choose_topic(cleaned_payload["topic"], heuristic_topic, query, risk_flags)
    topic = _normalize_text(topic, 260, risk_flags, "topic")
    topic = _clean_search_topic(topic)
    if _looks_like_ai_agent_request(query, topic):
        if not category:
            category = "cs.AI"
            risk_flags.append("category_defaulted_to_cs.AI_for_agent_query")
        topic = _normalize_ai_agent_topic(topic, query)

    sort_mode = DEFAULT_SORT_MODE
    rewrite_limit = DEFAULT_REWRITE_LIMIT
    min_relevance_score = DEFAULT_MIN_RELEVANCE

    planner_wants_local_db = cleaned_payload["intent"] == "query_local_db"
    explicit_local_db = _is_local_db_query(query)
    if planner_wants_local_db and not explicit_local_db:
        risk_flags.append("planner_local_db_overruled")

    if explicit_local_db:
        question = cleaned_payload["question"] or query
        return {
            "intent": "query_local_db",
            "question": question,
            "topic": None,
            "paper_title": None,
            "author": None,
            "year": None,
            "category": None,
            "citation_count": 0,
            "count": None,
            "sort_mode": None,
            "rewrite_limit": 0,
            "min_relevance_score": None,
            "ignored_intents": ignored_intents,
            "risk_flags": risk_flags,
            "final_mode": "synthesize",
            "response": None,
        }

    summary_request = cleaned_payload["intent"] == "summarize_paper" or (
        paper_title is not None and _has_any_hint(query, SUMMARY_HINTS)
    )
    if summary_request:
        if not paper_title:
            return _build_clarification(ignored_intents, risk_flags, "Please specify the paper title you want summarized.")
        return {
            "intent": "summarize_paper",
            "question": None,
            "topic": None,
            "paper_title": paper_title,
            "author": None,
            "year": None,
            "category": None,
            "citation_count": 0,
            "count": None,
            "sort_mode": None,
            "rewrite_limit": 0,
            "min_relevance_score": None,
            "ignored_intents": ignored_intents,
            "risk_flags": risk_flags,
            "final_mode": "synthesize",
            "response": None,
        }

    actionable_filters = any([topic, paper_title, author, category])
    explicit_search_intent = cleaned_payload["intent"] in {"search_papers", "citation_stats"} or _looks_like_search_request(query)
    if explicit_search_intent and (actionable_filters or cleaned_payload["intent"] == "citation_stats"):
        if topic is None and paper_title is None:
            topic = _fallback_topic_from_filters(author, category, year)
        # Preserve the original intent (citation_stats vs search_papers) for execution plan routing
        resolved_intent = "citation_stats" if cleaned_payload["intent"] == "citation_stats" else "search_papers"
        return {
            "intent": resolved_intent,
            "question": query,
            "topic": topic,
            "paper_title": paper_title,
            "author": author,
            "year": year,
            "category": category,
            "category_strict": bool(explicit_category),
            "citation_count": citation_count,
            "count": count,
            "sort_mode": sort_mode,
            "rewrite_limit": rewrite_limit,
            "min_relevance_score": min_relevance_score,
            "ignored_intents": ignored_intents,
            "risk_flags": risk_flags,
            "final_mode": "synthesize",
            "response": None,
        }

    clarification = cleaned_payload["clarification_question"] or "Please clarify the paper topic, title, author, or category you want to search."
    return _build_clarification(ignored_intents, risk_flags, clarification)


def build_execution_plan(normalized_request: dict) -> list[dict]:
    intent = normalized_request.get("intent")

    if intent == "citation_stats":
        # Multi-step plan: first retrieve, then compute statistics
        count = normalized_request.get("count") or 5
        plan = [
            {
                "step_type": "tool",
                "tool_name": "arxiv_research_tool",
                "args": {
                    "question": normalized_request["question"],
                    "topic": normalized_request["topic"],
                    "paper_title": normalized_request["paper_title"],
                    "author": normalized_request["author"],
                    "year": normalized_request["year"],
                    "category": normalized_request["category"],
                    "category_strict": normalized_request.get("category_strict", False),
                    "citation_count": 0,
                    "count": count,
                    "sort_mode": normalized_request.get("sort_mode") or "relevance_then_recency",
                    "rewrite_limit": normalized_request.get("rewrite_limit") or 1,
                    "min_relevance_score": normalized_request.get("min_relevance_score") or 0.35,
                },
                "goal": "Step 1/2: Retrieve papers and enrich each with citation_count via Semantic Scholar.",
            },
            {
                "step_type": "tool",
                "tool_name": "citation_stat_tool",
                "args": {
                    "operation": normalized_request.get("stat_operation") or "average",
                },
                "goal": "Step 2/2: Compute citation statistics (average/sum/sort) from the retrieved papers.",
            },
        ]
        return plan

    if intent == "search_papers":
        return [
            {
                "step_type": "tool",
                "tool_name": "arxiv_research_tool",
                "args": {
                    "question": normalized_request["question"],
                    "topic": normalized_request["topic"],
                    "paper_title": normalized_request["paper_title"],
                    "author": normalized_request["author"],
                    "year": normalized_request["year"],
                    "category": normalized_request["category"],
                    "category_strict": normalized_request.get("category_strict", False),
                    "citation_count": normalized_request["citation_count"],
                    "count": normalized_request["count"],
                    "sort_mode": normalized_request["sort_mode"],
                    "rewrite_limit": normalized_request["rewrite_limit"],
                    "min_relevance_score": normalized_request["min_relevance_score"],
                },
                "goal": "Search papers with a structured request, rerank by relevance and recency, and retry with a rewritten query if needed.",
            }
        ]

    if intent == "query_local_db":
        return [
            {
                "step_type": "tool",
                "tool_name": "query_research_db",
                "args": {"question": normalized_request["question"]},
                "goal": "Query the local research database.",
            }
        ]

    if intent == "summarize_paper":
        return [
            {
                "step_type": "tool",
                "tool_name": "summarize_paper_tool",
                "args": {"paper_title": normalized_request["paper_title"]},
                "goal": "Summarize the requested paper.",
            }
        ]

    return [{"step_type": "respond", "content": normalized_request.get("response", "Please clarify your request.")}]


def inspect_research_request(raw_query: str) -> HarnessDecision:
    started_at = time.perf_counter()
    request = build_structured_request(raw_query, {})
    should_call_agent = request["intent"] != "clarify"
    reason = "structured_request_ready" if should_call_agent else "needs_clarification"
    normalized_query = _normalize_whitespace(raw_query)
    return HarnessDecision(
        should_call_agent=should_call_agent,
        intent=request["intent"],
        topic=request.get("topic"),
        paper_title=request.get("paper_title"),
        author=request.get("author"),
        year=request.get("year"),
        category=request.get("category"),
        citation_count=request.get("citation_count", 0) or 0,
        count=request.get("count", DEFAULT_COUNT) or DEFAULT_COUNT,
        reason=reason,
        normalized_query=normalized_query,
        signals={
            "search_like": _looks_like_search_request(normalized_query),
            "local_db_like": _is_local_db_query(normalized_query),
            "summary_like": _has_any_hint(normalized_query, SUMMARY_HINTS),
        },
        latency_seconds=round(time.perf_counter() - started_at, 4),
    )


def _build_clarification(ignored_intents: list[str], risk_flags: list[str], message: str) -> dict:
    return {
        "intent": "clarify",
        "question": None,
        "topic": None,
        "paper_title": None,
        "author": None,
        "year": None,
        "category": None,
        "citation_count": 0,
        "count": None,
        "sort_mode": None,
        "rewrite_limit": 0,
        "min_relevance_score": None,
        "ignored_intents": ignored_intents,
        "risk_flags": risk_flags,
        "final_mode": "passthrough",
        "response": message,
    }


def _normalize_planner_payload(payload: dict) -> dict:
    return {
        "intent": str(payload.get("intent", "")).strip().lower(),
        "topic": _clean_optional_text(payload.get("topic")),
        "paper_title": _clean_optional_text(payload.get("paper_title")),
        "author": _clean_optional_text(payload.get("author")),
        "question": _clean_optional_text(payload.get("question")),
        "clarification_question": _clean_optional_text(payload.get("clarification_question")),
        "category": _clean_optional_text(payload.get("category")),
        "count": _safe_int(payload.get("count")),
        "year": _safe_int(payload.get("year")),
        "citation_count": _safe_int(payload.get("citation_count")),
        "ignored_intents": [str(item).strip() for item in payload.get("ignored_intents", []) if str(item).strip()],
    }


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text).replace("\u3000", " ").split())


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    cleaned = _normalize_whitespace(str(value)).strip(" []{}\"'`.,!?;:()[]")
    return cleaned or None


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_chinese_count(token: str) -> int | None:
    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    cleaned = str(token or "").strip()
    if not cleaned:
        return None
    if cleaned == "十":
        return 10
    if cleaned in mapping:
        return mapping[cleaned]
    if cleaned.startswith("十") and len(cleaned) == 2 and cleaned[1] in mapping:
        return 10 + mapping[cleaned[1]]
    if cleaned.endswith("十") and len(cleaned) == 2 and cleaned[0] in mapping:
        return mapping[cleaned[0]] * 10
    if "十" in cleaned:
        left, _, right = cleaned.partition("十")
        left_value = mapping.get(left, 1 if left == "" else None)
        right_value = mapping.get(right, 0 if right == "" else None)
        if left_value is not None and right_value is not None:
            return left_value * 10 + right_value
    return None


def _normalize_text(value: str | None, max_length: int, risk_flags: list[str], field_name: str) -> str | None:
    if not value:
        return None
    cleaned = _clean_optional_text(value)
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        risk_flags.append(f"{field_name}_truncated")
        cleaned = cleaned[:max_length].rstrip()
    return cleaned


def _normalize_count(value: int, risk_flags: list[str]) -> int:
    bounded = max(1, min(int(value), MAX_COUNT))
    if bounded != int(value):
        risk_flags.append(f"count_capped_to_{bounded}")
    return bounded


def _normalize_year(value: int | None, risk_flags: list[str]) -> int | None:
    if value is None:
        return None
    if value < 1991 or value > CURRENT_YEAR + 1:
        risk_flags.append("year_out_of_range_reset")
        return None
    return value


def _normalize_citation_count(value: int, risk_flags: list[str]) -> int:
    bounded = max(0, min(int(value), MAX_CITATION_COUNT))
    if bounded != int(value):
        risk_flags.append(f"citation_count_capped_to_{bounded}")
    return bounded


def _normalize_author(value: str | None, risk_flags: list[str]) -> str | None:
    author = _normalize_text(value, 120, risk_flags, "author")
    if author:
        author = author.removeprefix("by ").removeprefix("author ").strip()
    return author or None


def _normalize_category(value: str | None, risk_flags: list[str]) -> str | None:
    if not value:
        return None
    cleaned = _normalize_whitespace(value).strip().lower()

    alias = CATEGORY_ALIASES.get(cleaned)
    if alias:
        risk_flags.append(f"category_normalized_to_{alias}")
        return alias

    explicit = ARXIV_CATEGORY_PATTERN.search(cleaned)
    if explicit:
        token = explicit.group(0)
        parts = token.split(".", 1)
        if len(parts) == 2:
            return f"{parts[0].lower()}.{parts[1].upper()}"
        return parts[0].lower()

    for phrase, code in CATEGORY_ALIASES.items():
        if _phrase_in_text(phrase, cleaned):
            risk_flags.append(f"category_normalized_to_{code}")
            return code

    risk_flags.append("category_not_normalized")
    return value


def _extract_technical_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TECHNICAL_TOKEN_PATTERN.findall(str(text or ""))
        if len(token) >= 2
    }


def _topic_specificity_score(topic: str | None, raw_query: str) -> int:
    cleaned = _normalize_whitespace(topic or "")
    if not cleaned:
        return 0
    word_count = len(cleaned.split())
    score = min(word_count, 8)
    technical_overlap = _extract_technical_tokens(cleaned) & _extract_technical_tokens(raw_query)
    score += len(technical_overlap) * 2
    if any(marker in cleaned.lower() for marker in (" with ", " using ", " via ", " based on ", " under ")):
        score += 1
    return score


def _choose_topic(
    planner_topic: str | None,
    heuristic_topic: str | None,
    raw_query: str,
    risk_flags: list[str],
) -> str | None:
    normalized_planner = _clean_optional_text(planner_topic)
    normalized_heuristic = _clean_optional_text(heuristic_topic)

    if not normalized_planner:
        return normalized_heuristic
    if not normalized_heuristic:
        return normalized_planner

    lowered_planner = normalized_planner.lower()
    lowered_heuristic = normalized_heuristic.lower()
    if lowered_planner == lowered_heuristic:
        return normalized_planner
    if lowered_heuristic in lowered_planner:
        return normalized_planner
    if lowered_planner in lowered_heuristic:
        risk_flags.append("topic_preserved_from_query")
        return normalized_heuristic

    planner_tokens = _extract_technical_tokens(normalized_planner)
    heuristic_tokens = _extract_technical_tokens(normalized_heuristic)
    raw_tokens = _extract_technical_tokens(raw_query)
    planner_score = _topic_specificity_score(normalized_planner, raw_query)
    heuristic_score = _topic_specificity_score(normalized_heuristic, raw_query)

    if heuristic_score > planner_score + 1:
        risk_flags.append("topic_preserved_from_query")
        return normalized_heuristic
    if heuristic_tokens & raw_tokens and not (planner_tokens & raw_tokens):
        risk_flags.append("topic_preserved_from_query")
        return normalized_heuristic
    if len(normalized_heuristic) > len(normalized_planner) + 24 and heuristic_score >= planner_score:
        risk_flags.append("topic_preserved_from_query")
        return normalized_heuristic
    return normalized_planner


def _strip_search_request_wrapper(text: str) -> str:
    cleaned = _normalize_whitespace(text)
    patterns = (
        r"^(?:what are some|which|what|where|how)\s+(?:research|papers?|studies|resources?|tools?|work)?\s*(?:that|which|on|for|about)?\s*",
        r"^(?:are there (?:any|some)\s+)?(?:research\s+)?(?:papers?|studies|resources?|tools?|articles?|results?|work)\s+(?:that|which|on|for|about)?\s*",
        r"^(?:which|what|where|how)\s+(?:research|papers?|studies|resources?|tools?|work)\s+(?:that|which|on|for|about|used|use|uses|utilized|improves?|improved|proved)?\s*",
        r"^(?:could you|can you|would you|please)\s+(?:recommend|find|search|look up|direct me to)?\s*(?:research|papers?|studies|resources?|tools?|work)?\s*(?:that|which|on|for|about)?\s*",
        r"^(?:could you|can you|would you|please)\s+(?:refer me to|suggest)\s+(?:research|papers?|studies|resources?|tools?|work)?\s*(?:that|which|on|for|about)?\s*",
        r"^(?:where can i find|can you direct me to|could you direct me to)\s+",
        r"^(?:is there (?:a|any)\s+(?:paper|study|research|work)\s+)(?:that|which|exploring|about)?\s*",
        r"^(?:is there any|are there any)\s+",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:paper|papers|study|studies|research|resources?|tools?)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,.;:?!")


def _clean_search_topic(text: str | None) -> str | None:
    cleaned = _strip_search_request_wrapper(text or "")
    cleaned = re.sub(r"^(?:refer me to|suggest|suggest that|is a|is an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfor the purposes of\b", " for ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:the context of|context of)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:?!")
    if cleaned.lower() in {"paper", "papers", "research", "studies", "study", "resources", "tools"}:
        return None
    return cleaned or None


def _phrase_in_text(phrase: str, text: str) -> bool:
    lowered_text = str(text or "").lower()
    lowered_phrase = str(phrase or "").lower()
    if not lowered_phrase:
        return False
    if lowered_phrase in {"ai", "ml", "cv"}:
        return bool(re.search(rf"\b{re.escape(lowered_phrase)}\b", lowered_text))
    if " " in lowered_phrase or "." in lowered_phrase or "-" in lowered_phrase:
        return lowered_phrase in lowered_text
    return bool(re.search(rf"\b{re.escape(lowered_phrase)}\b", lowered_text))


def _has_any_hint(text: str, hints: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(hint in text or hint in lowered for hint in hints)


def _looks_like_ai_agent_request(query: str, topic: str | None) -> bool:
    combined = f"{query or ''} {topic or ''}".lower()
    if not combined.strip():
        return False
    agent_markers = (" agent", "agents", "agentic", "智能体")
    if not any(marker in combined for marker in agent_markers):
        return False
    non_ai_markers = (
        "anticancer",
        "drug",
        "drugs",
        "chemical",
        "chemotherapy",
        "hepatitis",
        "virus",
        "bacteria",
        "药物",
        "抗癌",
        "病毒",
        "细菌",
    )
    if any(marker in combined for marker in non_ai_markers):
        return False
    return True


def _normalize_ai_agent_topic(topic: str | None, query: str) -> str:
    lowered_topic = str(topic or "").strip().lower()
    lowered_query = str(query or "").strip().lower()
    if any(marker in lowered_query for marker in ("latest", "recent", "newest", "最新", "最近")):
        return "AI agents"
    if lowered_topic in {"agent", "agents", "ai agent", "ai agents", "智能体"} or not lowered_topic:
        return "AI agents"
    if "agent" in lowered_topic and "ai" not in lowered_topic and "multi-agent" not in lowered_topic:
        return f"AI {topic}"
    return topic or "AI agents"


def _is_local_db_query(text: str) -> bool:
    return _has_any_hint(text, LOCAL_DB_HINTS)


def _looks_like_search_request(text: str) -> bool:
    lowered = text.lower()
    return (
        _has_any_hint(text, SEARCH_HINTS)
        or "论文" in text
        or "文献" in text
        or "papers" in lowered
        or "studies" in lowered
        or "research" in lowered
        or "resources" in lowered
        or "arxiv" in lowered
        or "recommend research" in lowered
        or "where can i find" in lowered
        or bool(re.search(r"(?:找|搜|搜索|查|查找|检索).{0,12}(?:论文|文献|文章|资料|领域|方向|篇)?", text))
    )


def _mentions_recentness(text: str) -> bool:
    return _has_any_hint(text, RECENT_HINTS)


def _extract_requested_count(text: str) -> int | None:
    patterns = (
        r"(\d+)\s*(?:papers?|articles?|results?)",
        r"(\d+)\s*篇",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return int(matches[-1])
    chinese_match = re.search(r"([一二两三四五六七八九十]+)\s*篇", text)
    if chinese_match:
        parsed = _parse_chinese_count(chinese_match.group(1))
        if parsed is not None:
            return parsed
    if re.search(r"\bseveral\b|几篇", text, re.IGNORECASE):
        return DEFAULT_COUNT
    if re.search(r"若干篇|一些论文|一些文献", text):
        return DEFAULT_COUNT
    return None


def _extract_year(text: str) -> int | None:
    explicit = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if explicit:
        return int(explicit[-1])
    if "去年" in text or "last year" in text.lower():
        return CURRENT_YEAR - 1
    if "今年" in text or "this year" in text.lower():
        return CURRENT_YEAR
    return None


def _extract_author(text: str) -> str | None:
    patterns = (
        r"(?:author|authors?)\s*[:：]\s*([A-Za-z][A-Za-z .'\-]{2,80}?)(?=\s+(?:in\b|with\b|category\b|cat\b)|[,.]|$)",
        r"\bby\s+([A-Za-z][A-Za-z .'\-]{2,80}?)(?=\s+(?:in\b|with\b|category\b|cat\b)|[,.]|$)",
        r"作者\s*[:：]\s*([^\s，。,.;；]{2,40})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_category(text: str) -> str | None:
    explicit = ARXIV_CATEGORY_PATTERN.search(text)
    if explicit:
        return explicit.group(0)

    lowered = text.lower()
    for phrase, code in CATEGORY_ALIASES.items():
        if _phrase_in_text(phrase, lowered):
            return code
    return None


def _extract_citation_count(text: str) -> int | None:
    patterns = (
        r"(?:at least|over|more than|>=)\s*(\d+)\s*(?:citations?|citation)",
        r"(?:citations?|citation)\s*(?:over|above|>=)\s*(\d+)",
        r"(?:至少|不少于|高于|超过)\s*(\d+)\s*(?:次引用|引用)",
        r"(\d+)\s*(?:次引用|引用)以上",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_paper_title(text: str) -> str | None:
    quoted = re.search(r"[\"'“”‘’《](.+?)[\"'“”‘’》]", text)
    if quoted:
        return quoted.group(1).strip()

    summary_style = re.search(
        r"^(?:请)?(?:帮我|给我)?(?:总结|概述|解读|分析)(?:一下)?(?:这篇)?(?:论文|paper)?\s*[:：]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if summary_style:
        candidate = summary_style.group(1).strip()
        if candidate and candidate not in {"一下", "这篇", "论文"}:
            return candidate.strip(" \"'“”‘’《》")

    english_summary_style = re.search(
        r"^(?:please\s+)?(?:summarize|summary\s+of|analyze|explain)\s+(?:the\s+paper\s+)?[:：]?\s*(.+)$",
        text,
        re.IGNORECASE,
    )
    if english_summary_style:
        candidate = english_summary_style.group(1).strip()
        if candidate:
            return candidate.strip(" \"'“”‘’《》")

    exact_title = re.search(
        r"(?:title|paper title|标题)\s*(?:is|=|为|:|：)\s*([A-Za-z0-9][A-Za-z0-9 :,'\-]{3,240})",
        text,
        re.IGNORECASE,
    )
    if exact_title:
        return exact_title.group(1).strip()

    return None


def _extract_topic(text: str) -> str | None:
    if re.search(r"(?:author|authors?)\s*[:：]", text, re.IGNORECASE) and not re.search(r"\bon\s+", text, re.IGNORECASE) and "关于" not in text:
        return None

    structured = re.search(r"topic\s*[:：]\s*\[(.+?)\]", text, re.IGNORECASE)
    if structured:
        return structured.group(1).strip()

    chinese_domain = re.search(
        r"(?:在)?(?:找|搜|搜索|查|查找|检索|想找|帮我找|给我找)?\s*(?:[一二两三四五六七八九十\d]+\s*篇?)?\s*([A-Za-z0-9\u4e00-\u9fff\-/ ]{1,60}?)(?:领域|方向)",
        text,
    )
    if chinese_domain:
        return _strip_topic_tail(chinese_domain.group(1))

    chinese_search = re.search(
        r"(?:在)?(?:找|搜|搜索|查|查找|检索|想找|帮我找|给我找)\s*(?:[一二两三四五六七八九十\d]+\s*篇?)?\s*([A-Za-z0-9\u4e00-\u9fff\-/ ]{1,80}?)(?:的)?(?:论文|文献|文章|工作|研究)?(?:并|并且|然后|，|。|$)",
        text,
    )
    if chinese_search:
        return _strip_topic_tail(chinese_search.group(1))

    english = re.search(r"(?:papers?|articles?|results?)\s+on\s+(.+?)(?:[.?!]|$)", text, re.IGNORECASE)
    if english:
        return _strip_topic_tail(english.group(1))

    chinese = re.search(r"关于\s*(.+?)(?:的论文|的文献|论文|文献|吧|。|？|\?|$)", text)
    if chinese:
        return _strip_topic_tail(chinese.group(1))

    if _looks_like_search_request(text):
        cleaned = re.sub(r"\b(19\d{2}|20\d{2})\b", "", _strip_search_request_wrapper(text))
        cleaned = re.sub(r"(?:找|搜|搜索|检索|find|search|papers?|articles?|results?|studies|research|resources?|tools?|给我|帮我)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bby\s+[A-Za-z][A-Za-z .'\-]{2,80}", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:author|authors?)\s*[:：]\s*[A-Za-z][A-Za-z .'\-]{2,80}", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:category|cat)\s*[:：]?\s*[A-Za-z.\-]+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:with at least|at least|more than|over)\s*\d+\s*(?:citations?|citation)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:could|would|should|please|recommend|show|give|there|any|some|used|use|uses|utilized|improves?|improved|proved|first|find|look up)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(?:that|which)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:，。！？")
        if cleaned.lower() in {"please", "latest", "recent"}:
            return None
        cleaned = _strip_topic_tail(cleaned)
        return cleaned or None

    return None


def _strip_topic_tail(value: str) -> str:
    cleaned = re.sub(r"(?:with at least .+ citations?)$", "", value, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip()
    cleaned = re.sub(r"\bby\s+[A-Za-z][A-Za-z .'\-]{2,80}(?:\s+in\s+(?:19\d{2}|20\d{2}))?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip()
    cleaned = re.sub(r"\bin\s+(19\d{2}|20\d{2})$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(19\d{2}|20\d{2})$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcategory\s*[:：]?\s*[A-Za-z.\-]+$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:在)?(?:找|搜|搜索|查|查找|检索|想找|帮我找|给我找)\s*", "", cleaned)
    cleaned = re.sub(r"^(?:[一二两三四五六七八九十\d]+\s*篇?)\s*", "", cleaned)
    cleaned = re.sub(r"^(?:几篇|若干篇|一些)\s*", "", cleaned)
    cleaned = re.sub(r"(?:并|并且|并进行|然后).*$", "", cleaned)
    cleaned = re.sub(r"(?:优劣势对比|优缺点对比|对比分析|对比|比较).*$", "", cleaned)
    # ---- Chinese noise stripping: superlatives + demonstratives + counts ----
    # Strip trailing superlative adjectives: 最火的, 最新的, 最热的, 最红的, etc.
    cleaned = re.sub(r"(?:最[火新热红牛好强棒]的?|最新|最近|最火|最热|最好)\s*$", "", cleaned)
    # Strip trailing demonstrative + count: 那 3 篇, 这 2 个, 那些, etc.
    cleaned = re.sub(r"(?:那|这)\s*(?:[一二两三四五六七八九十\d]+\s*)?(?:篇|个|本|条|项)?\s*$", "", cleaned)
    # Strip bare trailing count + counter: 3 篇, 5 papers, etc.
    cleaned = re.sub(r"(?:\d+\s*(?:篇|个|本|条|项|papers?|articles?|results?))\s*$", "", cleaned)
    # Strip trailing isolated 的 (superlative remnant)
    cleaned = re.sub(r"(?:最[火新热红牛好强棒]的?|最新|的)\s*$", "", cleaned)
    # ---- English noise stripping ----
    # "the hottest", "the latest", "the best" etc.
    cleaned = re.sub(r"\s*(?:the\s+)?(?:hottest|latest|newest|best|top)\s*$", "", cleaned, flags=re.IGNORECASE)
    # ---- End noise stripping ----
    cleaned = re.sub(r"(?:论文|文献|文章|工作|研究)$", "", cleaned)
    cleaned = re.sub(r"(?:领域|方向)$", "", cleaned)
    cleaned = cleaned.strip(" ,.;:，。！？")
    return cleaned


def _fallback_topic_from_filters(author: str | None, category: str | None, year: int | None) -> str | None:
    parts = [part for part in (author, category, str(year) if year else None) if part]
    if not parts:
        return None
    return " ".join(parts)
