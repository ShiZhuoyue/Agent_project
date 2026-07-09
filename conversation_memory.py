import json
import math
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_openai import ChatOpenAI
from sqlalchemy import bindparam, text

from database import get_vector_db
from storage import ensure_structured_storage_ready, get_structured_store_engine, now_utc_timestamp

MAX_MEMORY_CANDIDATES = 24
HIGH_IMPORTANCE_THRESHOLD = 0.80
MID_IMPORTANCE_THRESHOLD = 0.55

from dotenv import load_dotenv
load_dotenv()

# print(os.getenv("OPENAI_MODEL"))
def _connect():
    ensure_structured_storage_ready()
    return get_structured_store_engine().connect()


def _memory_scope_sql(user_id: str | None) -> tuple[str, dict[str, Any]]:
    if str(user_id or "").strip():
        return "AND (user_id = :user_id OR user_id IS NULL)", {"user_id": user_id}
    return "AND user_id IS NULL", {}


def _make_memory_llm() -> ChatOpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    if not api_key or not base_url:
        return None
    # model = os.getenv("OPENAI_MEMORY_MODEL") or os.getenv("OPENAI_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
    model = "qwen-turbo"
    return ChatOpenAI(model=model, temperature=0, openai_api_key=api_key, base_url=base_url, request_timeout=45)


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text).strip().replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


def _clip_text(value: Any, limit: int = 240) -> str:
    cleaned = " ".join(str(value or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + " ..."


def _clean_keyword_list(values: list[Any], limit: int = 8) -> list[str]:
    cleaned = []
    for value in values:
        token = " ".join(str(value or "").split()).strip()
        if token and token not in cleaned:
            cleaned.append(token)
    return cleaned[:limit]


def _normalize_importance(value: Any, default: float = 0.5) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(score, 1.0))


def _looks_like_constraint(text: str) -> bool:
    lowered = str(text or "").lower()
    hints = (
        "must",
        "only",
        "at least",
        "do not",
        "don't",
        "avoid",
        "必须",
        "只能",
        "至少",
        "不要",
        "限定",
        "约束",
        "记住",
    )
    return any(hint in lowered or hint in text for hint in hints)


def _looks_like_preference(text: str) -> bool:
    lowered = str(text or "").lower()
    hints = (
        "prefer",
        "preference",
        "like to",
        "i want",
        "希望",
        "偏好",
        "倾向",
        "我想要",
        "习惯",
    )
    return any(hint in lowered or hint in text for hint in hints)


def _looks_like_open_loop(text: str) -> bool:
    lowered = str(text or "").lower()
    hints = (
        "continue",
        "next step",
        "follow up",
        "todo",
        "later",
        "continue later",
        "继续",
        "下一步",
        "后面",
        "待办",
        "后续",
        "还要",
    )
    return any(hint in lowered or hint in text for hint in hints)


def _infer_memory_kind(user_input: str, normalized_request: dict[str, Any]) -> str:
    if _looks_like_constraint(user_input):
        return "constraint"
    if _looks_like_preference(user_input):
        return "preference"
    if _looks_like_open_loop(user_input):
        return "task_state"
    if normalized_request.get("paper_title") or normalized_request.get("author"):
        return "reference"
    if normalized_request.get("intent") == "clarify":
        return "ephemeral"
    return "episodic"


def _build_fallback_memory_card(
    user_input: str,
    assistant_answer: str,
    normalized_request: dict[str, Any],
    tool_outputs: list[str],
) -> dict[str, Any]:
    intent = str(normalized_request.get("intent") or "unknown")
    topic = normalized_request.get("topic") or normalized_request.get("paper_title") or user_input
    constraints: list[str] = []
    if normalized_request.get("category"):
        constraints.append(f"category={normalized_request['category']}")
    if normalized_request.get("year"):
        constraints.append(f"year={normalized_request['year']}")
    if normalized_request.get("citation_count"):
        constraints.append(f"min_citations={normalized_request['citation_count']}")

    entities = _clean_keyword_list(
        [
            normalized_request.get("topic"),
            normalized_request.get("paper_title"),
            normalized_request.get("author"),
            normalized_request.get("category"),
        ],
        limit=6,
    )
    tool_takeaways = [_clip_text(item, limit=240) for item in tool_outputs[:2] if str(item).strip()]
    open_loops = [_clip_text(user_input, limit=180)] if _looks_like_open_loop(user_input) else []
    memory_kind = _infer_memory_kind(user_input, normalized_request)

    importance_score = 0.34
    if intent != "clarify":
        importance_score += 0.08
    if entities:
        importance_score += 0.10
    if constraints:
        importance_score += 0.14
    if tool_takeaways:
        importance_score += 0.08
    if open_loops:
        importance_score += 0.16
    if memory_kind in {"constraint", "preference"}:
        importance_score += 0.16
    if normalized_request.get("paper_title"):
        importance_score += 0.10

    importance_score = _normalize_importance(importance_score, default=0.5)
    summary = (
        f"User discussed {topic}. intent={intent}. "
        f"Key constraints: {', '.join(constraints) or 'none'}. "
        f"Assistant outcome: {_clip_text(assistant_answer, limit=220)}"
    ).strip()
    keywords = _clean_keyword_list([intent, *entities, *constraints], limit=8)
    compact_payload = {
        "goal": _clip_text(topic, limit=180),
        "constraints": constraints[:4],
        "decisions": [_clip_text(assistant_answer, limit=180)] if assistant_answer else [],
        "entities": entities,
        "open_loops": open_loops[:3],
        "tool_takeaways": tool_takeaways[:2],
    }
    return {
        "summary": _clip_text(summary, limit=420),
        "keywords": keywords,
        "importance_score": importance_score,
        "memory_kind": memory_kind,
        "compact_payload": compact_payload,
    }


def _normalize_compact_payload(payload: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return fallback
    normalized = {
        "goal": _clip_text(payload.get("goal") or fallback.get("goal"), limit=180),
        "constraints": _clean_keyword_list(payload.get("constraints", []) if isinstance(payload.get("constraints"), list) else fallback.get("constraints", []), limit=5),
        "decisions": [_clip_text(item, limit=180) for item in (payload.get("decisions", []) if isinstance(payload.get("decisions"), list) else fallback.get("decisions", []))][:4],
        "entities": _clean_keyword_list(payload.get("entities", []) if isinstance(payload.get("entities"), list) else fallback.get("entities", []), limit=6),
        "open_loops": [_clip_text(item, limit=180) for item in (payload.get("open_loops", []) if isinstance(payload.get("open_loops"), list) else fallback.get("open_loops", []))][:4],
        "tool_takeaways": [_clip_text(item, limit=220) for item in (payload.get("tool_takeaways", []) if isinstance(payload.get("tool_takeaways"), list) else fallback.get("tool_takeaways", []))][:3],
    }
    return normalized


def _generate_memory_card(
    user_input: str,
    assistant_answer: str,
    normalized_request: dict[str, Any],
    tool_outputs: list[str],
) -> dict[str, Any]:
    fallback = _build_fallback_memory_card(user_input, assistant_answer, normalized_request, tool_outputs)
    llm = _make_memory_llm()
    if llm is None:
        return fallback

    prompt = (
        "You are writing long-term conversational memory for a research agent.\n"
        "Return JSON only with keys: summary, keywords, importance_score, memory_kind, compact_payload.\n"
        "summary must be one compact English paragraph under 120 words.\n"
        "importance_score must be a float between 0 and 1.\n"
        "memory_kind must be one of: constraint, preference, task_state, decision, reference, episodic, ephemeral.\n"
        "compact_payload must be an object with keys: goal, constraints, decisions, entities, open_loops, tool_takeaways.\n"
        "Preserve only durable facts: user goal, confirmed constraints, chosen direction, key entities, and unresolved needs.\n"
        "Do not include filler, repeated tool logs, or transient chatter.\n\n"
        f"user_input: {user_input}\n"
        f"normalized_request: {json.dumps(normalized_request, ensure_ascii=False)}\n"
        f"tool_outputs: {json.dumps(tool_outputs[:3], ensure_ascii=False)}\n"
        f"assistant_answer: {assistant_answer}\n"
    )

    try:
        response = llm.invoke(prompt)
        payload = _parse_json_object(getattr(response, "content", response))
    except Exception:
        payload = {}

    summary = _clip_text(payload.get("summary") or fallback["summary"], limit=420)
    keywords_raw = payload.get("keywords") if isinstance(payload.get("keywords"), list) else fallback["keywords"]
    importance_score = _normalize_importance(payload.get("importance_score"), default=fallback["importance_score"])
    memory_kind = str(payload.get("memory_kind") or fallback["memory_kind"]).strip().lower() or fallback["memory_kind"]
    compact_payload = _normalize_compact_payload(payload.get("compact_payload"), fallback["compact_payload"])

    return {
        "summary": summary,
        "keywords": _clean_keyword_list(keywords_raw, limit=8),
        "importance_score": importance_score,
        "memory_kind": memory_kind,
        "compact_payload": compact_payload,
    }


def _storage_tier(importance_score: float, memory_kind: str) -> str:
    if importance_score >= HIGH_IMPORTANCE_THRESHOLD or memory_kind in {"constraint", "preference", "decision"}:
        return "critical"
    if importance_score >= MID_IMPORTANCE_THRESHOLD or memory_kind in {"task_state", "reference"}:
        return "salient"
    return "trace"


def _serialize_embedding(vector: list[float] | None) -> str:
    return json.dumps(vector or [])


def _deserialize_embedding(raw_value: str | None) -> list[float]:
    if not raw_value:
        return []
    try:
        payload = json.loads(raw_value)
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _deserialize_json(raw_value: str | None, default: Any) -> Any:
    if not raw_value:
        return default
    try:
        return json.loads(raw_value)
    except Exception:
        return default


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in str(text).replace("\n", " ").split() if token.strip()]


def _token_overlap_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    text_tokens = set(_tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _embed_text(text: str) -> list[float]:
    _, embeddings = get_vector_db()
    return embeddings.embed_query(text)


def _recency_score(created_at: str | None) -> float:
    if not created_at:
        return 0.0
    try:
        created_time = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    age_days = max((datetime.now(timezone.utc) - created_time).total_seconds() / 86400.0, 0.0)
    return max(0.0, 1.0 - min(age_days, 30.0) / 30.0)


def _reuse_score(access_count: int, recall_count: int) -> float:
    access_signal = min(max(access_count, 0), 8) / 8.0
    recall_signal = min(max(recall_count, 0), 4) / 4.0
    return recall_signal * 0.7 + access_signal * 0.3


def _composite_recall_score(
    relevance_score: float,
    importance_score: float,
    recency_score: float,
    reuse_score: float,
) -> float:
    return (
        relevance_score * 0.55
        + importance_score * 0.25
        + recency_score * 0.10
        + reuse_score * 0.10
    )


def store_turn_memory(
    user_id: str | None,
    thread_id: str,
    user_input: str,
    assistant_answer: str,
    normalized_request: dict[str, Any],
    tool_outputs: list[str],
    conversation_excerpt: list[dict[str, str]],
) -> dict[str, Any]:
    memory_card = _generate_memory_card(user_input, assistant_answer, normalized_request, tool_outputs)
    summary = memory_card["summary"]
    keywords = memory_card["keywords"]
    importance_score = _normalize_importance(memory_card["importance_score"], default=0.5)
    memory_kind = memory_card["memory_kind"]
    storage_tier = _storage_tier(importance_score, memory_kind)

    try:
        embedding = _embed_text(f"{summary}\nkeywords: {', '.join(keywords)}")
    except Exception:
        embedding = []

    full_payload = {
        "user_input": user_input,
        "assistant_answer": assistant_answer,
        "normalized_request": normalized_request,
        "tool_outputs": tool_outputs,
        "conversation_excerpt": conversation_excerpt,
    }
    compact_payload = {
        "summary": summary,
        "memory_kind": memory_kind,
        "goal": memory_card["compact_payload"].get("goal", ""),
        "constraints": memory_card["compact_payload"].get("constraints", []),
        "decisions": memory_card["compact_payload"].get("decisions", []),
        "entities": memory_card["compact_payload"].get("entities", []),
        "open_loops": memory_card["compact_payload"].get("open_loops", []),
        "tool_takeaways": memory_card["compact_payload"].get("tool_takeaways", []),
        "normalized_request": {
            key: normalized_request.get(key)
            for key in ("intent", "topic", "paper_title", "author", "year", "category", "citation_count")
            if normalized_request.get(key) is not None
        },
    }
    memory_id = f"mem_{uuid4().hex[:10]}"
    created_at = now_utc_timestamp()

    connection = _connect()
    try:
        connection.execute(
            text(
                """
                INSERT INTO turn_memories(
                    memory_id, user_id, thread_id, summary, keywords, full_payload, compact_payload, embedding,
                    importance_score, storage_tier, memory_kind, access_count, recall_count,
                    last_recalled_at, created_at
                )
                VALUES (
                    :memory_id, :user_id, :thread_id, :summary, :keywords, :full_payload, :compact_payload, :embedding,
                    :importance_score, :storage_tier, :memory_kind, 0, 0, NULL, :created_at
                )
                """
            ),
            {
                "memory_id": memory_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "summary": summary,
                "keywords": _serialize_json(keywords),
                "full_payload": _serialize_json(full_payload),
                "compact_payload": _serialize_json(compact_payload),
                "embedding": _serialize_embedding(embedding),
                "importance_score": importance_score,
                "storage_tier": storage_tier,
                "memory_kind": memory_kind,
                "created_at": created_at,
            },
        )
        connection.commit()
    finally:
        connection.close()

    return {
        "memory_id": memory_id,
        "summary": summary,
        "keywords": keywords,
        "importance_score": round(importance_score, 4),
        "storage_tier": storage_tier,
        "memory_kind": memory_kind,
    }


def search_turn_memories(
    user_id: str | None,
    thread_id: str,
    query: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    if not str(query or "").strip():
        return []

    scope_sql, scope_params = _memory_scope_sql(user_id)
    connection = _connect()
    try:
        rows = connection.execute(
            text(
                f"""
                SELECT
                    memory_id,
                    summary,
                    keywords,
                    compact_payload,
                    embedding,
                    importance_score,
                    storage_tier,
                    memory_kind,
                    access_count,
                    recall_count,
                    created_at
                FROM turn_memories
                WHERE thread_id = :thread_id
                  {scope_sql}
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {**scope_params, "thread_id": thread_id, "limit": MAX_MEMORY_CANDIDATES},
        ).fetchall()

        if not rows:
            return []

        try:
            query_vector = _embed_text(query)
        except Exception:
            query_vector = []

        scored: list[dict[str, Any]] = []
        for row in rows:
            (
                memory_id,
                summary,
                keywords_raw,
                compact_payload_raw,
                embedding_raw,
                importance_score_raw,
                storage_tier,
                memory_kind,
                access_count,
                recall_count,
                created_at,
            ) = row
            keywords = _deserialize_json(keywords_raw, [])
            compact_payload = _deserialize_json(compact_payload_raw, {})
            summary_text = str(summary or "")
            lexical_text = " ".join(
                [
                    summary_text,
                    " ".join(str(item) for item in keywords),
                    " ".join(str(item) for item in compact_payload.get("entities", [])),
                    " ".join(str(item) for item in compact_payload.get("constraints", [])),
                    " ".join(str(item) for item in compact_payload.get("open_loops", [])),
                ]
            )

            semantic_score = _cosine_similarity(query_vector, _deserialize_embedding(embedding_raw)) if query_vector else 0.0
            lexical_score = _token_overlap_score(query, lexical_text)
            relevance_score = semantic_score * 0.70 + lexical_score * 0.30
            importance_score = _normalize_importance(importance_score_raw, default=0.5)
            recency_score = _recency_score(created_at)
            reuse = _reuse_score(int(access_count or 0), int(recall_count or 0))
            overall_score = _composite_recall_score(relevance_score, importance_score, recency_score, reuse)

            if overall_score <= 0:
                continue

            scored.append(
                {
                    "memory_id": memory_id,
                    "summary": summary_text,
                    "keywords": keywords,
                    "created_at": created_at,
                    "score": round(overall_score, 4),
                    "relevance_score": round(relevance_score, 4),
                    "importance_score": round(importance_score, 4),
                    "recency_score": round(recency_score, 4),
                    "reuse_score": round(reuse, 4),
                    "storage_tier": storage_tier or "salient",
                    "memory_kind": memory_kind or "episodic",
                    "access_count": int(access_count or 0),
                    "recall_count": int(recall_count or 0),
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        selected = scored[:limit]
        if selected:
            connection.execute(
                text(
                    "UPDATE turn_memories SET access_count = access_count + 1 WHERE memory_id = :memory_id"
                ),
                [{"memory_id": item["memory_id"]} for item in selected],
            )
            connection.commit()
        return selected
    finally:
        connection.close()


def _select_payload_for_tier(
    summary: str,
    keywords: list[str],
    full_payload: dict[str, Any],
    compact_payload: dict[str, Any],
    storage_tier: str,
    detail_mode: str,
) -> dict[str, Any]:
    if detail_mode == "full":
        return full_payload
    if detail_mode == "compact":
        return compact_payload or full_payload
    if detail_mode == "summary":
        return {
            "summary": summary,
            "keywords": keywords,
            "entities": compact_payload.get("entities", []),
            "constraints": compact_payload.get("constraints", []),
            "open_loops": compact_payload.get("open_loops", []),
        }

    if storage_tier == "critical":
        return full_payload
    if storage_tier == "salient":
        return compact_payload or full_payload
    return {
        "summary": summary,
        "keywords": keywords,
        "goal": compact_payload.get("goal", ""),
        "constraints": compact_payload.get("constraints", []),
        "entities": compact_payload.get("entities", []),
        "open_loops": compact_payload.get("open_loops", []),
    }


def load_turn_memories(
    user_id: str | None,
    thread_id: str,
    memory_ids: list[str],
    detail_mode: str = "auto",
) -> list[dict[str, Any]]:
    if not memory_ids:
        return []

    scope_sql, scope_params = _memory_scope_sql(user_id)
    connection = _connect()
    try:
        rows = connection.execute(
            text(
                f"""
                SELECT
                    memory_id,
                    summary,
                    keywords,
                    full_payload,
                    compact_payload,
                    importance_score,
                    storage_tier,
                    memory_kind,
                    access_count,
                    recall_count,
                    created_at
                FROM turn_memories
                WHERE thread_id = :thread_id
                  {scope_sql}
                  AND memory_id IN :memory_ids
                """
            ).bindparams(bindparam("memory_ids", expanding=True)),
            {**scope_params, "thread_id": thread_id, "memory_ids": memory_ids},
        ).fetchall()

        if rows:
            timestamp = now_utc_timestamp()
            connection.execute(
                text(
                    """
                    UPDATE turn_memories
                    SET recall_count = recall_count + 1, last_recalled_at = :timestamp
                    WHERE memory_id = :memory_id
                    """
                ),
                [
                    {"timestamp": timestamp, "memory_id": memory_id}
                    for memory_id in memory_ids
                ],
            )
            connection.commit()
    finally:
        connection.close()

    records_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        (
            memory_id,
            summary,
            keywords_raw,
            full_payload_raw,
            compact_payload_raw,
            importance_score_raw,
            storage_tier,
            memory_kind,
            access_count,
            recall_count,
            created_at,
        ) = row

        full_payload = _deserialize_json(full_payload_raw, {})
        compact_payload = _deserialize_json(compact_payload_raw, {})
        keywords = _deserialize_json(keywords_raw, [])
        chosen_payload = _select_payload_for_tier(
            summary=str(summary or ""),
            keywords=keywords,
            full_payload=full_payload,
            compact_payload=compact_payload,
            storage_tier=storage_tier or "salient",
            detail_mode=detail_mode,
        )

        records_by_id[memory_id] = {
            "memory_id": memory_id,
            "summary": str(summary or ""),
            "keywords": keywords,
            "payload": chosen_payload,
            "compact_payload": compact_payload,
            "importance_score": round(_normalize_importance(importance_score_raw, default=0.5), 4),
            "storage_tier": storage_tier or "salient",
            "memory_kind": memory_kind or "episodic",
            "access_count": int(access_count or 0),
            "recall_count": int(recall_count or 0) + 1,
            "created_at": created_at,
            "detail_mode": detail_mode if detail_mode != "auto" else (storage_tier or "salient"),
            "has_archival_payload": bool(full_payload),
        }

    return [records_by_id[memory_id] for memory_id in memory_ids if memory_id in records_by_id]


def load_recent_turn_memories(
    user_id: str | None,
    thread_id: str,
    limit: int = 2,
    detail_mode: str = "auto",
) -> list[dict[str, Any]]:
    if not str(thread_id or "").strip():
        return []

    safe_limit = max(1, int(limit))
    scope_sql, scope_params = _memory_scope_sql(user_id)
    connection = _connect()
    try:
        rows = connection.execute(
            text(
                f"""
                SELECT memory_id
                FROM turn_memories
                WHERE thread_id = :thread_id
                  {scope_sql}
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {**scope_params, "thread_id": thread_id, "limit": safe_limit},
        ).fetchall()
    finally:
        connection.close()

    memory_ids = [str(row[0]) for row in rows if row and row[0]]
    if not memory_ids:
        return []
    return load_turn_memories(user_id, thread_id, memory_ids, detail_mode=detail_mode)
