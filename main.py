import hashlib
import hmac
import json
import os
import re
from functools import lru_cache
from time import perf_counter
from typing import Any, Iterator, Optional, List

from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from research_harness import build_structured_request
from storage import (
    count_registered_users,
    create_chat_thread,
    create_user,
    create_user_session,
    derive_thread_title,
    delete_user_session,
    get_active_session,
    get_chat_thread_for_user,
    get_user_by_username,
    list_chat_messages_for_user,
    list_chat_threads_for_user,
    save_chat_message,
    touch_user_session,
    update_chat_thread_activity,
)

load_dotenv()

app = FastAPI(title="DeepResearch API Server", version="1.2.0")

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "deepresearch_session")
PASSWORD_HASH_ITERATIONS = int(os.getenv("PASSWORD_HASH_ITERATIONS", "390000"))
CONTEXT_FOLLOW_UP_HINTS = (
    "继续",
    "接着",
    "然后",
    "再",
    "对比",
    "比较",
    "优劣势",
    "优缺点",
    "总结",
    "概述",
    "分析",
    "归纳",
    "这些",
    "这两篇",
    "这几篇",
    "它们",
    "刚才",
    "上面",
    "前面",
    "之前",
    "same",
    "them",
    "those",
    "compare",
    "comparison",
    "summarize",
    "summary",
    "continue",
    "follow up",
)


@lru_cache(maxsize=1)
def get_agent_executor():
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("OPENAI_API_BASE"):
        missing.append("OPENAI_API_BASE")
    if missing:
        raise RuntimeError(f"Runtime configuration is incomplete: {', '.join(missing)}")

    try:
        from agent import create_research_agent
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Missing runtime dependency: {exc.name}") from exc

    return create_research_agent()


class ResearchRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None


class ResearchResponse(BaseModel):
    thread_id: str
    answer: str
    metrics: dict
    status: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    email: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    user_id: str
    username: str
    display_name: Optional[str] = None
    email: Optional[str] = None
    created_at: str


class AuthResponse(BaseModel):
    user: UserResponse
    session_token: str
    expires_at: str


class CreateThreadRequest(BaseModel):
    title: Optional[str] = None


class ThreadResponse(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    last_query: Optional[str] = None
    message_count: int = 0


class ChatMessageResponse(BaseModel):
    message_id: str
    thread_id: str
    user_id: str
    role: str
    content: str
    metrics: dict
    status: Optional[str] = None
    created_at: str


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _session_ttl_hours() -> int:
    try:
        return max(int(os.getenv("SESSION_TTL_HOURS", "168")), 1)
    except ValueError:
        return 168


def _registration_open() -> bool:
    return count_registered_users() == 0 or _bool_env("ALLOW_PUBLIC_REGISTRATION", default=False)


def _normalize_username(username: str) -> str:
    return str(username or "").strip().lower()


def _validate_credentials_payload(username: str, password: str) -> tuple[str, str]:
    normalized_username = _normalize_username(username)
    if not USERNAME_PATTERN.fullmatch(normalized_username):
        raise HTTPException(
            status_code=422,
            detail="Username must be 3-32 chars and contain only letters, numbers, dot, dash, or underscore.",
        )
    if len(str(password or "")) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")
    return normalized_username, str(password)


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS)
    return f"{PASSWORD_HASH_ITERATIONS}${salt.hex()}${derived.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        raw_iterations, salt_hex, digest_hex = str(stored_hash or "").split("$", 2)
        iterations = int(raw_iterations)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except Exception:
        return False

    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def _public_user(user_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": user_row["user_id"],
        "username": user_row["username"],
        "display_name": user_row.get("display_name"),
        "email": user_row.get("email"),
        "created_at": user_row["created_at"],
    }


def _set_session_cookie(response: Response, session_token: str, expires_at: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        secure=_bool_env("SECURE_SESSION_COOKIES", default=False),
        samesite="lax",
        max_age=_session_ttl_hours() * 3600,
    )
    response.headers["X-Session-Expires-At"] = expires_at


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, samesite="lax")


def _extract_session_token(
    authorization: str | None,
    session_cookie: str | None,
) -> str | None:
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    if session_cookie:
        return session_cookie.strip()
    return None


def get_current_user(
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> dict[str, Any]:
    session_token = _extract_session_token(authorization, session_cookie)
    if not session_token:
        raise HTTPException(status_code=401, detail="Authentication required.")

    session = get_active_session(session_token)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid.")

    touch_user_session(session_token)
    return {
        **session["user"],
        "session_id": session["session_id"],
        "session_expires_at": session["expires_at"],
    }


def _looks_like_contextual_follow_up(query: str) -> bool:
    lowered = str(query or "").lower()
    if any(hint in query or hint in lowered for hint in CONTEXT_FOLLOW_UP_HINTS):
        return True
    return bool(
        re.search(
            r"(?:找|搜|搜索|查|查找|检索).{0,12}(?:并|然后|再|对比|比较|总结|分析)",
            str(query or ""),
        )
    )


def _load_previous_structured_request(user_id: str, thread_id: str, current_query: str) -> dict[str, Any] | None:
    messages = list_chat_messages_for_user(user_id, thread_id, limit=24)
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if not content or content == str(current_query or "").strip():
            continue
        candidate_request = build_structured_request(content, {})
        if candidate_request.get("intent") == "clarify":
            continue
        if any(
            [
                candidate_request.get("topic"),
                candidate_request.get("paper_title"),
                candidate_request.get("author"),
                candidate_request.get("category"),
            ]
        ):
            candidate_request["source_query"] = content
            return candidate_request
    return None


def _inherit_contextual_preflight(query: str, prior_request: dict[str, Any]) -> dict[str, Any]:
    planner_payload = {
        "intent": prior_request.get("intent"),
        "topic": prior_request.get("topic"),
        "paper_title": prior_request.get("paper_title"),
        "author": prior_request.get("author"),
        "year": prior_request.get("year"),
        "category": prior_request.get("category"),
        "citation_count": prior_request.get("citation_count"),
        "count": prior_request.get("count"),
    }
    inherited_request = build_structured_request(query, planner_payload)
    if inherited_request.get("intent") == "clarify":
        return inherited_request

    inherited_request["risk_flags"] = list(
        dict.fromkeys(
            [
                *inherited_request.get("risk_flags", []),
                "context_inherited_from_previous_turn",
            ]
        )
    )
    inherited_request["context_source_query"] = prior_request.get("source_query")

    if inherited_request.get("paper_title"):
        inherited_request["question"] = (
            f"{query} | context_paper_title: {inherited_request['paper_title']}"
        )
    elif inherited_request.get("topic"):
        inherited_request["question"] = (
            f"{query} | context_topic: {inherited_request['topic']}"
        )

    return inherited_request


def _preflight_request(
    query: str,
    user_id: str | None = None,
    thread_id: str | None = None,
) -> tuple[dict[str, Any], float]:
    started_at = perf_counter()
    heuristic_request = build_structured_request(query, {})
    if (
        heuristic_request.get("intent") == "clarify"
        and user_id
        and thread_id
        and _looks_like_contextual_follow_up(query)
    ):
        prior_request = _load_previous_structured_request(user_id, thread_id, query)
        if prior_request:
            inherited_request = _inherit_contextual_preflight(query, prior_request)
            if inherited_request.get("intent") != "clarify":
                heuristic_request = inherited_request
    latency = round(perf_counter() - started_at, 4)
    return heuristic_request, latency


def _base_agent_input(query: str, preflight_latency: float, thread_id: str, user_id: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    return {
        "input": query,
        "user_id": user_id,
        "thread_id": thread_id,
        "messages": [HumanMessage(content=query)],
        "metrics": {"00_Harness_Preflight": preflight_latency},
        # ========== Observer审计纠偏闭环：初始化默认值 ==========
        "max_replan_times": 3,       # 最大重规划次数，防止死循环
        "current_replan_round": 0,   # 当前重规划轮次计数器
        # ========== 三级文献分级精读：初始化默认值 ==========
        "rated_papers": [],          # Observer打分后的论文列表
    }


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                parts.append(str(text) if text is not None else json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    return str(content)


def _extract_final_answer(messages: list[Any]) -> str:
    from langchain_core.messages import AIMessage

    for message in reversed(messages or []):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            text = _stringify_content(getattr(message, "content", ""))
            if text:
                return text
    return "Analysis completed."


def _truncate(text: str, limit: int = 1200) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + " ..."


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"


def _resolve_thread_for_user(user_id: str, requested_thread_id: str | None, query: str) -> dict[str, Any]:
    if requested_thread_id:
        thread = get_chat_thread_for_user(user_id, requested_thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Chat thread not found.")
        updated_thread = update_chat_thread_activity(requested_thread_id, query=query)
        return updated_thread or thread

    return create_chat_thread(
        user_id=user_id,
        title=derive_thread_title(query),
        last_query=query,
    )


def _persist_user_message(user_id: str, thread_id: str, query: str) -> None:
    save_chat_message(
        user_id=user_id,
        thread_id=thread_id,
        role="user",
        content=query,
    )


def _persist_assistant_message(
    user_id: str,
    thread_id: str,
    answer: str,
    metrics: dict[str, Any],
    status: str,
) -> None:
    save_chat_message(
        user_id=user_id,
        thread_id=thread_id,
        role="assistant",
        content=answer,
        metrics=metrics,
        status=status,
    )


def _stream_updates(
    current_user_id: str,
    current_thread_id: str,
    query: str,
    preflight_latency: float,
) -> Iterator[str]:
    metrics: dict[str, Any] = {"00_Harness_Preflight": preflight_latency}
    status = "success"
    final_answer = ""
    config = {"configurable": {"thread_id": current_thread_id}}
    stream_started = False

    yield _sse_event(
        "status",
        {
            "stage": "preflight",
            "label": "Harness preflight completed",
            "state": "running",
            "detail": f"thread_id={current_thread_id}",
        },
    )

    try:
        executor = get_agent_executor()
        for chunk in executor.stream(
            _base_agent_input(query, preflight_latency, current_thread_id, current_user_id),
            config=config,
            stream_mode="updates",
        ):
            stream_started = True
            if not isinstance(chunk, dict):
                continue

            for node_name, node_payload in chunk.items():
                if not isinstance(node_payload, dict):
                    continue

                node_metrics = node_payload.get("metrics", {})
                if isinstance(node_metrics, dict):
                    metrics.update(node_metrics)

                if node_name == "memory_router":
                    candidates = node_payload.get("memory_candidates", [])
                    recalled = node_payload.get("recalled_memories", [])
                    candidate_preview = ", ".join(
                        [
                            f"{item.get('memory_id')}({item.get('score', 0.0):.2f}/{item.get('storage_tier', 'salient')})"
                            for item in candidates[:3]
                        ]
                    ) or "none"
                    recalled_preview = ", ".join(item.get("memory_id", "unknown") for item in recalled[:2]) or "none"
                    yield _sse_event(
                        "update",
                        {
                            "node": "memory_router",
                            "label": "Memory router evaluated prior turns",
                            "detail": f"candidates={candidate_preview} | recalled={recalled_preview}",
                        },
                    )
                elif node_name == "planner":
                    yield _sse_event(
                        "update",
                        {
                            "node": "planner",
                            "label": "Planner normalized the request",
                            "detail": _truncate(
                                json.dumps(
                                    node_payload.get("normalized_request", {}),
                                    ensure_ascii=False,
                                ),
                                limit=1500,
                            ),
                        },
                    )
                elif node_name == "executor":
                    messages = node_payload.get("messages", [])
                    message = messages[-1] if messages else None
                    goal = _stringify_content(getattr(message, "content", ""))
                    tool_calls = getattr(message, "tool_calls", []) or []
                    tool_names = [call.get("name", "tool") for call in tool_calls if isinstance(call, dict)]
                    detail = goal or "Executor prepared the next step."
                    if tool_names:
                        detail = f"{detail} | tool_calls={', '.join(tool_names)}"
                    yield _sse_event(
                        "update",
                        {
                            "node": "executor",
                            "label": "Executor scheduled the next action",
                            "detail": _truncate(detail),
                        },
                    )
                elif node_name == "tools":
                    messages = node_payload.get("messages", [])
                    tool_message = messages[-1] if messages else None
                    preview = _truncate(_stringify_content(getattr(tool_message, "content", "")), limit=1600)
                    yield _sse_event(
                        "update",
                        {
                            "node": "tools",
                            "label": "Tool execution returned",
                            "detail": preview or "Tool execution completed.",
                        },
                    )
                elif node_name == "observer":
                    past_steps = node_payload.get("past_steps", [])
                    rated_papers = node_payload.get("rated_papers", [])
                    if past_steps:
                        yield _sse_event(
                            "update",
                            {
                                "node": "observer",
                                "label": "Observer recorded the latest result",
                                "detail": _truncate(str(past_steps[-1]), limit=1200),
                            },
                        )
                    # ========== 三级文献分级精读：SSE推送打分结果 ==========
                    if rated_papers:
                        level_counts: dict[str, int] = {"deep": 0, "medium": 0, "coarse": 0}
                        for rp in rated_papers:
                            lv = str(rp.get("read_level", "coarse")).lower()
                            level_counts[lv] = level_counts.get(lv, 0) + 1
                        yield _sse_event(
                            "update",
                            {
                                "node": "observer",
                                "label": "Observer 论文分级打分完成",
                                "detail": (
                                    f"共 {len(rated_papers)} 篇论文: "
                                    f"深度精读={level_counts.get('deep', 0)}, "
                                    f"中度阅读={level_counts.get('medium', 0)}, "
                                    f"粗读={level_counts.get('coarse', 0)}"
                                ),
                            },
                        )
                elif node_name == "executor_read":
                    # ========== 三级文献分级精读：SSE推送精读进度 ==========
                    past_steps = node_payload.get("past_steps", [])
                    read_count: int = len(past_steps) if past_steps else 0
                    yield _sse_event(
                        "update",
                        {
                            "node": "executor_read",
                            "label": f"分层精读完成 ({read_count} 篇论文)",
                            "detail": _truncate(str(past_steps[-1]) if past_steps else "", limit=1200),
                        },
                    )
                elif node_name == "synthesizer":
                    messages = node_payload.get("messages", [])
                    final_answer = _extract_final_answer(messages)
                    yield _sse_event(
                        "answer",
                        {
                            "answer": final_answer,
                        },
                    )
                elif node_name == "memory_writer":
                    yield _sse_event(
                        "update",
                        {
                            "node": "memory_writer",
                            "label": "Conversation memory stored",
                            "detail": "This turn was compacted and written to long-term memory.",
                        },
                    )
    except Exception as stream_exc:
        if not stream_started:
            try:
                executor = get_agent_executor()
                result = executor.invoke(
                    _base_agent_input(query, preflight_latency, current_thread_id, current_user_id),
                    config=config,
                )
                metrics.update(result.get("metrics", {}))
                final_answer = _extract_final_answer(result.get("messages", []))
                yield _sse_event(
                    "update",
                    {
                        "node": "system",
                        "label": "Streaming fallback engaged",
                        "detail": _truncate(str(stream_exc), limit=300),
                    },
                )
            except Exception as fallback_exc:
                status = "error"
                yield _sse_event(
                    "error",
                    {
                        "detail": str(fallback_exc),
                    },
                )
        else:
            status = "error"
            yield _sse_event(
                "error",
                {
                    "detail": str(stream_exc),
                },
            )

    if not final_answer and status != "error":
        status = "partial"
        final_answer = "Analysis completed, but no final synthesized answer was captured from the stream."

    if status != "error":
        _persist_assistant_message(current_user_id, current_thread_id, final_answer, metrics, status)
        yield _sse_event(
            "final",
            {
                "thread_id": current_thread_id,
                "answer": final_answer,
                "metrics": metrics,
                "status": status,
            },
        )


def _invoke_once(req: ResearchRequest, current_user: dict[str, Any]) -> ResearchResponse:
    thread = _resolve_thread_for_user(current_user["user_id"], req.thread_id, req.query)
    current_thread_id = thread["thread_id"]
    heuristic_request, preflight_latency = _preflight_request(
        req.query,
        user_id=current_user["user_id"],
        thread_id=current_thread_id,
    )
    _persist_user_message(current_user["user_id"], current_thread_id, req.query)
    config = {"configurable": {"thread_id": current_thread_id}}

    if heuristic_request["intent"] == "clarify":
        metrics = {"00_Harness_Preflight": preflight_latency}
        _persist_assistant_message(
            current_user["user_id"],
            current_thread_id,
            heuristic_request["response"],
            metrics,
            "clarify",
        )
        return ResearchResponse(
            thread_id=current_thread_id,
            answer=heuristic_request["response"],
            metrics=metrics,
            status="clarify",
        )

    result = get_agent_executor().invoke(
        _base_agent_input(req.query, preflight_latency, current_thread_id, current_user["user_id"]),
        config=config,
    )
    answer = _extract_final_answer(result.get("messages", []))
    metrics = result.get("metrics", {})
    _persist_assistant_message(current_user["user_id"], current_thread_id, answer, metrics, "success")
    return ResearchResponse(
        thread_id=current_thread_id,
        answer=answer,
        metrics=metrics,
        status="success",
    )


@app.get("/")
def read_root():
    return {
        "status": "Online",
        "service": "DeepResearch AI Engine",
        "registration_open": _registration_open(),
    }


@app.post("/auth/register", response_model=AuthResponse)
def register(request: RegisterRequest, response: Response):
    if not _registration_open():
        raise HTTPException(status_code=403, detail="Public registration is disabled.")

    username, password = _validate_credentials_payload(request.username, request.password)
    try:
        user = create_user(
            username=username,
            password_hash=_hash_password(password),
            display_name=request.display_name,
            email=request.email,
        )
    except ValueError as exc:
        if str(exc) == "username_already_exists":
            raise HTTPException(status_code=409, detail="Username already exists.") from exc
        raise

    session = create_user_session(user["user_id"], ttl_hours=_session_ttl_hours())
    _set_session_cookie(response, session["session_id"], session["expires_at"])
    return {
        "user": _public_user(user),
        "session_token": session["session_id"],
        "expires_at": session["expires_at"],
    }


@app.post("/auth/login", response_model=AuthResponse)
def login(request: LoginRequest, response: Response):
    username, password = _validate_credentials_payload(request.username, request.password)
    user = get_user_by_username(username)
    if not user or not _verify_password(password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    session = create_user_session(user["user_id"], ttl_hours=_session_ttl_hours())
    _set_session_cookie(response, session["session_id"], session["expires_at"])
    return {
        "user": _public_user(user),
        "session_token": session["session_id"],
        "expires_at": session["expires_at"],
    }


@app.post("/auth/logout")
def logout(
    response: Response,
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    session_token = _extract_session_token(authorization, session_cookie)
    if session_token:
        delete_user_session(session_token)
    _clear_session_cookie(response)
    return {"status": "logged_out"}


@app.get("/auth/me", response_model=UserResponse)
def me(current_user: dict[str, Any] = Depends(get_current_user)):
    return _public_user(current_user)


@app.get("/v1/threads", response_model=list[ThreadResponse])
def list_threads(current_user: dict[str, Any] = Depends(get_current_user)):
    return list_chat_threads_for_user(current_user["user_id"])


@app.post("/v1/threads", response_model=ThreadResponse)
def create_thread(req: CreateThreadRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    thread = create_chat_thread(
        user_id=current_user["user_id"],
        title=req.title or "New chat",
    )
    return {**thread, "message_count": 0}


@app.get("/v1/threads/{thread_id}/messages", response_model=list[ChatMessageResponse])
def get_thread_messages(thread_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    thread = get_chat_thread_for_user(current_user["user_id"], thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Chat thread not found.")
    return list_chat_messages_for_user(current_user["user_id"], thread_id)


@app.post("/v1/chat", response_model=ResearchResponse)
async def chat_with_agent(req: ResearchRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    try:
        return _invoke_once(req, current_user)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/chat/stream")
async def chat_with_agent_stream(
    req: ResearchRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    thread = _resolve_thread_for_user(current_user["user_id"], req.thread_id, req.query)
    current_thread_id = thread["thread_id"]
    heuristic_request, preflight_latency = _preflight_request(
        req.query,
        user_id=current_user["user_id"],
        thread_id=current_thread_id,
    )
    _persist_user_message(current_user["user_id"], current_thread_id, req.query)

    if heuristic_request["intent"] == "clarify":
        metrics = {"00_Harness_Preflight": preflight_latency}
        _persist_assistant_message(
            current_user["user_id"],
            current_thread_id,
            heuristic_request["response"],
            metrics,
            "clarify",
        )
        stream = iter(
            [
                _sse_event(
                    "status",
                    {
                        "stage": "preflight",
                        "label": "Clarification is required",
                        "state": "complete",
                        "detail": "The harness stopped before agent execution.",
                    },
                ),
                _sse_event(
                    "final",
                    {
                        "thread_id": current_thread_id,
                        "answer": heuristic_request["response"],
                        "metrics": metrics,
                        "status": "clarify",
                    },
                ),
            ]
        )
    else:
        stream = _stream_updates(current_user["user_id"], current_thread_id, req.query, preflight_latency)

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ========== Observer审计纠偏闭环：审计日志查询接口 ==========
# GET /api/audit/logs?thread_id=<thread_id>
# 根据 thread_id 返回当前会话的全部结构化审计日志，
# 包含每轮检索关键词、论文列表、缺陷说明、修正关键词、文献溯源信息。
class AuditLogItem(BaseModel):
    round_index: int
    task_type: str
    original_query: str
    planner_keywords: List[str]
    retrieved_papers: List[dict]
    paper_summary_results: List[dict]
    defect_type: Optional[str]
    defect_desc: Optional[str]
    need_replan: bool
    suggest_new_keywords: List[str]
    # ========== 三级文献分级精读：新增字段 ==========
    rated_papers: List[dict] = []  # Observer对每篇论文的相关度打分与阅读等级


class AuditLogsResponse(BaseModel):
    thread_id: str
    total_rounds: int
    current_defect: Optional[str]
    current_replan_round: int
    max_replan_times: int
    logs: List[AuditLogItem]


@app.get("/api/audit/logs", response_model=AuditLogsResponse)
def get_audit_logs(
    thread_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> AuditLogsResponse:
    """返回指定 thread_id 的 Observer 审计纠偏闭环全部结构化日志。

    每轮日志包含：
    - 检索关键词（planner_keywords）
    - 检索到的论文列表（retrieved_papers）
    - 缺陷类型与说明（defect_type / defect_desc）
    - 建议修正关键词（suggest_new_keywords）
    - 文献溯源信息（paper_summary_results）
    """
    try:
        agent = get_agent_executor()
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        state_snapshot = agent.get_state(config)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to retrieve agent state: {exc}",
        ) from exc

    if state_snapshot is None or state_snapshot.values is None:
        return AuditLogsResponse(
            thread_id=thread_id,
            total_rounds=0,
            current_defect=None,
            current_replan_round=0,
            max_replan_times=3,
            logs=[],
        )

    state_values: dict[str, Any] = dict(state_snapshot.values)
    raw_logs: list[dict[str, Any]] = state_values.get("audit_logs", []) or []
    current_defect: Optional[str] = state_values.get("current_defect")
    current_replan_round: int = state_values.get("current_replan_round", 0)
    max_replan_times: int = state_values.get("max_replan_times", 3)

    structured_logs: List[AuditLogItem] = []
    for idx, log_entry in enumerate(raw_logs):
        structured_logs.append(
            AuditLogItem(
                round_index=idx,
                task_type=str(log_entry.get("task_type", "")),
                original_query=str(log_entry.get("original_query", "")),
                planner_keywords=list(log_entry.get("planner_keywords", [])),
                retrieved_papers=list(log_entry.get("retrieved_papers", [])),
                paper_summary_results=list(log_entry.get("paper_summary_results", [])),
                defect_type=log_entry.get("defect_type"),
                defect_desc=log_entry.get("defect_desc"),
                need_replan=bool(log_entry.get("need_replan", False)),
                suggest_new_keywords=list(log_entry.get("suggest_new_keywords", [])),
                # ========== 三级文献分级精读：传递打分结果 ==========
                rated_papers=list(log_entry.get("rated_papers", [])),
            )
        )

    return AuditLogsResponse(
        thread_id=thread_id,
        total_rounds=len(structured_logs),
        current_defect=current_defect,
        current_replan_round=current_replan_round,
        max_replan_times=max_replan_times,
        logs=structured_logs,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
