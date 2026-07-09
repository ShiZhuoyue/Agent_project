import atexit
import contextlib
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

STORAGE_BACKEND_AUTO = "auto"
STORAGE_BACKEND_SQLITE = "sqlite"
STORAGE_BACKEND_POSTGRES = "postgresql"

DEFAULT_SQLITE_STORAGE_PATH = "app_storage.db"
DEFAULT_SQLITE_CHECKPOINT_PATH = "agent_checkpoint.db"
DEFAULT_LEGACY_CONVERSATION_MEMORY_PATH = "conversation_memory.db"
DEFAULT_LEGACY_HYBRID_INDEX_PATH = "./vector_db_storage/hybrid_chunks.jsonl"


def now_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def utc_timestamp_after_hours(hours: int) -> str:
    safe_hours = max(int(hours), 1)
    return (datetime.now(timezone.utc) + timedelta(hours=safe_hours)).strftime("%Y-%m-%d %H:%M:%S")


def _parse_utc_timestamp(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    try:
        return datetime.strptime(raw_value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _normalized_database_url() -> str:
    return str(os.getenv("DATABASE_URL", "")).strip().strip("\"'")


def _normalize_backend_name(raw_value: str | None) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized in {"", STORAGE_BACKEND_AUTO}:
        return STORAGE_BACKEND_AUTO
    if normalized in {"sqlite", "sqlite3"}:
        return STORAGE_BACKEND_SQLITE
    if normalized in {"postgres", "postgresql"}:
        return STORAGE_BACKEND_POSTGRES
    raise RuntimeError(
        "Unsupported STORAGE_BACKEND. Use one of: auto, sqlite, postgresql."
    )


def get_storage_backend() -> str:
    configured_backend = _normalize_backend_name(os.getenv("STORAGE_BACKEND"))
    if configured_backend == STORAGE_BACKEND_AUTO:
        return (
            STORAGE_BACKEND_POSTGRES
            if _normalized_database_url().startswith(("postgresql://", "postgres://"))
            else STORAGE_BACKEND_SQLITE
        )
    return configured_backend


def get_sqlite_storage_path() -> str:
    configured = str(os.getenv("SQLITE_STORAGE_DB_PATH", DEFAULT_SQLITE_STORAGE_PATH)).strip()
    return configured or DEFAULT_SQLITE_STORAGE_PATH


def get_sqlite_checkpoint_path() -> str:
    configured = str(
        os.getenv("AGENT_CHECKPOINT_DB_PATH", DEFAULT_SQLITE_CHECKPOINT_PATH)
    ).strip()
    return configured or DEFAULT_SQLITE_CHECKPOINT_PATH


def get_legacy_conversation_memory_path() -> str:
    return str(
        os.getenv(
            "LEGACY_CONVERSATION_MEMORY_PATH",
            DEFAULT_LEGACY_CONVERSATION_MEMORY_PATH,
        )
    ).strip()


def get_legacy_hybrid_index_path() -> str:
    return str(
        os.getenv("LEGACY_HYBRID_INDEX_PATH", DEFAULT_LEGACY_HYBRID_INDEX_PATH)
    ).strip()


def _ensure_parent_dir(path_value: str) -> None:
    parent = Path(path_value).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def _sqlalchemy_postgres_url() -> str:
    database_url = _normalized_database_url()
    if not database_url:
        raise RuntimeError(
            "STORAGE_BACKEND resolved to postgresql, but DATABASE_URL is missing."
        )
    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


def get_structured_store_url() -> str:
    backend = get_storage_backend()
    if backend == STORAGE_BACKEND_POSTGRES:
        return _sqlalchemy_postgres_url()

    sqlite_path = get_sqlite_storage_path()
    _ensure_parent_dir(sqlite_path)
    return f"sqlite:///{Path(sqlite_path).resolve().as_posix()}"


@lru_cache(maxsize=1)
def get_structured_store_engine():
    backend = get_storage_backend()
    url = get_structured_store_url()
    if backend == STORAGE_BACKEND_SQLITE:
        return create_engine(
            url,
            future=True,
            connect_args={"check_same_thread": False},
        )
    return create_engine(url, future=True, pool_pre_ping=True)


def reset_storage_runtime() -> None:
    if get_structured_store_engine.cache_info().currsize:
        get_structured_store_engine().dispose()
        get_structured_store_engine.cache_clear()
    ensure_structured_storage_ready.cache_clear()


def _dispose_structured_store_engine() -> None:
    if get_structured_store_engine.cache_info().currsize:
        get_structured_store_engine().dispose()


atexit.register(_dispose_structured_store_engine)


def _turn_memories_expected_columns() -> dict[str, str]:
    return {
        "memory_id": "TEXT PRIMARY KEY",
        "user_id": "TEXT",
        "thread_id": "TEXT NOT NULL",
        "summary": "TEXT NOT NULL",
        "keywords": "TEXT NOT NULL",
        "full_payload": "TEXT NOT NULL",
        "compact_payload": "TEXT",
        "embedding": "TEXT",
        "importance_score": "REAL DEFAULT 0.5",
        "storage_tier": "TEXT DEFAULT 'salient'",
        "memory_kind": "TEXT DEFAULT 'episodic'",
        "access_count": "INTEGER DEFAULT 0",
        "recall_count": "INTEGER DEFAULT 0",
        "last_recalled_at": "TEXT",
        "created_at": "TEXT NOT NULL",
    }


def _hybrid_chunks_expected_columns() -> dict[str, str]:
    return {
        "chunk_id": "TEXT PRIMARY KEY",
        "arxiv_id": "TEXT NOT NULL",
        "record_type": "TEXT NOT NULL",
        "page_content": "TEXT NOT NULL",
        "metadata": "TEXT NOT NULL",
        "created_at": "TEXT NOT NULL",
    }


def _users_expected_columns() -> dict[str, str]:
    return {
        "user_id": "TEXT PRIMARY KEY",
        "username": "TEXT NOT NULL UNIQUE",
        "password_hash": "TEXT NOT NULL",
        "display_name": "TEXT",
        "email": "TEXT",
        "created_at": "TEXT NOT NULL",
        "updated_at": "TEXT NOT NULL",
    }


def _user_sessions_expected_columns() -> dict[str, str]:
    return {
        "session_id": "TEXT PRIMARY KEY",
        "user_id": "TEXT NOT NULL",
        "created_at": "TEXT NOT NULL",
        "expires_at": "TEXT NOT NULL",
        "last_seen_at": "TEXT NOT NULL",
    }


def _chat_threads_expected_columns() -> dict[str, str]:
    return {
        "thread_id": "TEXT PRIMARY KEY",
        "user_id": "TEXT NOT NULL",
        "title": "TEXT NOT NULL",
        "created_at": "TEXT NOT NULL",
        "updated_at": "TEXT NOT NULL",
        "last_query": "TEXT",
    }


def _chat_messages_expected_columns() -> dict[str, str]:
    return {
        "message_id": "TEXT PRIMARY KEY",
        "thread_id": "TEXT NOT NULL",
        "user_id": "TEXT NOT NULL",
        "role": "TEXT NOT NULL",
        "content": "TEXT NOT NULL",
        "metrics": "TEXT",
        "status": "TEXT",
        "created_at": "TEXT NOT NULL",
    }


def _create_table_sql(table_name: str, expected_columns: dict[str, str]) -> str:
    columns_sql = ",\n            ".join(
        f"{column_name} {column_def}"
        for column_name, column_def in expected_columns.items()
    )
    return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {columns_sql}
        )
    """


def _ensure_table_columns(table_name: str, expected_columns: dict[str, str]) -> None:
    inspector = inspect(get_structured_store_engine())
    existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
    missing = [
        (column_name, column_def)
        for column_name, column_def in expected_columns.items()
        if column_name not in existing_columns
    ]
    if not missing:
        return

    with get_structured_store_engine().begin() as connection:
        for column_name, column_def in missing:
            connection.execute(
                text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")
            )


def _table_row_count(table_name: str) -> int:
    with get_structured_store_engine().connect() as connection:
        result = connection.execute(text(f"SELECT count(*) FROM {table_name}"))
        return int(result.scalar_one())


def _maybe_migrate_legacy_turn_memories() -> None:
    if _table_row_count("turn_memories") > 0:
        return

    legacy_path = get_legacy_conversation_memory_path()
    if not os.path.exists(legacy_path):
        return

    if (
        get_storage_backend() == STORAGE_BACKEND_SQLITE
        and os.path.abspath(legacy_path) == os.path.abspath(get_sqlite_storage_path())
    ):
        return

    legacy_connection = sqlite3.connect(legacy_path)
    try:
        table_exists = legacy_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='turn_memories'"
        ).fetchone()
        if not table_exists:
            return

        rows = legacy_connection.execute(
            """
            SELECT
                memory_id,
                thread_id,
                summary,
                keywords,
                full_payload,
                compact_payload,
                embedding,
                importance_score,
                storage_tier,
                memory_kind,
                access_count,
                recall_count,
                last_recalled_at,
                created_at
            FROM turn_memories
            """
        ).fetchall()
    finally:
        legacy_connection.close()

    if not rows:
        return

    payloads = [
        {
            "memory_id": row[0],
            "user_id": None,
            "thread_id": row[1],
            "summary": row[2],
            "keywords": row[3],
            "full_payload": row[4],
            "compact_payload": row[5],
            "embedding": row[6],
            "importance_score": row[7] if row[7] is not None else 0.5,
            "storage_tier": row[8] or "salient",
            "memory_kind": row[9] or "episodic",
            "access_count": row[10] if row[10] is not None else 0,
            "recall_count": row[11] if row[11] is not None else 0,
            "last_recalled_at": row[12],
            "created_at": row[13] or now_utc_timestamp(),
        }
        for row in rows
    ]

    with get_structured_store_engine().begin() as connection:
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
                    :importance_score, :storage_tier, :memory_kind, :access_count, :recall_count,
                    :last_recalled_at, :created_at
                )
                """
            ),
            payloads,
        )


def _maybe_migrate_legacy_hybrid_chunks() -> None:
    if _table_row_count("hybrid_chunks") > 0:
        return

    legacy_path = get_legacy_hybrid_index_path()
    if not os.path.exists(legacy_path):
        return

    payloads: list[dict[str, Any]] = []
    with open(legacy_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = str(record.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            payloads.append(
                {
                    "chunk_id": chunk_id,
                    "arxiv_id": str(record.get("arxiv_id") or ""),
                    "record_type": str(record.get("record_type") or "fulltext_chunk"),
                    "page_content": str(record.get("page_content") or ""),
                    "metadata": json.dumps(record.get("metadata", {}), ensure_ascii=False),
                    "created_at": now_utc_timestamp(),
                }
            )

    if not payloads:
        return

    with get_structured_store_engine().begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO hybrid_chunks(
                    chunk_id, arxiv_id, record_type, page_content, metadata, created_at
                )
                VALUES (
                    :chunk_id, :arxiv_id, :record_type, :page_content, :metadata, :created_at
                )
                """
            ),
            payloads,
        )


@lru_cache(maxsize=1)
def ensure_structured_storage_ready() -> None:
    engine = get_structured_store_engine()
    with engine.begin() as connection:
        connection.execute(text(_create_table_sql("turn_memories", _turn_memories_expected_columns())))
        connection.execute(text(_create_table_sql("hybrid_chunks", _hybrid_chunks_expected_columns())))
        connection.execute(text(_create_table_sql("users", _users_expected_columns())))
        connection.execute(text(_create_table_sql("user_sessions", _user_sessions_expected_columns())))
        connection.execute(text(_create_table_sql("chat_threads", _chat_threads_expected_columns())))
        connection.execute(text(_create_table_sql("chat_messages", _chat_messages_expected_columns())))

    _ensure_table_columns("turn_memories", _turn_memories_expected_columns())
    _ensure_table_columns("hybrid_chunks", _hybrid_chunks_expected_columns())
    _ensure_table_columns("users", _users_expected_columns())
    _ensure_table_columns("user_sessions", _user_sessions_expected_columns())
    _ensure_table_columns("chat_threads", _chat_threads_expected_columns())
    _ensure_table_columns("chat_messages", _chat_messages_expected_columns())

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_turn_memories_user_thread_created "
                "ON turn_memories(user_id, thread_id, created_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_turn_memories_user_thread_importance "
                "ON turn_memories(user_id, thread_id, importance_score)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_hybrid_chunks_record_type "
                "ON hybrid_chunks(record_type)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_hybrid_chunks_arxiv_id "
                "ON hybrid_chunks(arxiv_id)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id "
                "ON user_sessions(user_id)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at "
                "ON user_sessions(expires_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_chat_threads_user_updated "
                "ON chat_threads(user_id, updated_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created "
                "ON chat_messages(thread_id, created_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created "
                "ON chat_messages(user_id, created_at)"
            )
        )

    _maybe_migrate_legacy_turn_memories()
    _maybe_migrate_legacy_hybrid_chunks()


def _fetch_one(query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    ensure_structured_storage_ready()
    with get_structured_store_engine().connect() as connection:
        row = connection.execute(text(query), params or {}).mappings().first()
        return dict(row) if row else None


def _fetch_all(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    ensure_structured_storage_ready()
    with get_structured_store_engine().connect() as connection:
        rows = connection.execute(text(query), params or {}).mappings().all()
        return [dict(row) for row in rows]


def _normalize_username(username: str) -> str:
    return str(username or "").strip().lower()


def _clean_title(title: str | None, fallback: str = "New chat") -> str:
    cleaned = " ".join(str(title or "").split()).strip()
    if not cleaned:
        return fallback
    return cleaned[:80]


def derive_thread_title(query: str | None) -> str:
    return _clean_title(query, fallback="New chat")


def count_registered_users() -> int:
    ensure_structured_storage_ready()
    with get_structured_store_engine().connect() as connection:
        result = connection.execute(text("SELECT count(*) FROM users"))
        return int(result.scalar_one())


def create_user(
    username: str,
    password_hash: str,
    display_name: str | None = None,
    email: str | None = None,
) -> dict[str, Any]:
    ensure_structured_storage_ready()
    created_at = now_utc_timestamp()
    user_payload = {
        "user_id": f"user_{secrets.token_hex(12)}",
        "username": _normalize_username(username),
        "password_hash": password_hash,
        "display_name": str(display_name or "").strip() or None,
        "email": str(email or "").strip() or None,
        "created_at": created_at,
        "updated_at": created_at,
    }
    try:
        with get_structured_store_engine().begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO users(
                        user_id, username, password_hash, display_name, email, created_at, updated_at
                    )
                    VALUES(
                        :user_id, :username, :password_hash, :display_name, :email, :created_at, :updated_at
                    )
                    """
                ),
                user_payload,
            )
    except IntegrityError as exc:
        raise ValueError("username_already_exists") from exc
    return get_user_by_id(user_payload["user_id"]) or {}


def get_user_by_username(username: str) -> dict[str, Any] | None:
    normalized_username = _normalize_username(username)
    if not normalized_username:
        return None
    return _fetch_one(
        """
        SELECT user_id, username, password_hash, display_name, email, created_at, updated_at
        FROM users
        WHERE username = :username
        """,
        {"username": normalized_username},
    )


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    if not str(user_id or "").strip():
        return None
    return _fetch_one(
        """
        SELECT user_id, username, password_hash, display_name, email, created_at, updated_at
        FROM users
        WHERE user_id = :user_id
        """,
        {"user_id": user_id},
    )


def create_user_session(user_id: str, ttl_hours: int = 168) -> dict[str, Any]:
    ensure_structured_storage_ready()
    created_at = now_utc_timestamp()
    expires_at = utc_timestamp_after_hours(ttl_hours)
    payload = {
        "session_id": f"ses_{secrets.token_urlsafe(32)}",
        "user_id": user_id,
        "created_at": created_at,
        "expires_at": expires_at,
        "last_seen_at": created_at,
    }
    with get_structured_store_engine().begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO user_sessions(
                    session_id, user_id, created_at, expires_at, last_seen_at
                )
                VALUES(
                    :session_id, :user_id, :created_at, :expires_at, :last_seen_at
                )
                """
            ),
            payload,
        )
    return payload


def delete_user_session(session_id: str) -> None:
    if not str(session_id or "").strip():
        return
    ensure_structured_storage_ready()
    with get_structured_store_engine().begin() as connection:
        connection.execute(
            text("DELETE FROM user_sessions WHERE session_id = :session_id"),
            {"session_id": session_id},
        )


def touch_user_session(session_id: str) -> None:
    if not str(session_id or "").strip():
        return
    ensure_structured_storage_ready()
    with get_structured_store_engine().begin() as connection:
        connection.execute(
            text(
                """
                UPDATE user_sessions
                SET last_seen_at = :last_seen_at
                WHERE session_id = :session_id
                """
            ),
            {"session_id": session_id, "last_seen_at": now_utc_timestamp()},
        )


def get_active_session(session_id: str) -> dict[str, Any] | None:
    if not str(session_id or "").strip():
        return None
    row = _fetch_one(
        """
        SELECT
            s.session_id,
            s.user_id,
            s.created_at AS session_created_at,
            s.expires_at,
            s.last_seen_at,
            u.username,
            u.password_hash,
            u.display_name,
            u.email,
            u.created_at AS user_created_at,
            u.updated_at AS user_updated_at
        FROM user_sessions s
        JOIN users u ON u.user_id = s.user_id
        WHERE s.session_id = :session_id
        """,
        {"session_id": session_id},
    )
    if not row:
        return None

    expires_at = _parse_utc_timestamp(str(row.get("expires_at") or ""))
    if expires_at and expires_at <= datetime.now(timezone.utc):
        delete_user_session(session_id)
        return None

    return {
        "session_id": row["session_id"],
        "expires_at": row["expires_at"],
        "last_seen_at": row["last_seen_at"],
        "user": {
            "user_id": row["user_id"],
            "username": row["username"],
            "password_hash": row["password_hash"],
            "display_name": row.get("display_name"),
            "email": row.get("email"),
            "created_at": row.get("user_created_at"),
            "updated_at": row.get("user_updated_at"),
        },
    }


def create_chat_thread(user_id: str, title: str | None = None, last_query: str | None = None) -> dict[str, Any]:
    ensure_structured_storage_ready()
    timestamp = now_utc_timestamp()
    payload = {
        "thread_id": secrets.token_hex(16),
        "user_id": user_id,
        "title": _clean_title(title or last_query),
        "created_at": timestamp,
        "updated_at": timestamp,
        "last_query": str(last_query or "").strip() or None,
    }
    with get_structured_store_engine().begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO chat_threads(
                    thread_id, user_id, title, created_at, updated_at, last_query
                )
                VALUES(
                    :thread_id, :user_id, :title, :created_at, :updated_at, :last_query
                )
                """
            ),
            payload,
        )
    return payload


def get_chat_thread(thread_id: str) -> dict[str, Any] | None:
    if not str(thread_id or "").strip():
        return None
    return _fetch_one(
        """
        SELECT thread_id, user_id, title, created_at, updated_at, last_query
        FROM chat_threads
        WHERE thread_id = :thread_id
        """,
        {"thread_id": thread_id},
    )


def get_chat_thread_for_user(user_id: str, thread_id: str) -> dict[str, Any] | None:
    if not str(user_id or "").strip() or not str(thread_id or "").strip():
        return None
    return _fetch_one(
        """
        SELECT thread_id, user_id, title, created_at, updated_at, last_query
        FROM chat_threads
        WHERE thread_id = :thread_id AND user_id = :user_id
        """,
        {"thread_id": thread_id, "user_id": user_id},
    )


def update_chat_thread_activity(thread_id: str, query: str | None = None) -> dict[str, Any] | None:
    thread = get_chat_thread(thread_id)
    if not thread:
        return None

    new_title = thread.get("title") or "New chat"
    if new_title == "New chat" and str(query or "").strip():
        new_title = derive_thread_title(query)

    payload = {
        "thread_id": thread_id,
        "title": _clean_title(new_title),
        "updated_at": now_utc_timestamp(),
        "last_query": str(query or thread.get("last_query") or "").strip() or None,
    }
    ensure_structured_storage_ready()
    with get_structured_store_engine().begin() as connection:
        connection.execute(
            text(
                """
                UPDATE chat_threads
                SET title = :title, updated_at = :updated_at, last_query = :last_query
                WHERE thread_id = :thread_id
                """
            ),
            payload,
        )
    return get_chat_thread(thread_id)


def list_chat_threads_for_user(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    if not str(user_id or "").strip():
        return []
    return _fetch_all(
        """
        SELECT
            t.thread_id,
            t.title,
            t.created_at,
            t.updated_at,
            t.last_query,
            COUNT(m.message_id) AS message_count
        FROM chat_threads t
        LEFT JOIN chat_messages m ON m.thread_id = t.thread_id
        WHERE t.user_id = :user_id
        GROUP BY t.thread_id, t.title, t.created_at, t.updated_at, t.last_query
        ORDER BY t.updated_at DESC
        LIMIT :limit
        """,
        {"user_id": user_id, "limit": max(int(limit), 1)},
    )


def save_chat_message(
    user_id: str,
    thread_id: str,
    role: str,
    content: str,
    metrics: dict[str, Any] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    ensure_structured_storage_ready()
    payload = {
        "message_id": f"msg_{secrets.token_hex(12)}",
        "thread_id": thread_id,
        "user_id": user_id,
        "role": role,
        "content": str(content or "").strip(),
        "metrics": json.dumps(metrics or {}, ensure_ascii=False),
        "status": status,
        "created_at": now_utc_timestamp(),
    }
    with get_structured_store_engine().begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO chat_messages(
                    message_id, thread_id, user_id, role, content, metrics, status, created_at
                )
                VALUES(
                    :message_id, :thread_id, :user_id, :role, :content, :metrics, :status, :created_at
                )
                """
            ),
            payload,
        )
    return {
        "message_id": payload["message_id"],
        "thread_id": thread_id,
        "user_id": user_id,
        "role": role,
        "content": payload["content"],
        "metrics": metrics or {},
        "status": status,
        "created_at": payload["created_at"],
    }


def list_chat_messages_for_user(user_id: str, thread_id: str, limit: int = 200) -> list[dict[str, Any]]:
    if not str(user_id or "").strip() or not str(thread_id or "").strip():
        return []
    return [
        {
            **row,
            "metrics": json.loads(row["metrics"]) if row.get("metrics") else {},
        }
        for row in _fetch_all(
            """
            SELECT
                m.message_id,
                m.thread_id,
                m.user_id,
                m.role,
                m.content,
                m.metrics,
                m.status,
                m.created_at
            FROM chat_messages m
            JOIN chat_threads t ON t.thread_id = m.thread_id
            WHERE m.thread_id = :thread_id AND t.user_id = :user_id
            ORDER BY m.created_at ASC
            LIMIT :limit
            """,
            {"thread_id": thread_id, "user_id": user_id, "limit": max(int(limit), 1)},
        )
    ]


def build_agent_checkpointer(exit_stack: contextlib.ExitStack):
    #===================原代码===========================
    # backend = get_storage_backend()
    # if backend == STORAGE_BACKEND_POSTGRES:
    #     try:
    #         from langgraph.checkpoint.postgres import PostgresSaver
    #     except ModuleNotFoundError as exc:
    #         raise RuntimeError(
    #             "STORAGE_BACKEND=postgresql requires `langgraph-checkpoint-postgres` "
    #             "and `psycopg[binary]` in the runtime environment."
    #         ) from exc
    #
    #     checkpointer = exit_stack.enter_context(
    #         PostgresSaver.from_conn_string(_normalized_database_url())
    #     )
    #     checkpointer.setup()
    #     return checkpointer
    #
    # sqlite_path = get_sqlite_checkpoint_path()
    # _ensure_parent_dir(sqlite_path)
    # connection = exit_stack.enter_context(
    #     contextlib.closing(sqlite3.connect(sqlite_path, check_same_thread=False))
    # )
    # from langgraph.checkpoint.sqlite import SqliteSaver
    #
    # return SqliteSaver(connection)
    #=====================================================
    def build_agent_checkpointer(exit_stack: contextlib.ExitStack):
        # 环境变量控制：DEBUG_AGENT_MEMORY=1 本地内存调试
        if os.getenv("DEBUG_AGENT_MEMORY", "0") == "1":
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

        # 下面是你原来完整的Postgres/Sqlite生产逻辑不动
        backend = get_storage_backend()
        if backend == STORAGE_BACKEND_POSTGRES:
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "STORAGE_BACKEND=postgresql requires `langgraph-checkpoint-postgres` "
                    "and `psycopg[binary]` in the runtime environment."
                ) from exc

            checkpointer = exit_stack.enter_context(
                PostgresSaver.from_conn_string(_normalized_database_url())
            )
            checkpointer.setup()
            return checkpointer

        sqlite_path = get_sqlite_checkpoint_path()
        _ensure_parent_dir(sqlite_path)
        connection = exit_stack.enter_context(
            contextlib.closing(sqlite3.connect(sqlite_path, check_same_thread=False))
        )
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver(connection)
    #======启动之前cmd里设置========
    #本地调试启动前设置环境变量
    # set DEBUG_AGENT_MEMORY = 1
    # python main.py
    #=============================