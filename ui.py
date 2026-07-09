import json
import os
from typing import Any, Iterator

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("DEEPRESEARCH_API_URL", "http://localhost:8000/v1/chat")
API_BASE = API_URL.removesuffix("/v1/chat")
STREAM_URL = os.getenv("DEEPRESEARCH_STREAM_URL", API_BASE + "/v1/chat/stream")
HEALTH_URL = os.getenv("DEEPRESEARCH_HEALTH_URL", API_BASE + "/")
LOGIN_URL = API_BASE + "/auth/login"
REGISTER_URL = API_BASE + "/auth/register"
LOGOUT_URL = API_BASE + "/auth/logout"
THREADS_URL = API_BASE + "/v1/threads"

# Create a shared session that bypasses system proxy settings.
# VPN software often sets a system proxy that cannot reach localhost,
# causing spurious 502 errors on every API call.
_session = requests.Session()
_session.trust_env = False

st.set_page_config(
    page_title="DeepResearch Pro | API Client",
    page_icon="DR",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding-top: 2rem;}
        .stChatMessage {border-radius: 12px; margin-bottom: 1rem;}
        div[data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
        }
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin-top: 10vh;
        }
        .welcome-container h1 {
            font-size: 3.5rem;
            background: linear-gradient(90deg, #2c3e50, #4ca1af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .thread-button button {
            text-align: left;
            white-space: normal;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "threads" not in st.session_state:
    st.session_state.threads = []
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None


def fetch_service_health() -> dict[str, Any]:
    try:
        response = _session.get(HEALTH_URL, timeout=2)
        if response.status_code != 200:
            return {"online": False, "status_code": response.status_code}
        payload = response.json()
        return {
            "online": True,
            "status_code": response.status_code,
            "payload": payload if isinstance(payload, dict) else {},
        }
    except requests.RequestException as exc:
        return {"online": False, "error": str(exc), "payload": {}}


SERVICE_HEALTH = fetch_service_health()
REGISTRATION_OPEN = bool(SERVICE_HEALTH.get("payload", {}).get("registration_open", False))


def api_headers() -> dict[str, str]:
    token = st.session_state.auth_token
    return {"Authorization": f"Bearer {token}"} if token else {}


def parse_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    return json.dumps(payload, ensure_ascii=False)


def render_metrics(metrics: dict) -> None:
    if not metrics:
        return

    st.divider()
    columns = st.columns(len(metrics))
    for column, (name, value) in zip(columns, metrics.items()):
        label = name.split("_", 1)[-1].replace("_", " ")
        column.metric(label, f"{value}s")


def iter_sse_events(response: requests.Response) -> Iterator[tuple[str, dict]]:
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()

        if not line:
            if data_lines:
                payload_text = "\n".join(data_lines)
                try:
                    payload = json.loads(payload_text)
                except json.JSONDecodeError:
                    payload = {"raw": payload_text}
                yield event_name, payload
            event_name = "message"
            data_lines = []
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.partition(":")[2].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line.partition(":")[2].strip())

    if data_lines:
        payload_text = "\n".join(data_lines)
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = {"raw": payload_text}
        yield event_name, payload


def reset_auth_state() -> None:
    st.session_state.auth_token = None
    st.session_state.current_user = None
    st.session_state.threads = []
    st.session_state.active_thread_id = None
    st.session_state.messages = []


def refresh_threads() -> None:
    if not st.session_state.auth_token:
        st.session_state.threads = []
        return
    response = _session.get(THREADS_URL, headers=api_headers(), timeout=10)
    if response.status_code == 401:
        reset_auth_state()
        st.rerun()
    if response.status_code != 200:
        raise RuntimeError(parse_error(response))
    st.session_state.threads = response.json()


def load_thread_messages(thread_id: str) -> None:
    response = _session.get(
        f"{THREADS_URL}/{thread_id}/messages",
        headers=api_headers(),
        timeout=180,
    )
    if response.status_code == 401:
        reset_auth_state()
        st.rerun()
    if response.status_code != 200:
        raise RuntimeError(parse_error(response))
    st.session_state.active_thread_id = thread_id
    st.session_state.messages = response.json()


def bootstrap_user_session(payload: dict[str, Any]) -> None:
    st.session_state.auth_token = payload["session_token"]
    st.session_state.current_user = payload["user"]
    refresh_threads()
    if st.session_state.threads:
        load_thread_messages(st.session_state.threads[0]["thread_id"])
    else:
        st.session_state.active_thread_id = None
        st.session_state.messages = []


def create_new_thread() -> None:
    response = _session.post(
        THREADS_URL,
        headers=api_headers(),
        json={"title": "New chat"},
        timeout=10,
    )
    if response.status_code != 200:
        raise RuntimeError(parse_error(response))
    thread = response.json()
    refresh_threads()
    st.session_state.active_thread_id = thread["thread_id"]
    st.session_state.messages = []


def render_auth_forms(prefix: str) -> None:
    tab_labels = ["Login", "Register"] if REGISTRATION_OPEN else ["Login", "Register Closed"]
    login_tab, register_tab = st.tabs(tab_labels)

    with login_tab:
        with st.form(f"{prefix}_login_form", clear_on_submit=False):
            login_username = st.text_input("Username", key=f"{prefix}_login_username")
            login_password = st.text_input("Password", type="password", key=f"{prefix}_login_password")
            login_submitted = st.form_submit_button("Sign in", use_container_width=True)
        if login_submitted:
            response = _session.post(
                LOGIN_URL,
                json={"username": login_username, "password": login_password},
                timeout=10,
            )
            if response.status_code == 200:
                bootstrap_user_session(response.json())
                st.rerun()
            else:
                st.error(parse_error(response))

    with register_tab:
        if not REGISTRATION_OPEN:
            st.info("Registration is closed because this site already has an owner account. Please use Login.")
        else:
            with st.form(f"{prefix}_register_form", clear_on_submit=False):
                register_username = st.text_input("Create username", key=f"{prefix}_register_username")
                register_password = st.text_input("Create password", type="password", key=f"{prefix}_register_password")
                register_display_name = st.text_input("Display name (optional)", key=f"{prefix}_register_display_name")
                register_email = st.text_input("Email (optional)", key=f"{prefix}_register_email")
                register_submitted = st.form_submit_button("Create account", use_container_width=True)
            if register_submitted:
                response = _session.post(
                    REGISTER_URL,
                    json={
                        "username": register_username,
                        "password": register_password,
                        "display_name": register_display_name or None,
                        "email": register_email or None,
                    },
                    timeout=10,
                )
                if response.status_code == 200:
                    bootstrap_user_session(response.json())
                    st.rerun()
                else:
                    st.error(parse_error(response))


def render_message_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_metrics(message.get("metrics", {}))


with st.sidebar:
    st.markdown("### DeepResearch Pro")
    st.caption("Login-aware research client")
    st.divider()

    if st.session_state.current_user:
        user = st.session_state.current_user
        st.success(f"Signed in as {user['username']}")
        if user.get("display_name"):
            st.caption(user["display_name"])

        action_columns = st.columns(2)
        if action_columns[0].button("New chat", use_container_width=True):
            try:
                create_new_thread()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if action_columns[1].button("Refresh", use_container_width=True):
            try:
                refresh_threads()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        if st.button("Log out", use_container_width=True):
            try:
                _session.post(LOGOUT_URL, headers=api_headers(), timeout=10)
            finally:
                reset_auth_state()
                st.rerun()

        st.divider()
        st.markdown("#### Your chats")
        if not st.session_state.threads:
            st.caption("No chats yet. Start a new one.")
        for thread in st.session_state.threads:
            label = thread["title"]
            if thread.get("message_count"):
                label = f"{label} ({thread['message_count']})"
            if st.button(label, key=f"thread_{thread['thread_id']}", use_container_width=True):
                try:
                    load_thread_messages(thread["thread_id"])
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    else:
        st.caption("Use the sign-in form in the main panel if the sidebar is collapsed.")

    st.divider()
    st.markdown("#### Service Health")
    if SERVICE_HEALTH.get("online"):
        st.success("API Server: Online")
        st.caption(f"Registration open: {REGISTRATION_OPEN}")
    elif "status_code" in SERVICE_HEALTH:
        st.error(f"API Server: {SERVICE_HEALTH['status_code']}")
    else:
        st.error("API Server: Offline")


if not st.session_state.current_user:
    hero_col, auth_col = st.columns([1.2, 1], gap="large")
    with hero_col:
        st.markdown(
            """
            <div class="welcome-container">
                <h1>DeepResearch</h1>
                <p style="font-size:1.4rem; color:gray;">Sign in to access your personal research workspace.</p>
                <p style="font-size:14px; color:#8693ab; margin-top:10px;">
                    Your chats and long-term memory will be scoped to your account.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with auth_col:
        st.markdown("### Sign In")
        st.caption("Create the first account here, then log in from this page next time.")
        render_auth_forms("main")
else:
    active_label = st.session_state.active_thread_id or "new chat"
    header_col, action_col = st.columns([4, 1])
    with header_col:
        st.markdown("## DeepResearch Session")
        st.caption(f"user={st.session_state.current_user['username']} | thread={active_label}")
    with action_col:
        if st.button("New chat", key="main_new_chat", use_container_width=True):
            try:
                create_new_thread()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    if st.session_state.messages:
        render_message_history()
    else:
        st.info("Start a new chat or open one from the sidebar.")


if st.session_state.current_user and (user_input := st.chat_input("Describe your research task...")):
    st.session_state.messages.append({"role": "user", "content": user_input, "metrics": {}, "status": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        captured_answer = ""
        captured_metrics: dict[str, Any] = {}
        captured_status = "success"
        resolved_thread_id = st.session_state.active_thread_id

        with st.status("Connecting to the research engine...", expanded=True) as status:
            try:
                with _session.post(
                    STREAM_URL,
                    headers=api_headers(),
                    json={"query": user_input, "thread_id": st.session_state.active_thread_id},
                    timeout=(5, 300),
                    stream=True,
                ) as response:
                    if response.status_code == 401:
                        reset_auth_state()
                        status.update(label="Session expired", state="error")
                        st.error("Please log in again.")
                    elif response.status_code != 200:
                        status.update(label="Streaming request failed", state="error")
                        st.error(f"Backend error: {parse_error(response)}")
                    else:
                        for event_name, payload in iter_sse_events(response):
                            if event_name == "status":
                                label = payload.get("label", "Working...")
                                detail = payload.get("detail", "")
                                state = payload.get("state", "running")
                                status.update(label=label, state=state if state in {"running", "complete", "error"} else "running")
                                if detail:
                                    status.write(detail)
                            elif event_name == "update":
                                node = payload.get("node", "agent")
                                label = payload.get("label", "Update")
                                detail = payload.get("detail", "")
                                status.write(f"[{node}] {label}")
                                if detail:
                                    status.write(detail)
                            elif event_name == "answer":
                                captured_answer = payload.get("answer", captured_answer)
                                if captured_answer:
                                    answer_placeholder.markdown(captured_answer)
                            elif event_name == "final":
                                resolved_thread_id = payload.get("thread_id", resolved_thread_id)
                                captured_answer = payload.get("answer", captured_answer)
                                captured_metrics = payload.get("metrics", {})
                                captured_status = payload.get("status", "success")
                                if captured_answer:
                                    answer_placeholder.markdown(captured_answer)
                                if captured_status == "clarify":
                                    final_label = "Need clarification"
                                elif captured_status == "partial":
                                    final_label = "Streaming complete with partial capture"
                                else:
                                    final_label = "Streaming complete"
                                status.update(label=final_label, state="complete")
                            elif event_name == "error":
                                captured_status = "error"
                                status.update(label="Streaming interrupted", state="error")
                                st.error(payload.get("detail", "Unknown streaming error"))

                if resolved_thread_id:
                    st.session_state.active_thread_id = resolved_thread_id
                if captured_answer:
                    render_metrics(captured_metrics)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": captured_answer,
                            "metrics": captured_metrics,
                            "status": captured_status,
                        }
                    )
                    refresh_threads()
                elif captured_status != "error":
                    fallback_message = "The stream finished without a final answer."
                    answer_placeholder.markdown(fallback_message)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": fallback_message,
                            "metrics": captured_metrics,
                            "status": "partial",
                        }
                    )
                    refresh_threads()
            except requests.RequestException as exc:
                status.update(label="Connection interrupted", state="error")
                st.error(f"Could not reach the API service: {exc}")
