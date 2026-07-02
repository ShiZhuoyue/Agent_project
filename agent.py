import atexit
import contextlib
import json
import operator
import os
import re
import time
from typing import Annotated, Any, List, Optional
from uuid import uuid4

from typing_extensions import TypedDict

from conversation_memory import (
    load_recent_turn_memories,
    load_turn_memories,
    search_turn_memories,
    store_turn_memory,
)
from research_harness import build_execution_plan, build_structured_request
from storage import build_agent_checkpointer

from observer_audit import AuditRecord
from observer_audit import audit_and_correct
from observer_audit import PaperItem  # 三级文献分级精读：论文打分类型

_CHECKPOINTER_STACK = contextlib.ExitStack()
atexit.register(_CHECKPOINTER_STACK.close)


class AgentState(TypedDict):
    input: str
    user_id: str
    thread_id: str
    normalized_request: dict
    approved_plan: List[dict]
    memory_candidates: list[dict]
    recalled_memories: list[dict]
    risk_flags: Annotated[List[str], operator.add]
    final_mode: str
    past_steps: Annotated[List[str], operator.add]
    messages: Annotated[List[Any], operator.add]
    metrics: Annotated[dict, operator.ior]
    # ========== Observer审计纠偏闭环新增状态字段 ==========
    # audit_logs: 累积存储每轮Observer审计的结构化日志，不使用reducer，
    # 由 observer_node 通过 audit_and_correct 返回完整列表直接覆盖。
    audit_logs: List[dict]
    # current_defect: 当前轮次Observer检测到的缺陷类型
    #   - missing_info: 文献数量不足或覆盖领域不全
    #   - view_conflict: 多篇论文核心结论互相矛盾
    #   - outdated: 绝大多数文献年份过旧
    #   - None: 本轮检索结果无缺陷
    current_defect: Optional[str]
    # suggest_keywords: Observer建议的修正/补充检索关键词，
    # 在重规划时由 planner_node 注入prompt用于生成更精准的检索计划。
    suggest_keywords: List[str]
    # need_replan: Observer判定是否需要触发重规划闭环。
    # True → route_after_observer 路由回 planner；False → 进入 synthesizer。
    need_replan: bool
    # max_replan_times: 最大重规划次数上限（默认3），防止死循环。
    max_replan_times: int
    # current_replan_round: 当前已执行的重规划轮次计数器，
    # 每次触发重规划时 observer_node 自增1，达到 max_replan_times 后强制终止闭环。
    current_replan_round: int
    # ========== 三级文献分级精读：新增状态字段 ==========
    # rated_papers: Observer 审计后对每篇论文的相关度打分与阅读等级建议，
    # 由 audit_and_correct 返回，供 executor_read 分层精读节点消费。
    # 每项为 PaperItem: {title, relevance_score(0-10), read_level(coarse/medium/deep), read_reason}
    rated_papers: List[dict]



SEARCH_RESULT_PATTERN = re.compile(
    r"^\d+\.\s+(?P<title>.*?)\s+\|\s+source=(?P<source>[^|]+)\s+\|\s+year=(?P<year>[^|]+)\s+\|\s+score=(?P<score>[0-9.]+)",
    re.MULTILINE,
)
FORMATTED_RESULT_PATTERN = re.compile(
    r"^\d+\.\s+(?P<title>.*?)\s+\((?P<year>[^,]+),\s+source:\s+(?P<source>[^,]+),\s+score:\s+(?P<score>[0-9.]+)\)",
    re.MULTILINE,
)
RANKED_RESULT_WITH_PIPE_PATTERN = re.compile(
    r"^\d+\.\s+(?P<title>.*?)\s+\|\s+source=(?P<source>[^|]+)\s+\|\s+year=(?P<year>[^|]+)\s+\|\s+score=(?P<score>[0-9.]+)",
    re.MULTILINE,
)


def _sanitize_answer_text(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n")
    cleaned = re.sub(r"([!?])\1{5,}", r"\1\1\1", cleaned)
    cleaned = re.sub(r"(\.{4,})", "...", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()


def _parse_search_results(tool_facts: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for match in SEARCH_RESULT_PATTERN.finditer(str(tool_facts or "")):
        results.append(
            {
                "title": match.group("title").strip(),
                "source": match.group("source").strip(),
                "year": match.group("year").strip(),
                "score": match.group("score").strip(),
            }
        )
    if results:
        return results
    for match in FORMATTED_RESULT_PATTERN.finditer(str(tool_facts or "")):
        results.append(
            {
                "title": match.group("title").strip(),
                "source": match.group("source").strip(),
                "year": match.group("year").strip(),
                "score": match.group("score").strip(),
            }
        )
    return results


def _parse_ranked_candidates(tool_facts: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in str(tool_facts or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = RANKED_RESULT_WITH_PIPE_PATTERN.match(line)
        if match:
            current = {
                "title": match.group("title").strip(),
                "source": match.group("source").strip(),
                "year": match.group("year").strip(),
                "score": match.group("score").strip(),
                "summary_hint": "",
            }
            candidates.append(current)
            continue

        if current is not None and line.lower().startswith("summary_hint:"):
            current["summary_hint"] = line.split(":", 1)[1].strip()

    return candidates


def _is_title_seeking_request(question: str) -> bool:
    lowered = str(question or "").lower()
    patterns = (
        "which paper",
        "what paper",
        "where can i find",
        "could you recommend",
        "can you recommend",
        "what are some studies",
        "what are some papers",
        "first proved",
    )
    return any(pattern in lowered for pattern in patterns)


def _build_grounded_search_answer(normalized_request: dict[str, Any], tool_facts: str) -> str:
    results = _parse_search_results(tool_facts)
    if not results:
        return ""

    question = str(normalized_request.get("question") or "").strip()
    lines: list[str] = []
    if _is_title_seeking_request(question):
        lines.append("Top grounded matches I found:")
    else:
        lines.append("Top results I found:")

    for index, item in enumerate(results[:3], start=1):
        year_text = item["year"] if item["year"] != "n/a" else "year unknown"
        lines.append(
            f"{index}. {item['title']} ({year_text}, source: {item['source']}, score: {item['score']})"
        )

    lines.append(
        "I am only using retrieved evidence here, so I cannot claim an exact match unless the retrieved titles make it explicit."
    )
    return "\n".join(lines)


def _build_grounded_search_summary_answer(
    normalized_request: dict[str, Any],
    tool_facts: str,
    user_input: str,
) -> str:
    candidates = _parse_ranked_candidates(tool_facts)
    if not candidates:
        return ""

    requested_count = int(normalized_request.get("count") or 2)
    top_candidates = candidates[: max(1, min(requested_count, len(candidates), 3))]
    prefers_chinese = bool(re.search(r"[\u4e00-\u9fff]", str(user_input or "")))

    if prefers_chinese:
        lines = ["根据当前检索结果，我先总结最相关的几篇论文："]
        for index, item in enumerate(top_candidates, start=1):
            year_text = item["year"] if item["year"] != "n/a" else "年份未知"
            summary_hint = item.get("summary_hint") or "工具返回了论文标题，但摘要信息不足，因此这里只能确认它是高相关检索结果。"
            lines.append(f"{index}. {item['title']}（{year_text}，来源：{item['source']}，分数：{item['score']}）")
            lines.append(f"   论文大意：{summary_hint}")
            lines.append("   相关性：这是当前检索结果里排名靠前的候选，与您的问题直接相关。")
        lines.append("以上内容仅基于当前检索到的标题和摘要线索整理，没有补充未检索到的论文信息。")
        return "\n".join(lines)

    lines = ["Here are the top retrieved papers and grounded summaries:"]
    for index, item in enumerate(top_candidates, start=1):
        year_text = item["year"] if item["year"] != "n/a" else "year unknown"
        summary_hint = item.get("summary_hint") or "The tool returned the title, but there was not enough abstract evidence to summarize it further."
        lines.append(f"{index}. {item['title']} ({year_text}, source: {item['source']}, score: {item['score']})")
        lines.append(f"   What it is about: {summary_hint}")
        lines.append("   Why it is relevant: This candidate ranked near the top of the current retrieval results for your request.")
    lines.append("This answer is grounded only in the retrieved titles and summary hints above.")
    return "\n".join(lines)


def _latest_tool_output(messages: list[Any]) -> str:
    for message in reversed(messages or []):
        if message.__class__.__name__ == "ToolMessage":
            return _stringify_content(getattr(message, "content", ""))
    return ""


def _is_summary_followup_request(text: str) -> bool:
    lowered = str(text or "").lower()
    hints = (
        "summarize",
        "summary",
        "overview",
        "analyze",
        "explain this paper",
        "总结",
        "总结一下",
        "概述",
        "解读",
        "分析一下",
    )
    return any(hint in lowered or hint in text for hint in hints)


def _search_request_needs_summary(text: str) -> bool:
    lowered = str(text or "").lower()
    hints = (
        "summarize",
        "summary",
        "overview",
        "analyze",
        "analysis",
        "总结",
        "概述",
        "分析",
        "并总结",
        "并概述",
    )
    return any(hint in lowered or hint in text for hint in hints)


def _requested_result_index(text: str) -> int:
    raw_text = str(text or "")
    lowered = raw_text.lower()
    ordinal_patterns = (
        (r"第\s*1\s*篇", 0),
        (r"第\s*2\s*篇", 1),
        (r"第\s*3\s*篇", 2),
        (r"第一篇", 0),
        (r"第二篇", 1),
        (r"第三篇", 2),
        (r"\bfirst\b", 0),
        (r"\bsecond\b", 1),
        (r"\bthird\b", 2),
        (r"\btop\s*1\b", 0),
        (r"\btop\s*2\b", 1),
        (r"\btop\s*3\b", 2),
    )
    for pattern, index in ordinal_patterns:
        if re.search(pattern, lowered, re.IGNORECASE) or re.search(pattern, raw_text):
            return index
    return 0


def _extract_titles_from_memory_payload(memory: dict[str, Any]) -> list[str]:
    payload = memory.get("payload", {})
    if not isinstance(payload, dict):
        return []
    titles: list[str] = []
    normalized_request = payload.get("normalized_request", {})
    if isinstance(normalized_request, dict):
        paper_title = normalized_request.get("paper_title")
        if paper_title:
            titles.append(str(paper_title))
    for tool_output in payload.get("tool_outputs", []) or []:
        titles.extend([item["title"] for item in _parse_search_results(str(tool_output or ""))])
    return titles


def _resolve_followup_paper_title(state: AgentState, user_input: str) -> str | None:
    requested_index = _requested_result_index(user_input)

    recent_titles = [item["title"] for item in _parse_search_results(_latest_tool_output(state.get("messages", [])))]
    if recent_titles:
        if requested_index < len(recent_titles):
            return recent_titles[requested_index]
        return recent_titles[0]

    for message in reversed(state.get("messages", [])):
        if message.__class__.__name__ == "AIMessage" and not getattr(message, "tool_calls", None):
            ai_titles = [item["title"] for item in _parse_search_results(_stringify_content(getattr(message, "content", "")))]
            if ai_titles:
                if requested_index < len(ai_titles):
                    return ai_titles[requested_index]
                return ai_titles[0]

    recalled_titles: list[str] = []
    for memory in state.get("recalled_memories", []):
        recalled_titles.extend(_extract_titles_from_memory_payload(memory))
    deduped_titles: list[str] = []
    seen: set[str] = set()
    for title in recalled_titles:
        normalized = title.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_titles.append(title.strip())
    if deduped_titles:
        if requested_index < len(deduped_titles):
            return deduped_titles[requested_index]
        return deduped_titles[0]
    return None

def create_research_agent():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    from tools import arxiv_research_tool, citation_stat_tool, query_research_db, summarize_paper_tool

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    planner_llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0.1,
        openai_api_key=api_key,
        base_url=base_url,
        request_timeout=60,
    )
    tools = [arxiv_research_tool, citation_stat_tool, query_research_db, summarize_paper_tool]

    def _format_recent_dialogue(messages: list[Any], limit: int = 4) -> str:
        dialogue: list[str] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                content = _stringify_content(getattr(message, "content", ""))
                if content:
                    dialogue.append(f"user: {content}")
            elif isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                content = _stringify_content(getattr(message, "content", ""))
                if content:
                    dialogue.append(f"assistant: {content}")
        return "\n".join(dialogue[-limit:])

    def _format_memory_candidates(candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return "none"
        return "\n".join(
            [
                f"{item['memory_id']} | score={item['score']:.3f} | relevance={item.get('relevance_score', 0.0):.3f} | "
                f"importance={item.get('importance_score', 0.0):.3f} | recency={item.get('recency_score', 0.0):.3f} | "
                f"reuse={item.get('reuse_score', 0.0):.3f} | tier={item.get('storage_tier', 'salient')} | "
                f"kind={item.get('memory_kind', 'episodic')} | summary={item['summary']} | "
                f"keywords={', '.join(str(keyword) for keyword in item.get('keywords', []))}"
                for item in candidates
            ]
        )

    def _format_recalled_memories(memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "none"

        rendered: list[str] = []
        for memory in memories:
            payload = memory.get("payload", {})
            compact_payload = memory.get("compact_payload", {}) if isinstance(memory.get("compact_payload"), dict) else {}
            tool_outputs = payload.get("tool_outputs", []) if isinstance(payload, dict) else []
            constraints = payload.get("constraints", []) if isinstance(payload, dict) else []
            decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
            entities = payload.get("entities", []) if isinstance(payload, dict) else []
            open_loops = payload.get("open_loops", []) if isinstance(payload, dict) else []
            rendered.append(
                (
                    f"memory_id: {memory['memory_id']}\n"
                    f"tier: {memory.get('storage_tier', 'salient')} | kind: {memory.get('memory_kind', 'episodic')} | "
                    f"importance: {memory.get('importance_score', 0.0):.3f}\n"
                    f"summary: {memory.get('summary', '')}\n"
                    f"user_input: {payload.get('user_input', '') if isinstance(payload, dict) else ''}\n"
                    f"assistant_answer: {_truncate_text(payload.get('assistant_answer', ''), 900) if isinstance(payload, dict) else ''}\n"
                    f"goal: {payload.get('goal', '') if isinstance(payload, dict) else ''}\n"
                    f"constraints: {json.dumps(constraints or compact_payload.get('constraints', []), ensure_ascii=False)}\n"
                    f"decisions: {json.dumps(decisions or compact_payload.get('decisions', []), ensure_ascii=False)}\n"
                    f"entities: {json.dumps(entities or compact_payload.get('entities', []), ensure_ascii=False)}\n"
                    f"open_loops: {json.dumps(open_loops or compact_payload.get('open_loops', []), ensure_ascii=False)}\n"
                    f"normalized_request: {json.dumps(payload.get('normalized_request', {}) if isinstance(payload, dict) else {}, ensure_ascii=False)}\n"
                    f"tool_outputs: {json.dumps(tool_outputs[:2] if isinstance(tool_outputs, list) else [], ensure_ascii=False)}"
                )
            )
        return "\n\n".join(rendered)

    def memory_router_node(state: AgentState):
        start = time.time()
        latest_user_input = _get_latest_user_input(state)
        user_id = state.get("user_id")
        thread_id = state.get("thread_id") or "default"
        candidates = search_turn_memories(user_id, thread_id, latest_user_input, limit=4)
        recalled_memories: list[dict[str, Any]] = []
        followup_like = _looks_like_memory_reference(latest_user_input) or _is_summary_followup_request(
            latest_user_input
        )

        if candidates:
            router_prompt = (
                "You are deciding whether the agent should recall long-term conversation memory.\n"
                "Return JSON only with keys: use_memory, selected_ids, reason.\n"
                "Use memory only when the current user turn depends on earlier decisions, unresolved tasks, "
                "references like 'continue', 'same as before', 'that paper', prior preferences, or earlier constraints.\n"
                "selected_ids must contain only ids from the candidate list.\n\n"
                f"recent_dialogue:\n{_format_recent_dialogue(state.get('messages', []), limit=4)}\n\n"
                f"latest_user_input:\n{latest_user_input}\n\n"
                f"candidate_memory_cards:\n{_format_memory_candidates(candidates)}"
            )
            try:
                router_response = planner_llm.invoke(
                    [
                        SystemMessage(content="You are a strict JSON memory router."),
                        HumanMessage(content=router_prompt),
                    ]
                )
                router_payload = _parse_json_object(getattr(router_response, "content", router_response))
            except Exception:
                router_payload = {}

            selected_ids = [
                str(memory_id)
                for memory_id in router_payload.get("selected_ids", [])
                if isinstance(memory_id, str) and any(item["memory_id"] == memory_id for item in candidates)
            ]
            if followup_like and not selected_ids and candidates:
                selected_ids = [candidates[0]["memory_id"]]

            if router_payload.get("use_memory") or selected_ids:
                recalled_memories = load_turn_memories(user_id, thread_id, selected_ids[:2])

        if followup_like and not recalled_memories:
            recalled_memories = load_recent_turn_memories(
                user_id,
                thread_id,
                limit=2,
                detail_mode="auto",
            )

        return {
            "memory_candidates": candidates,
            "recalled_memories": recalled_memories,
            "metrics": {"00_Memory_Router": round(time.time() - start, 2)},
        }
    #======================================
    def extract_core_chinese_subject(query: str) -> str:
        # 剔除数字、量词、形容词干扰文字
        noise_pattern = r"\d+|三篇|两篇|最火|最热|最新|顶尖|去年|最近|然后|先帮我|发邮件|办公室"
        clean_text = re.sub(noise_pattern, "", query)
        # 只提取所有汉字
        chinese_words = re.findall(r'[\u4e00-\u9fff]+', clean_text)
        return "".join(chinese_words).strip()
    #=======================================


    def planner_node(state: AgentState):
        start = time.time()
        raw_user_input = _get_latest_user_input(state)
        planning_instruction = (
            "【最高优先级强制规则：topic字段提纯规范，违反视为解析错误】\n"
            "1. 提取topic时，仅保留纯研究核心名词/专业术语，**必须彻底删除所有口语修饰、量词、时间、动作助词**：\n"
            "需删除词汇清单：帮我、检索、查找、有关、最新、最近、去年、数字、\\d+、篇、的论文、的文章、一下、那、几、次、下载、搜索、最、很、非常、hottest、latest、superlatives、demonstratives\n"
            "2. 严格保留用户原始语言：中文提问只输出中文主题，英文提问只输出英文主题，严禁互相翻译；\n"
            "3. 标准示例（严格模仿输出）：\n"
            "用户输入：常温超导最火的那 3 篇 → topic='常温超导'\n"
            "用户输入：the latest 5 papers on diffusion models → topic='diffusion models'\n"
            "用户输入：最近关于大语言模型的最新研究 → topic='大语言模型'\n"
            "用户输入：帮我检索一下最新的有关AI的论文 → topic='AI'\n"
            "用户输入：找混合专家模型(MoE)的文献不要超过2篇 → topic='混合专家模型(MoE)'\n"
            "4. 禁止在topic里保留动作、时间、数量、语气词，topic只能是纯科研关键词，不能是完整句子片段。\n\n"
            "Allowed intents: search_papers, query_local_db, summarize_paper, citation_stats, clarify.\n"
            "Return JSON with exactly these keys:\n"
            '{"intent":"","topic":null,"paper_title":null,"author":null,"year":null,'
            '"category":null,"citation_count":null,"count":null,"question":null,'
            '"clarification_question":null,"ignored_intents":[]}\n'
            "General Rules:\n"
            "- Use null for missing fields.\n"
            "- Do not invent paper titles, authors, years, categories, or citation thresholds.\n"
            "- citation_count means the minimum desired citation threshold.\n"
            "- If the user asks for recent or latest work, keep year as null unless the user gave an explicit year.\n"
            "- Set category only when the user explicitly states an arXiv category or clearly names a canonical domain like NLP or computer vision.\n"
            "- Preserve distinctive technical terminology/acronyms/method names ONLY (e.g. MoE, C++ & C#, Sora), do NOT preserve ordinary descriptive filler words.\n"
            "- Do not collapse a narrow request into a broad field label. Keep specific method terms intact.\n"
            "- Use query_local_db only when the user explicitly refers to local, indexed, downloaded, or vector-database content.\n"
            "- If the request is mixed, keep the main research intent and place unrelated actions into ignored_intents.\n"
            "- GUARDRAIL: When the user has not provided a clear research topic, paper title, author, or category, "
            "do NOT generate a retrieval call (search_papers or citation_stats). Use intent=clarify instead. "
            "This prevents wasteful tool calls with empty or meaningless search filters.\n"
            "- GUARDRAIL: A valid topic must contain substantive keywords identifying a specific research area, "
            "method, or subject (e.g., 'diffusion models', 'graph neural networks', '检索增强生成'). "
            "Single generic words without a concrete subject (e.g., 'latest', 'recent', 'best', 'new') are NOT valid topics. "
            "If the user only says '搜论文' or '找文献' without naming a subject, use intent=clarify.\n"
            "- TOOL ROUTING RULE: When the user asks to compute citation statistics (average, sum, sort by citations) for N papers, "
            "use intent=citation_stats. The execution plan will always use arxiv_research_tool FIRST to retrieve papers, "
            "then citation_stat_tool SECOND to compute statistics. Never skip the retrieval step.\n"
            "- TOOL ROUTING RULE: arxiv_research_tool only retrieves papers and reports per-paper citation_count. "
            "It CANNOT compute averages or sums. citation_stat_tool handles all statistical computation.\n"
            "- If the request is not actionable, use intent=clarify."
        )
        memory_context = _format_recalled_memories(state.get("recalled_memories", []))
        # ========== Observer审计纠偏闭环：注入上轮审计反馈 ==========
        # 当 Observer 判定需要重规划时，从 state 中读取上轮缺陷描述和
        # 建议的修正关键词，注入到 planner 的 prompt 中，引导生成更精准的检索计划。
        audit_feedback_context: str = ""
        current_defect: Optional[str] = state.get("current_defect")
        suggest_keywords: List[str] = state.get("suggest_keywords", [])
        if current_defect or suggest_keywords:
            audit_feedback_context = "\n\n【Observer审计纠偏反馈 - 请根据以下问题重新规划检索】\n"
            if current_defect:
                audit_feedback_context += f"上轮检索缺陷类型: {current_defect}\n"
            if suggest_keywords:
                audit_feedback_context += (
                    f"建议补充检索关键词: {', '.join(suggest_keywords)}\n"
                    f"请在本次规划中优先使用上述关键词进行检索，弥补上轮文献覆盖不足的问题。\n"
                )
            audit_feedback_context += "请结合上述反馈重新生成 topic 和检索计划，确保文献覆盖面、时效性和一致性。\n"

        response = planner_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "HARD NON-NEGOTIABLE RULE:\n"
                        "The topic field MUST remain Chinese.DO NOT translate Chinese keywords into English under any circumstance."
                        "Do NOT automatically convert topic into English for arxiv API.The backend will handle translation automatically."
                        "Any English value in topic will cause evaluation failure."
                        "You are a research request normalizer for a tool-using research agent. "
                        "Use conversation history and recalled memory only when needed to resolve references. "
                        "Output one strict JSON object only and never add prose. "
                        "Do not invent missing facts. "
                        "When the user names a rare method, metric, architecture detail, or benchmark, keep it in topic instead of abstracting it away. "
                        "ABSOLUTE RULE: topic must be Chinese raw text. Never translate Chinese words into English. The backend will translate for arxiv automatically.\n"
                        "When extracting topic, strictly preserve the user's original language. Do not translate into English. Do not perform language conversion."
                        "Strip noise modifiers from topic: remove superlatives (最火的, hottest, latest), counts (3篇, 5 papers), "
                        "demonstratives (那, those), and filler words. Topic must be the pure research subject only. "
                        "Example: input='常温超导最火的那 3 篇' → topic='常温超导'. "
                        "TOOL ROUTING: arxiv_research_tool retrieves papers with per-paper citation_count — it CANNOT compute averages or sums. "
                        "When the user requests citation statistics (average, sum, sort by citations), use intent=citation_stats. "
                        "The system will then execute arxiv_research_tool first, followed by citation_stat_tool. "
                        "Never call citation_stat_tool before arxiv_research_tool returns results. "
                        "For any multi-paper statistical query, always plan retrieval before computation. "
                        "GUARDRAIL: When the user has not provided a clear research topic, paper title, author, "
                        "or category, use intent=clarify — never fabricate a topic or search with empty filters. "
                        "A topic like 'latest', 'recent', or 'best' without a concrete subject is invalid and must trigger clarification."
                    )
                ),
                HumanMessage(
                    content=(
                        f"recent_dialogue:\n{_format_recent_dialogue(state.get('messages', []), limit=4)}\n\n"
                        f"recalled_memory_context:\n{memory_context}\n\n"
                        f"{planning_instruction}\n\n"
                        f"user_request: {raw_user_input}"
                        f"{audit_feedback_context}"
                    )
                ),
            ]
        )

        planner_payload = _parse_json_object(getattr(response, "content", response))

        # ======================================
        # 强制锁死topic：仅中文问句才提纯覆盖；纯英文输入保持原样不变
        has_chinese_char = re.search(r'[\u4e00-\u9fff]', raw_user_input)
        if planner_payload.get("topic") and has_chinese_char:
            planner_payload["topic"] = extract_core_chinese_subject(raw_user_input)
        # =======================================

        normalized_request = build_structured_request(raw_user_input, planner_payload)

        if (
            normalized_request.get("intent") == "clarify"
            and _is_summary_followup_request(raw_user_input)
        ) or (
            normalized_request.get("intent") == "summarize_paper"
            and not normalized_request.get("paper_title")
        ):
            inferred_paper_title = _resolve_followup_paper_title(state, raw_user_input)
            if inferred_paper_title:
                normalized_request = {
                    "intent": "summarize_paper",
                    "question": None,
                    "topic": None,
                    "paper_title": inferred_paper_title,
                    "author": None,
                    "year": None,
                    "category": None,
                    "citation_count": 0,
                    "count": None,
                    "sort_mode": None,
                    "rewrite_limit": 0,
                    "min_relevance_score": None,
                    "ignored_intents": normalized_request.get("ignored_intents", []),
                    "risk_flags": normalized_request.get("risk_flags", []) + ["summary_target_inferred_from_context"],
                    "final_mode": "synthesize",
                    "response": None,
                }
        approved_plan = build_execution_plan(normalized_request)
        return {
            "normalized_request": normalized_request,
            "approved_plan": approved_plan,
            "risk_flags": normalized_request.get("risk_flags", []),
            "final_mode": normalized_request.get("final_mode", "synthesize"),
            "metrics": {"01_Planner_Logic": round(time.time() - start, 2)},
            "cleaned_topic": normalized_request.get("topic")
        }

    def executor_node(state: AgentState):
        start = time.time()
        current_step = state["approved_plan"][0]

        print(f"[DEBUG] 原始 args: {current_step['args']}")

        if current_step["step_type"] == "respond":
            return {
                "messages": [AIMessage(content=current_step["content"])],
                "metrics": {"02_Executor_Thought": round(time.time() - start, 2)},
            }

        # ===== Parameter validation: abort if no actionable search criteria =====
        _SEARCH_TOOLS = {"arxiv_research_tool", "query_research_db"}
        tool_name = current_step["tool_name"]
        if tool_name in _SEARCH_TOOLS:
            topic = str(current_step.get("args", {}).get("topic") or "").strip()
            paper_title = str(current_step.get("args", {}).get("paper_title") or "").strip()
            author = str(current_step.get("args", {}).get("author") or "").strip()
            category = str(current_step.get("args", {}).get("category") or "").strip()
            question = str(current_step.get("args", {}).get("question") or "").strip()

            # All search criteria are empty / whitespace → abort
            if not any([topic, paper_title, author, category]):
                return {
                    "messages": [AIMessage(content=(
                        "I cannot execute a retrieval call because no research topic, paper title, author, "
                        "or category was specified. Please provide a clear research topic (e.g., a specific "
                        "method, domain, or paper title) so I can search effectively."
                    ))],
                    "final_mode": "passthrough",
                    "metrics": {"02_Executor_Thought": round(time.time() - start, 2)},
                }

            # Topic is semantically vague: too short / generic stopwords only
            _VAGUE_STOPWORDS = {
                "latest", "recent", "new", "newest", "best", "top", "good",
                "最新", "最近", "最好", "最火", "顶尖", "热门",
                "research", "paper", "papers", "study", "studies",
            }
            topic_tokens = set(topic.lower().split())
            if topic and topic_tokens.issubset(_VAGUE_STOPWORDS):
                return {
                    "messages": [AIMessage(content=(
                        f"The topic '{topic}' is too vague for a meaningful literature search. "
                        "Please specify a concrete research area, method, or subject "
                        "(e.g., 'retrieval-augmented generation', 'graph neural networks', 'reinforcement learning')."
                    ))],
                    "final_mode": "passthrough",
                    "metrics": {"02_Executor_Thought": round(time.time() - start, 2)},
                }
        # ===== End parameter validation =====

        # Per-tool allowed args whitelist
        TOOL_ARG_WHITELIST: dict[str, set[str]] = {
            "arxiv_research_tool": {
                "question", "topic", "paper_title", "author", "year",
                "category", "category_strict", "citation_count", "count",
                "sort_mode", "rewrite_limit", "min_relevance_score"
            },
            "citation_stat_tool": {
                "search_results", "operation"
            },
            "query_research_db": {
                "question"
            },
            "summarize_paper_tool": {
                "paper_title"
            },
        }

        allowed = TOOL_ARG_WHITELIST.get(tool_name, set())

        # For citation_stat_tool, inject the previous tool's output as search_results
        if tool_name == "citation_stat_tool":
            prev_output = _latest_tool_output(state.get("messages", []))
            if prev_output and "search_results" not in current_step["args"]:
                current_step["args"]["search_results"] = prev_output

        # Filter args to allowed set for this specific tool
        filtered_args = {k: v for k, v in current_step["args"].items() if k in allowed}

        print(f"[DEBUG] 过滤后 args: {filtered_args}")

        # 从状态里拿到planner已经清洗完毕的topic
        cleaned_topic = state.get("cleaned_topic")

        # 只在topic不为空时强制覆盖
        if cleaned_topic:
            filtered_args["topic"] = cleaned_topic

        raw_text = state["messages"][-1].content


        def extract_core_chinese_subject(query: str) -> str:
            # 先删除所有干扰语句、量词、修饰词
            noise = r"\d+|一次性|下载|篇|帮我|搜索|查找|关于|的论文|的文章|最近|最新|最火"
            text = re.sub(noise, "", query)
            # 只保留连续中文汉字
            all_chinese = re.findall(r'[\u4e00-\u9fff]+', text)
            # 拼接纯主题名词
            raw_topic = "".join(all_chinese).strip()
            return raw_topic

        # 拿到用户原始提问
        raw_question = state["messages"][-1].content
        # 仅中文问句执行覆盖
        if re.search(r'[\u4e00-\u9fff]', raw_question):
            filtered_args["topic"] = extract_core_chinese_subject(raw_question)

        tool_call = {
            "name": current_step["tool_name"],
            "args": filtered_args,
            "id": f"call_{uuid4().hex[:8]}",
            "type": "tool_call",
        }
        return {
            "messages": [AIMessage(content=current_step["goal"], tool_calls=[tool_call])],
            "metrics": {"02_Executor_Thought": round(time.time() - start, 2)},
        }

    def timed_tool_node(state: AgentState):
        start = time.time()
        tool_output = ToolNode(tools).invoke(state)
        return {**tool_output, "metrics": {"03_Tools_IO": round(time.time() - start, 2)}}

    # ========== Observer审计纠偏闭环：observer_node ==========
    # 该节点在每次工具调用返回后执行，承担两个职责：
    # 1. 记录已完成步骤的摘要（past_steps），供 synthesizer 汇总使用。
    # 2. 当计划中所有步骤执行完毕后，调用 audit_and_correct 对检索结果
    #    进行质量审计，判定是否存在 missing_info / view_conflict / outdated 缺陷。
    #    若审计发现缺陷且未超过 max_replan_times，则触发闭环 → 回到 planner 重规划；
    #    否则进入 synthesizer 生成最终回答。
    def observer_node(state: AgentState):
        start = time.time()
        current_step = state["approved_plan"][0] if state.get("approved_plan") else None
        remaining_plan = state["approved_plan"][1:] if state.get("approved_plan") else []
        last_message = state["messages"][-1]

        # --- 记录已完成的步骤结果（保留原有逻辑） ---
        if isinstance(last_message, ToolMessage):
            tool_name = current_step["tool_name"] if current_step else "tool"
            snapshot = _truncate_text(getattr(last_message, "content", ""), 1500)
            past_steps = [f"{tool_name}: {snapshot}"]
        elif current_step and current_step["step_type"] == "respond":
            past_steps = [_truncate_text(getattr(last_message, "content", ""), 800)]
        else:
            past_steps = []

        # 如果计划还有剩余步骤，继续交给 executor 执行，暂不触发审计
        if remaining_plan:
            return {
                "past_steps": past_steps,
                "approved_plan": remaining_plan,
                "metrics": {"03_Observer": round(time.time() - start, 2)},
            }

        # --- 计划全部执行完毕，启动 Observer 审计纠偏闭环 ---
        # 从状态中提取审计所需的上下文信息
        raw_user_query: str = _get_latest_user_input(state)
        normalized_request: dict[str, Any] = state.get("normalized_request", {})
        planner_search_keywords: List[str] = (
            [normalized_request.get("topic", "")]
            if normalized_request.get("topic") else []
        )

        # 从最近的工具输出中解析检索到的论文列表
        tool_output: str = _latest_tool_output(state.get("messages", []))
        retrieved_papers: List[dict] = _parse_search_results(tool_output)

        # 论文摘要结果（从工具输出提取标题作为摘要占位）
        paper_summaries: List[dict] = [
            {"title": p.get("title", ""), "summary": p.get("summary_hint", "")}
            for p in retrieved_papers
        ]

        # 获取当前重规划轮次计数，检查是否超过上限
        current_replan_round: int = state.get("current_replan_round", 0)
        max_replan_times: int = state.get("max_replan_times", 3)

        # 构建审计状态字典，调用 observer_audit 模块的 audit_and_correct
        audit_state: dict[str, Any] = {
            "raw_user_query": raw_user_query,
            "planner_search_keywords": planner_search_keywords,
            "retrieved_papers": retrieved_papers,
            "paper_summaries": paper_summaries,
            "standard_task_type": normalized_request.get("intent", "search_papers"),
            "audit_logs": state.get("audit_logs", []),
        }

        audit_result: dict[str, Any] = audit_and_correct(audit_state)

        # 检查重规划次数限制，防止死循环
        need_replan: bool = audit_result.get("need_replan", False)
        if need_replan and current_replan_round >= max_replan_times:
            need_replan = False  # 达到上限，强制终止闭环

        # 重规划轮次自增（仅在确实触发重规划时）
        next_replan_round: int = current_replan_round + 1 if need_replan else current_replan_round

        return {
            "past_steps": past_steps,
            "approved_plan": remaining_plan,
            "audit_logs": audit_result.get("audit_logs", []),
            "current_defect": audit_result.get("current_defect"),
            "suggest_keywords": audit_result.get("suggest_keywords", []),
            "need_replan": need_replan,
            "current_replan_round": next_replan_round,
            # ========== 三级文献分级精读：传递打分结果 ==========
            "rated_papers": audit_result.get("rated_papers", []),
            "metrics": {"03_Observer_Audit": round(time.time() - start, 2)},
        }

    # ========== 三级文献分级精读：executor_read 分层精读节点 ==========
    # 当 Observer 审计通过（无缺陷）时，根据 rated_papers 中每篇论文的
    # read_level（coarse/medium/deep）使用不同 Prompt 策略进行分层精读：
    #   - coarse（粗读）: 1-2句核心概述
    #   - medium（中度）: 结构化摘要（目的、方法、关键发现）
    #   - deep（深度）: 综合分析（背景、方法、实验、结论、局限性、未来方向）
    # 精读结果注入 past_steps，供 synthesizer 整合为最终回答。
    def executor_read_node(state: AgentState):
        start = time.time()
        rated_papers: List[dict] = state.get("rated_papers", []) or []

        if not rated_papers:
            # 无分级论文时直接透传，不阻塞流程
            return {
                "past_steps": ["[executor_read] 无待精读论文"],
                "metrics": {"03b_Executor_Read": round(time.time() - start, 2)},
            }

        # ---- 三级分层阅读 Prompt 模板 ----
        COARSE_PROMPT: str = (
            "你是一位科研文献速读助手。请用1-2句话极其精炼地概括以下论文的核心内容，"
            "不需要详细展开。\n论文标题: {title}\n论文元数据: {metadata}\n"
            "请仅输出中文概述:"
        )
        MEDIUM_PROMPT: str = (
            "你是一位科研文献结构化阅读助手。请按以下格式总结论文:\n"
            "1. 研究目的\n2. 方法\n3. 关键发现\n"
            "论文标题: {title}\n论文元数据: {metadata}\n"
            "请仅输出结构化摘要:"
        )
        DEEP_PROMPT: str = (
            "你是一位资深科研评审专家。请对该论文进行深度综合分析，按以下维度输出:\n"
            "1. 研究背景与动机\n2. 核心方法/技术创新\n3. 实验设计与关键结果\n"
            "4. 主要结论与贡献\n5. 局限性\n6. 未来研究方向\n"
            "论文标题: {title}\n论文元数据: {metadata}\n"
            "请输出深度分析报告:"
        )

        read_results: list[str] = []
        for paper in rated_papers:
            title: str = str(paper.get("title", "未知标题"))
            read_level: str = str(paper.get("read_level", "coarse")).lower()
            relevance_score: float = float(paper.get("relevance_score", 0.0))
            read_reason: str = str(paper.get("read_reason", ""))

            # 构建论文元数据字符串
            metadata_parts: list[str] = []
            for key in ("source", "year", "score"):
                val = paper.get(key)
                if val:
                    metadata_parts.append(f"{key}={val}")
            metadata_str: str = ", ".join(metadata_parts) if metadata_parts else "无额外元数据"

            # 根据阅读等级选择对应 Prompt
            level_label: str
            if read_level == "deep":
                prompt_template = DEEP_PROMPT
                level_label = "深度"
            elif read_level == "medium":
                prompt_template = MEDIUM_PROMPT
                level_label = "中度"
            else:
                prompt_template = COARSE_PROMPT
                level_label = "粗读"

            filled_prompt: str = prompt_template.format(title=title, metadata=metadata_str)

            try:
                read_response = planner_llm.invoke([
                    SystemMessage(content="You are a precise academic literature reader. Output in Chinese."),
                    HumanMessage(content=filled_prompt),
                ])
                summary_text: str = _stringify_content(getattr(read_response, "content", read_response))
            except Exception:
                summary_text = f"[读取失败] {title}"

            # 标注阅读等级和相关度分数
            tagged_summary: str = (
                f"[{level_label}精读 | 相关度={relevance_score:.1f}/10 | {read_reason}]\n"
                f"**{title}**\n{summary_text}"
            )
            read_results.append(tagged_summary)

        return {
            "past_steps": read_results,
            "metrics": {"03b_Executor_Read": round(time.time() - start, 2)},
        }

    # ========== Observer审计后路由函数 ==========
    # 决定 observer_node 执行后的下一个节点：
    # - need_replan=True 且未超限 → 回到 planner 重新生成检索计划（闭环）
    # - need_replan=False 且有 rated_papers → 进入 executor_read 分层精读
    # - 计划还有剩余步骤 → 回到 executor 继续执行
    # - 计划执行完毕且无需重规划无精读 → 进入 synthesizer 生成最终答案
    def route_after_observer(state: AgentState) -> str:
        # Observer审计纠偏闭环：需要重规划则回到planner
        if state.get("need_replan", False):
            return "planner"
        # 计划中还有未执行的步骤，继续交给executor
        if state.get("approved_plan"):
            return "executor"
        # ========== 三级文献分级精读路由分支 ==========
        # 无缺陷但有打分论文 → 进入分层精读节点
        rated: list = state.get("rated_papers", []) or []
        if rated and not state.get("need_replan", False):
            return "executor_read"
        # 所有步骤完成且无需重规划，进入综合节点
        return "synthesizer"

    def synthesizer_node(state: AgentState):
        start = time.time()
        if state.get("final_mode") == "passthrough":
            # Echo the last assistant answer so it reaches the user via SSE stream
            passthrough_answer = _get_latest_assistant_answer(state)
            if passthrough_answer:
                return {
                    "messages": [AIMessage(content=passthrough_answer)],
                    "metrics": {"04_Synthesizer_Final": round(time.time() - start, 2)},
                }
            return {"metrics": {"04_Synthesizer_Final": round(time.time() - start, 2)}}

        all_facts = "\n".join(state.get("past_steps", [])[-6:])
        normalized_request = state.get("normalized_request", {})
        raw_tool_output = _latest_tool_output(state.get("messages", []))
        grounded_search_answer = ""
        if normalized_request.get("intent") == "search_papers" and not _search_request_needs_summary(
            normalized_request.get("question") or _get_latest_user_input(state)
        ):
            grounded_search_answer = _build_grounded_search_answer(normalized_request, raw_tool_output or all_facts)
        grounded_search_summary_answer = ""
        if normalized_request.get("intent") == "search_papers" and _search_request_needs_summary(
            normalized_request.get("question") or _get_latest_user_input(state)
        ):
            grounded_search_summary_answer = _build_grounded_search_summary_answer(
                normalized_request,
                raw_tool_output or all_facts,
                _get_latest_user_input(state),
            )
        # For citation_stats intent, pass through the stat tool output directly
        if normalized_request.get("intent") == "citation_stats":
            stat_output = _latest_tool_output(state.get("messages", []))
            if stat_output:
                return {
                    "messages": [AIMessage(content=stat_output)],
                    "metrics": {"04_Synthesizer_Final": round(time.time() - start, 2)},
                }
        if grounded_search_answer:
            return {
                "messages": [AIMessage(content=grounded_search_answer)],
                "metrics": {"04_Synthesizer_Final": round(time.time() - start, 2)},
            }
        if grounded_search_summary_answer:
            return {
                "messages": [AIMessage(content=grounded_search_summary_answer)],
                "metrics": {"04_Synthesizer_Final": round(time.time() - start, 2)},
            }
        risk_summary = ", ".join(state.get("risk_flags", [])) or "none"
        history_context = _format_recent_dialogue(state.get("messages", []), limit=6)
        memory_context = _format_recalled_memories(state.get("recalled_memories", []))

        response = planner_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a careful research assistant. "
                        "Answer using only the structured_request, explicit conversation context, recalled memory context, "
                        "and tool_facts. Treat tool_facts as the highest-priority evidence. "
                        "Recalled memory is secondary and may help with continuity, but it must not override direct tool evidence. "
                        "Do not invent papers, authors, dates, datasets, metrics, experimental results, or conclusions "
                        "that are not supported by the evidence. "
                        "If evidence is incomplete or missing, say so explicitly. "
                        "Distinguish clearly between supported facts and your own suggestions or inferences. "
                        "If you make a suggestion, label it as a suggestion."
                    )
                ),
                HumanMessage(
                    content=(
                        "Write a grounded answer for the user.\n"
                        "Requirements:\n"
                        "- First answer the user's actual request directly.\n"
                        "- Use only supported information from tool_facts and explicit conversation context.\n"
                        "- Use recalled_memory_context only for continuity, preferences, or prior confirmed decisions.\n"
                        "- Match the user's language when the latest user input clearly indicates one.\n"
                        "- If the user asked to find papers and summarize them, briefly summarize the top retrieved papers instead of only listing titles.\n"
                        "- When summarizing found papers, prefer a compact structure: title, what it is about, and why it is relevant.\n"
                        "- If the tools did not provide enough evidence, say what is known and what remains uncertain.\n"
                        "- Never fabricate citations or missing details.\n"
                        "- Prefer concise, factual writing over generic filler.\n"
                        "- If helpful, end with one short next-step suggestion.\n\n"
                        f"conversation_context:\n{history_context}\n\n"
                        f"recalled_memory_context:\n{memory_context}\n\n"
                        f"structured_request:\n{json.dumps(normalized_request, ensure_ascii=False)}\n\n"
                        f"risk_flags:\n{risk_summary}\n\n"
                        f"tool_facts:\n{all_facts}\n\n"
                        f"latest_user_input:\n{_get_latest_user_input(state)}"
                    )
                ),
            ]
        )
        content = _sanitize_answer_text(_stringify_content(getattr(response, "content", response)))
        return {
            "messages": [AIMessage(content=content)],
            "metrics": {"04_Synthesizer_Final": round(time.time() - start, 2)},
        }

    def memory_writer_node(state: AgentState):
        start = time.time()
        latest_user_input = _get_latest_user_input(state)
        assistant_answer = _get_latest_assistant_answer(state)
        if not latest_user_input or not assistant_answer:
            return {"metrics": {"05_Memory_Write": round(time.time() - start, 2)}}

        user_id = state.get("user_id")
        thread_id = state.get("thread_id") or "default"
        tool_outputs = _collect_tool_outputs(state)
        conversation_excerpt = _collect_conversation_excerpt(state)
        store_turn_memory(
            user_id=user_id,
            thread_id=thread_id,
            user_input=latest_user_input,
            assistant_answer=assistant_answer,
            normalized_request=state.get("normalized_request", {}),
            tool_outputs=tool_outputs,
            conversation_excerpt=conversation_excerpt,
        )
        return {"metrics": {"05_Memory_Write": round(time.time() - start, 2)}}

    # ========== LangGraph 工作流构建：Observer审计纠偏闭环 + 三级文献分级精读 ==========
    # 流程: memory_router → planner → executor(仅拉取元数据) → tools → observer_audit
    #       observer_audit ──need_replan→ planner（闭环重规划）
    #       observer_audit ──remaining→ executor（继续执行计划步骤）
    #       observer_audit ──无缺陷+有rated_papers→ executor_read（分层精读）
    #       executor_read → synthesizer → memory_writer → END
    workflow = StateGraph(AgentState)
    workflow.add_node("memory_router", memory_router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("tools", timed_tool_node)
    workflow.add_node("observer", observer_node)
    workflow.add_node("executor_read", executor_read_node)  # 三级文献分级精读节点
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("memory_writer", memory_writer_node)

    workflow.add_edge(START, "memory_router")
    workflow.add_edge("memory_router", "planner")
    workflow.add_conditional_edges("planner", lambda state: "executor" if state["approved_plan"] else "synthesizer")
    workflow.add_conditional_edges("executor", lambda state: "tools" if state["messages"][-1].tool_calls else "observer")
    workflow.add_edge("tools", "observer")
    # Observer审计纠偏闭环 + 三级文献分级精读路由：
    #   - need_replan=True → planner（闭环：重新生成检索关键词和计划）
    #   - approved_plan非空 → executor（继续执行剩余步骤）
    #   - 无缺陷且rated_papers非空 → executor_read（分层精读）
    #   - 否则 → synthesizer（进入最终答案生成）
    workflow.add_conditional_edges("observer", route_after_observer)
    workflow.add_edge("executor_read", "synthesizer")  # 精读完成后进入综合
    workflow.add_edge("synthesizer", "memory_writer")
    workflow.add_edge("memory_writer", END)

    return workflow.compile(checkpointer=build_agent_checkpointer(_CHECKPOINTER_STACK))


def _parse_json_object(raw_content: Any) -> dict:
    text = str(raw_content).strip()
    fenced = text.replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(fenced)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        start = fenced.find("{")
        end = fenced.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(fenced[start : end + 1])
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def _get_latest_user_input(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        content = getattr(message, "content", None)
        if content and message.__class__.__name__ == "HumanMessage":
            return str(content)
    return state["input"]


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    return str(content)


def _truncate_text(text: Any, limit: int = 1200) -> str:
    cleaned = " ".join(_stringify_content(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + " ..."


def _looks_like_memory_reference(text: str) -> bool:
    lowered = str(text or "").lower()
    hints = (
        "continue",
        "same",
        "before",
        "previous",
        "that paper",
        "those papers",
        "earlier",
        "刚才",
        "刚刚",
        "之前",
        "上次",
        "继续",
        "那个",
        "上述",
        "这篇",
        "上一篇",
        "上一条",
        "第一篇",
        "第二篇",
        "第三篇",
    )
    return any(hint in lowered or hint in text for hint in hints)


def _get_latest_assistant_answer(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.__class__.__name__ == "AIMessage" and not getattr(message, "tool_calls", None):
            content = _stringify_content(getattr(message, "content", ""))
            if content:
                return content
    return ""


def _collect_tool_outputs(state: AgentState) -> list[str]:
    outputs: list[str] = []
    for message in state.get("messages", []):
        if message.__class__.__name__ == "ToolMessage":
            content = _truncate_text(getattr(message, "content", ""), limit=2200)
            if content:
                outputs.append(content)
    return outputs[-3:]


def _collect_conversation_excerpt(state: AgentState) -> list[dict[str, str]]:
    excerpt: list[dict[str, str]] = []
    for message in state.get("messages", []):
        role_name = message.__class__.__name__
        if role_name == "HumanMessage":
            excerpt.append({"role": "user", "content": _truncate_text(getattr(message, "content", ""), limit=900)})
        elif role_name == "AIMessage" and not getattr(message, "tool_calls", None):
            excerpt.append({"role": "assistant", "content": _truncate_text(getattr(message, "content", ""), limit=900)})
    return excerpt[-6:]
