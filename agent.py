import atexit
import contextlib
import json
import operator
import os
import re
import time
from typing import Annotated, Any, List
from uuid import uuid4

from typing_extensions import TypedDict

# ★ 全局 LLM 推理超时限制：防止大模型卡死无限等待，避免线上接口超时崩溃
LLM_TIMEOUT = 15  # ★ 全局LLM推理超时15s，防止大模型卡死导致接口超时

from conversation_memory import (
    load_recent_turn_memories,
    load_turn_memories,
    search_turn_memories,
    store_turn_memory,
)
from research_harness import build_execution_plan, build_structured_request, _extract_topic
from storage import build_agent_checkpointer

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


# ★★★ 全局复用：前置轻量黑名单过滤函数 ★★★
# 匹配邮件、聊天等无关动作关键词，提前剥离无关文本再送入 planner，减少LLM冗余token
_PRE_FILTER_BLACKLIST = re.compile(
    r'(?:帮我|给我|替我|麻烦|请)(?:发(?:个|一封|一下)?(?:邮件|email|消息|微信|短信)|'
    r'打(?:个)?(?:电话|语音)|写(?:个|一份)?(?:报告|周报|日报|总结|PPT)|'
    r'做(?:个|一份)?(?:PPT|表格|文档|excel|word))|'
    r'(?:先|首先|再|然后|顺便|另外|此外|还|也)(?:帮我|给我)?(?:发(?:邮件|消息|微信)|打(?:电话)|写(?:报告)|做(?:PPT))|'
    r'发(?:个|一封|一下)?(?:邮件|email|消息|微信|短信)给',
    re.IGNORECASE
)

def pre_filter_irrelevant_text(user_input: str) -> str:
    """前置过滤：剥离邮件/聊天等无关动作文本，保留科研检索核心内容。"""
    return _PRE_FILTER_BLACKLIST.sub('', user_input).strip()


# ★★★ 全局复用：topic 深度清洗工具函数 ★★★
def clean_core_topic(raw_topic: str) -> str:
    """清洗 topic 字段冗余修饰文本，仅保留纯研究核心关键词。

    自动过滤：角色扮演话术、地点场景干扰、无关前置动作、反问诱导语气、多余数量描述；
    标准化特殊符号、分离粘连中英文、清理&/括号噪声。
    仅用于 topic 字段，author/doi/paper_title/year 不调用此函数。
    """
    if not raw_topic or not raw_topic.strip():
        return ""
    t = raw_topic.strip()

    # --- 第一层：删除四类干扰文本 ---
    # 1. 角色扮演话术：假设你是XX专家、作为诺贝尔奖得主...
    t = re.sub(r'(?:假设|如果|假如)(?:你|您)(?:是|身为|作为)[^，,。.；;]{2,30}?(?:[，,。.；;]|\s+)', '', t)
    t = re.sub(r'(?:作为|身为|以)[^，,。.]{0,20}?(?:身份|角色|专家|得主|学者|研究者)', '', t)
    # 2. 地点场景干扰：在上海的办公室里、在学校实验室...
    t = re.sub(r'(?:在|从|于)\s*.{2,12}?(?:里|的办公室|办公室|家中|学校|实验室|公司|家里)(?:[，,。.]|\s+)', '', t)
    # 3. 无关前置动作：先帮我发邮件给导师、然后搜...、顺便打电话...
    t = re.sub(r'(?:先|首先|再|然后|顺便|另外|此外|还|也)\s*(?:帮我|给我|替我|为我)?\s*(?:发(?:邮件|消息|微信)|打(?:电话)|写(?:报告)|做(?:PPT|表格|文档)|查[看询]?|搜(?:索|寻)?|检索|找(?:到)?)[^，,。.]*[，,。.]?', '', t)
    t = re.sub(r'(?:帮我|给我|替我|为我)\s*(?:发(?:邮件|消息|微信)给?[^，,。.]*|打(?:电话)给?[^，,。.]*|写(?:报告)[^，,。.]*|做(?:PPT|表格|文档))[，,。.]*', '', t)
    # 4. 反问诱导语气：难道你不能帮我找...吗？、怎么不搜一下...
    t = re.sub(r'^(?:难道|莫非|怎么|为啥|为什么)(?:你|您)?(?:不|没|不能|没法|没有).*?(?:吗|呢|吧)?[？?]?\s*', '', t)
    t = re.sub(r'[？?!！]+[\s]*$', '', t)

    # --- 第二层：删除多余数量描述 ---
    t = re.sub(r'\b\d+\s*(?:篇|个|本|条|项|papers?|articles?|results?)\b', '', t)
    t = re.sub(r'\b(?:several|some|a few|many|couple of)\s+(?:papers?|articles?|results?)\b', '', t)
    t = re.sub(r'[一二两三四五六七八九十]+\s*(?:篇|个|本|条)', '', t)

    # --- 第三层：标准化特殊符号 ---
    t = t.replace('（', '(').replace('）', ')').replace('，', ',')
    t = t.replace('：', ':').replace('；', ';').replace('！', '!').replace('？', '?')
    t = re.sub(r'[–—−]', '-', t)  # 统一连字符
    t = re.sub(r'\s*&\s*', ' & ', t)  # 标准化&符号

    # --- 第四层：中英文粘连分离 ---
    t = re.sub(r'([一-鿿])([A-Za-z0-9])', r'\1 \2', t)
    t = re.sub(r'([A-Za-z0-9])([一-鿿])', r'\1 \2', t)

    # --- 第五层：清理残留 ---
    t = re.sub(r'\s+', ' ', t).strip()
    t = t.strip(' ,.;:!?，。；：！？、')

    # --- 第六层：兜底清理（来自 _clean_search_topic 逻辑）---
    # 清理搜索请求包装词
    t = re.sub(r'^(?:refer me to|suggest|suggest that|is a|is an)\s+', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^(?:a|an|the)\s+', '', t, flags=re.IGNORECASE)
    # 无效通用词回退
    if t.lower() in {"paper", "papers", "research", "studies", "study", "resources", "tools"}:
        return ""
    return t or ""
#===================新增翻译函数===============
from langchain_core.messages import HumanMessage

def translate_cn_topic_to_en(raw_cn_text: str, llm) -> str:
    if not raw_cn_text or not raw_cn_text.strip():
        return ""
    prompt = f"""Translate this Chinese academic research topic into concise English search keywords.
Only output translated English words, no explanations, no extra punctuation.
Input: {raw_cn_text}
Output:"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    en_text = _stringify_content(resp.content).strip()
    return en_text

#============================================

def create_research_agent():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    from tools import arxiv_research_tool, citation_stat_tool, query_research_db, summarize_paper_tool

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    planner_llm = ChatOpenAI(
        model="qwen-turbo",
        temperature=0.1,
        openai_api_key=api_key,
        base_url=base_url,
        request_timeout=LLM_TIMEOUT,  # ★ 全局超时40s，防止大模型卡死
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


#=====新增通用翻译工具函数=================
    def translate_cn_topic_to_en(raw_cn_text: str, llm: ChatOpenAI) -> str:
        """中文检索主题 → 精简英文检索关键词，仅输出纯英文名词短语，无多余解释"""
        if not raw_cn_text or not raw_cn_text.strip():
            return ""
        prompt = f"""Translate this Chinese research topic into concise English search keywords, output ONLY the translated text, no extra words, no explanation.
    Input: {raw_cn_text}
    Output only English keywords:"""
        resp = llm.invoke([HumanMessage(content=prompt)])
        en_text = _stringify_content(resp.content).strip()
        return en_text
# =========================

    def planner_node(state: AgentState):
        start = time.time()
        raw_user_input = _get_latest_user_input(state)
        # ★ 前置黑名单过滤：剥离邮件/聊天等无关动作文本，减少LLM冗余token
        raw_user_input = pre_filter_irrelevant_text(raw_user_input)
        memory_context = _format_recalled_memories(state.get("recalled_memories", []))
        try:
            response = planner_llm.invoke(
            [
                SystemMessage(
                    content=(
                        # ── 合并精简版 System Prompt（去重 planning_instruction，减少 token 消耗）──
                        "You are a research request normalizer. Output one strict JSON object only, no markdown, no prose.\n"
                        "Use conversation history / recalled memory only to resolve cross-turn references.\n"
                        "\n"
                        "=== LANGUAGE RULES (top priority) ===\n"
                        "- English input → topic must be English keywords. NEVER translate to Chinese.\n"
                        "- Chinese input → topic must be Chinese keywords. NEVER translate to English.\n"
                        "- Never mix Chinese+English in one topic field.\n"
                        "- Topic = pure research core terms only. Strip ALL noise: superlatives(最火/hottest/latest), counts(3篇/5 papers), demonstratives(那/those), filler(帮我/检索/find/search for).\n"
                        "- Examples: \"常温超导最火的3篇\"→topic=\"常温超导\" | \"latest 5 papers on diffusion models\"→topic=\"diffusion models\" | \"找MoE文献不超过2篇\"→topic=\"混合专家模型(MoE)\"\n"
                        "- Keep rare technical terms intact: MoE, C++, Sora, specific methods/metrics/benchmarks.\n"
                        "\n"
                        "=== FIELD EXTRACTION RULES ===\n"
                        "- topic: pure keywords stripped of author/year/DOI/title/time. null if no clear subject.\n"
                        "- paper_title: only if user gives exact quoted/bookmarked title, else null.\n"
                        "- author: only if user explicitly names one, else null. Never guess.\n"
                        "- year: only explicit year (2023/2024). Relative time(最近/去年)→time_range, year=null.\n"
                        "- time_range: e.g. 近三年→2023-2025. null if not specified.\n"
                        "- doi: only if user gives explicit DOI, else null.\n"
                        "- category: only if user names arXiv category or canonical domain(NLP/CV/ML), else null.\n"
                        "- citation_count: min citation threshold, null if not specified.\n"
                        "- Never invent paper titles, authors, years, DOIs, or categories.\n"
                        "\n"
                        "=== INTENTS: search_papers | query_local_db | summarize_paper | citation_stats | clarify ===\n"
                        "- query_local_db: only when user says local/indexed/vector-database/已下载.\n"
                        "- summarize_paper: only when user gives paper title + summary/overview/总结 intent.\n"
                        "- citation_stats: when user asks avg/sum/sort by citations. Always retrieval first, then computation.\n"
                        "- clarify: no clear topic/title/author/category → clarify. Vague words alone(latest/recent/最新/最近) without concrete subject → clarify.\n"
                        "- Mixed request: keep main research intent, put unrelated actions into ignored_intents.\n"
                        "\n"
                        "=== OUTPUT JSON SCHEMA ===\n"
                        '{"intent":"","topic":null,"paper_title":null,"author":null,"year":null,"doi":null,"time_range":null,"category":null,"citation_count":null,"count":null,"question":"","clarification_question":null,"ignored_intents":[]}'
                    )
                ),
                HumanMessage(
                    content=(
                        f"recent_dialogue:\n{_format_recent_dialogue(state.get('messages', []), limit=4)}\n\n"
                        f"recalled_memory_context:\n{memory_context}\n\n"
                        f"user_request: {raw_user_input}"
                    )
                ),
            ]
        )
        except Exception:
            # LLM 调用失败时兜底：返回 clarify，让前端正常展示而非报错
            return {
                "normalized_request": build_structured_request(raw_user_input, {}),
                "approved_plan": [{"step_type": "respond", "content": "抱歉，请求处理超时，请重试。"}],
                "risk_flags": ["planner_llm_failed"],
                "final_mode": "passthrough",
                "metrics": {"01_Planner_Logic": round(time.time() - start, 2)},
            }

        planner_payload = _parse_json_object(getattr(response, "content", response))

        # ★ DEBUG：完整打印所有独立抽取字段，方便调试核对抽取结果
        print(f"[DEBUG] planner_payload → intent={planner_payload.get('intent')} | "
              f"topic={planner_payload.get('topic')!r} | "
              f"paper_title={planner_payload.get('paper_title')!r} | "
              f"author={planner_payload.get('author')!r} | "
              f"year={planner_payload.get('year')!r} | "
              f"doi={planner_payload.get('doi')!r} | "
              f"time_range={planner_payload.get('time_range')!r} | "
              f"category={planner_payload.get('category')!r} | "
              f"citation_count={planner_payload.get('citation_count')!r} | "
              f"count={planner_payload.get('count')!r}")

        # ======================================
        # ★ 新增：对所有 topic 执行全局文本清洗（四类干扰过滤+符号标准化）
        if planner_payload.get("topic"):
            planner_payload["topic"] = clean_core_topic(planner_payload["topic"])  # ★ 全局清洗函数
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

    # ★★★ 新增：空有效主题前置拦截函数 ★★★
    # 在 arxiv 工具调用前校验检索参数，无有效 topic 时直接返回提示，不发起 arxiv 检索
    def _validate_search_topic(topic: str | None, paper_title: str | None, author: str | None, category: str | None) -> str | None:
        """前置拦截校验：识别无有效检索主题时返回提示消息；验证通过则返回 None。"""
        _topic = str(topic or "").strip()
        _paper_title = str(paper_title or "").strip()
        _author = str(author or "").strip()
        _category = str(category or "").strip()

        # 拦截1：所有检索条件均为空/空白
        if not any([_topic, _paper_title, _author, _category]):
            return (
                "I cannot execute a retrieval call because no research topic, paper title, author, "
                "or category was specified. Please provide a clear research topic (e.g., a specific "
                "method, domain, or paper title) so I can search effectively."
            )

        # 拦截2：topic 仅含模糊停用词，无实质检索意义
        _VAGUE_STOPWORDS = {
            "latest", "recent", "new", "newest", "best", "top", "good",
            "最新", "最近", "最好", "最火", "顶尖", "热门",
            "research", "paper", "papers", "study", "studies",
        }
        topic_tokens = set(_topic.lower().split())
        if _topic and topic_tokens.issubset(_VAGUE_STOPWORDS):
            return (
                f"The topic '{_topic}' is too vague for a meaningful literature search. "
                "Please specify a concrete research area, method, or subject "
                "(e.g., 'retrieval-augmented generation', 'graph neural networks', 'reinforcement learning')."
            )

        return None  # 校验通过

    def executor_node(state: AgentState):
        start = time.time()
        current_step = state["approved_plan"][0]

        print(f"[DEBUG] 原始 args: {current_step['args']}")

        if current_step["step_type"] == "respond":
            return {
                "messages": [AIMessage(content=current_step["content"])],
                "metrics": {"02_Executor_Thought": round(time.time() - start, 2)},
            }

        # ★★★ 修改：调用前置拦截函数，替代原有内联校验 ★★★
        _SEARCH_TOOLS = {"arxiv_research_tool", "query_research_db"}
        tool_name = current_step["tool_name"]
        if tool_name in _SEARCH_TOOLS:
            args = current_step.get("args", {})
            intercept_msg = _validate_search_topic(
                topic=args.get("topic"),
                paper_title=args.get("paper_title"),
                author=args.get("author"),
                category=args.get("category"),
            )
            if intercept_msg is not None:  # 拦截命中：尝试启发式回退提取topic
                raw_input = _get_latest_user_input(state)
                fallback_topic = _extract_topic(raw_input)
                if fallback_topic:
                    fallback_topic = clean_core_topic(fallback_topic)
                    recheck = _validate_search_topic(
                        topic=fallback_topic,
                        paper_title=args.get("paper_title"),
                        author=args.get("author"),
                        category=args.get("category"),
                    )
                    if recheck is None:
                        # 启发式回退成功，使用新topic继续检索
                        print(f"[FALLBACK] 启发式回退提取到有效topic: {fallback_topic!r}")
                        current_step["args"]["topic"] = fallback_topic
                        intercept_msg = None  # 清除拦截，继续执行检索
                if intercept_msg is not None:
                    return {
                        "messages": [AIMessage(content=intercept_msg)],
                        "final_mode": "passthrough",
                        "metrics": {"02_Executor_Thought": round(time.time() - start, 2)},
                    }

        # ★ P0修复：intent=clarify 或无有效 topic 时终止流程，禁止调用 arxiv_research_tool ★
        normalized_intent = state.get("normalized_request", {}).get("intent", "")
        if normalized_intent == "clarify" and tool_name in _SEARCH_TOOLS:
            return {
                "messages": [AIMessage(content=(
                    "I need more specific information to conduct a meaningful literature search. "
                    "Please provide a clear research topic, paper title, or author name."
                ))],
                "final_mode": "passthrough",
                "metrics": {"02_Executor_Thought": round(time.time() - start, 2)},
            }

        # Per-tool allowed args whitelist  ★ doi/time_range 加入白名单透传
        TOOL_ARG_WHITELIST: dict[str, set[str]] = {
            "arxiv_research_tool": {
                "question", "topic", "paper_title", "doi", "author", "year",
                "time_range", "category", "category_strict", "citation_count", "count",
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

        # ★★★ 核心打通修复：优先复用 planner 已清洗字段，禁止重新从 question 提取脏 topic ★★★
        cleaned_topic = state.get("cleaned_topic")
        if cleaned_topic:
            filtered_args["topic"] = cleaned_topic  # 使用 planner 清洗后的干净 topic
            # 同步透传 planner 产出的独立字段（doi/time_range/author/year/paper_title）
            normalized_req = state.get("normalized_request", {})
            for _key in ("doi", "time_range", "author", "year", "paper_title"):
                _val = normalized_req.get(_key)
                if _val and _key in allowed:
                    filtered_args[_key] = _val

        # ★ P0修复：count 上限截断，超过全局 MAX_SEARCH_LIMIT 自动截断 ★
        from research_harness import MAX_SEARCH_LIMIT
        _raw_count = filtered_args.get("count")
        if _raw_count is not None and int(_raw_count) > MAX_SEARCH_LIMIT:
            print(f"[INTERCEPT] count={_raw_count} > MAX_SEARCH_LIMIT={MAX_SEARCH_LIMIT}，已自动截断")
            filtered_args["count"] = MAX_SEARCH_LIMIT

        # ★ DEBUG：打印最终下游 args，确认 topic/author/year/doi 与 planner 输出一致 ★
        print(f"[DEBUG] executor 最终 args → topic={filtered_args.get('topic')!r} | "
              f"author={filtered_args.get('author')!r} | year={filtered_args.get('year')!r} | "
              f"doi={filtered_args.get('doi')!r} | time_range={filtered_args.get('time_range')!r} | "
              f"count={filtered_args.get('count')!r} | paper_title={filtered_args.get('paper_title')!r}")

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

    def observer_node(state: AgentState):
        current_step = state["approved_plan"][0] if state.get("approved_plan") else None
        remaining_plan = state["approved_plan"][1:] if state.get("approved_plan") else []
        last_message = state["messages"][-1]

        if isinstance(last_message, ToolMessage):
            tool_name = current_step["tool_name"] if current_step else "tool"
            snapshot = _truncate_text(getattr(last_message, "content", ""), 1500)
            past_steps = [f"{tool_name}: {snapshot}"]
        elif current_step and current_step["step_type"] == "respond":
            past_steps = [_truncate_text(getattr(last_message, "content", ""), 800)]
        else:
            past_steps = []

        return {"past_steps": past_steps, "approved_plan": remaining_plan}

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

    workflow = StateGraph(AgentState)
    workflow.add_node("memory_router", memory_router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("tools", timed_tool_node)
    workflow.add_node("observer", observer_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("memory_writer", memory_writer_node)

    workflow.add_edge(START, "memory_router")
    workflow.add_edge("memory_router", "planner")
    workflow.add_conditional_edges("planner", lambda state: "executor" if state["approved_plan"] else "synthesizer")
    workflow.add_conditional_edges("executor", lambda state: "tools" if state["messages"][-1].tool_calls else "observer")
    workflow.add_edge("tools", "observer")
    workflow.add_conditional_edges("observer", lambda state: "executor" if state["approved_plan"] else "synthesizer")
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
