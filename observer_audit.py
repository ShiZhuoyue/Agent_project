import json
from typing import TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ========== 三级文献分级精读：PaperItem 结构化类型 ==========
# 每篇论文经 Observer 审计后，由 LLM 输出相关度分数与阅读等级，
# 用于驱动 executor_read 分层精读节点选择对应的 Prompt 策略。
class PaperItem(TypedDict):
    title: str                # 论文标题
    relevance_score: float    # 相关度分数 0-10，分数越高越相关
    read_level: str           # 阅读等级: coarse(粗读) / medium(中度) / deep(深度)
    read_reason: str          # 分级理由：为什么给出该分数和等级

# 结构化审计日志格式
class AuditRecord(TypedDict):
    task_type: str
    original_query: str
    planner_keywords: List[str]
    retrieved_papers: List[dict]
    paper_summary_results: List[dict]
    defect_type: Optional[str]  # missing_info / view_conflict / outdated / None
    defect_desc: Optional[str]
    need_replan: bool
    suggest_new_keywords: List[str]
    # ========== 三级文献分级精读：新增字段 ==========
    # rated_papers: Observer 对每篇论文的相关度打分与阅读等级建议，
    # 写入 AgentState.rated_papers 供 executor_read 分层精读节点消费。
    rated_papers: List[PaperItem]

class ObserverState(TypedDict):
    audit_logs: List[AuditRecord]
    current_defect: Optional[str]
    suggest_keywords: List[str]
    # ========== 三级文献分级精读：新增字段 ==========
    rated_papers: List[PaperItem]  # 经 Observer 打分后的论文列表，供 executor_read 消费

# 初始化LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 审计校验Prompt：判断三类缺陷、生成修正建议
audit_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是科研文献观测审计器，严格检查当前检索结果是否满足用户调研需求，仅输出JSON。
校验三类缺陷：
1. missing_info：文献数量不足、覆盖领域不全，无法支撑完整综述
2. view_conflict：多篇论文核心结论互相矛盾，缺少调和综述文献
3. outdated：绝大多数文献年份过旧，缺少近3年顶会/期刊最新成果

=== 三级文献分级精读评分（必须对每篇论文输出） ===
对检索到的每篇论文，根据其与用户需求的匹配程度给出：
- relevance_score: 0-10 相关度分数
  · 9-10分：与用户需求高度匹配，核心参考文献
  · 7-8分：与用户需求相关，重要参考
  · 5-6分：部分相关，可作背景参考
  · 0-4分：弱相关或无关
- read_level: 阅读等级，基于 relevance_score 判定
  · "deep"  (relevance_score >= 8)：深度精读——需要完整分析背景、方法、实验、结论、局限性
  · "medium"(5 <= relevance_score < 8)：中度阅读——提取研究目的、方法、关键发现
  · "coarse"(relevance_score < 5)：粗读——仅需1-2句概述
- read_reason: 分级理由（用中文简述为何给该分数和等级）

输出JSON字段：
need_replan: bool 是否需要重新规划检索
defect_type: str | null 缺陷类型
defect_desc: str 详细问题描述
suggest_new_keywords: list[str] 补充检索关键词
rated_papers: list[object] 每篇论文的打分结果，字段: title, relevance_score(float), read_level(str), read_reason(str)
"""),
    ("human", "用户原始需求：{user_query}\n本次检索关键词：{search_keys}\n检索到的论文列表：{paper_list}\n已提取文献摘要：{paper_summaries}")
])

audit_chain = audit_prompt | llm

def audit_and_correct(state: dict) -> dict:
    """Observer主节点函数：审计结果，判断是否需要重规划，并对论文进行三级分级打分。

    新增三级文献分级精读逻辑：
    - 从 LLM 响应中解析 rated_papers（每篇论文的相关度分数、阅读等级、分级理由）
    - 将 scored papers 合并到 retrieved_papers 中，同时输出独立的 rated_papers 列表
    - rated_papers 写入 AgentState 供 executor_read 分层精读节点消费
    """
    user_query = state["raw_user_query"]
    search_keys = state["planner_search_keywords"]
    paper_list = state["retrieved_papers"]
    paper_summaries = state["paper_summaries"]

    # LLM执行审计（含论文分级打分）
    resp = audit_chain.invoke({
        "user_query": user_query,
        "search_keys": search_keys,
        "paper_list": json.dumps(paper_list, ensure_ascii=False),
        "paper_summaries": json.dumps(paper_summaries, ensure_ascii=False)
    })
    audit_result = json.loads(resp.content)

    # ========== 三级文献分级精读：解析 LLM 输出的打分结果 ==========
    raw_rated: list[dict] = audit_result.get("rated_papers", []) or []
    rated_papers: list[PaperItem] = []
    # 建立标题到检索论文的映射，用于合并分数
    title_to_paper: dict[str, dict] = {}
    for paper in paper_list:
        title_key: str = str(paper.get("title", "")).strip().lower()
        title_to_paper[title_key] = paper

    for scored in raw_rated:
        scored_title: str = str(scored.get("title", "")).strip()
        relevance_score: float = float(scored.get("relevance_score", 0.0))
        read_level: str = str(scored.get("read_level", "coarse")).lower()
        read_reason: str = str(scored.get("read_reason", ""))

        # 规范化 read_level 到 coarse/medium/deep 三档
        if read_level not in ("coarse", "medium", "deep"):
            if relevance_score >= 8.0:
                read_level = "deep"
            elif relevance_score >= 5.0:
                read_level = "medium"
            else:
                read_level = "coarse"

        paper_item: PaperItem = {
            "title": scored_title,
            "relevance_score": relevance_score,
            "read_level": read_level,
            "read_reason": read_reason,
        }
        rated_papers.append(paper_item)

        # 合并相关度分数和阅读等级到原始 retrieved_papers
        scored_title_lower: str = scored_title.lower()
        if scored_title_lower in title_to_paper:
            title_to_paper[scored_title_lower]["relevance_score"] = relevance_score
            title_to_paper[scored_title_lower]["read_level"] = read_level
            title_to_paper[scored_title_lower]["read_reason"] = read_reason

    # 将合并后的论文列表重新组装（保留未被打分的论文）
    merged_papers: list[dict] = []
    for paper in paper_list:
        title_key: str = str(paper.get("title", "")).strip().lower()
        if title_key in title_to_paper:
            merged_papers.append(title_to_paper[title_key])
        else:
            merged_papers.append(paper)

    # 组装单条审计日志（包含分级打分结果）
    single_log: AuditRecord = {
        "task_type": state["standard_task_type"],
        "original_query": user_query,
        "planner_keywords": search_keys,
        "retrieved_papers": merged_papers,
        "paper_summary_results": paper_summaries,
        "defect_type": audit_result["defect_type"],
        "defect_desc": audit_result["defect_desc"],
        "need_replan": audit_result["need_replan"],
        "suggest_new_keywords": audit_result["suggest_new_keywords"],
        "rated_papers": rated_papers,
    }

    # 写入全局审计日志
    new_audit_logs = state.get("audit_logs", [])
    new_audit_logs.append(single_log)

    # 更新全局状态，传递给下一个分支判断
    # rated_papers 供 executor_read 分层精读节点消费
    return {
        "audit_logs": new_audit_logs,
        "current_defect": audit_result["defect_type"],
        "suggest_keywords": audit_result["suggest_new_keywords"],
        "need_replan": audit_result["need_replan"],
        "rated_papers": rated_papers,
    }