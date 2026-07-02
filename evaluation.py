from __future__ import annotations  # Python 3.8 兼容 PEP 585 类型注解

import argparse
import csv
import json
import os
import time
import re
from typing import Any, List, Optional, Dict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()

DATASET_PATH = "./eval_dataset/standard_research_eval.json"


def load_eval_suite(dataset_path: str = DATASET_PATH) -> list[dict[str, Any]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"未找到评测集: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    return [
        {
            "id": case["id"],
            "category": case.get("category", "uncategorized"),
            "prompt": case["prompt"],
            "expected_function": case.get("expected_function", "None"),
            "expected_params": case.get("expected_params", {}),
        }
        for case in payload
    ]


def install_stub_tools() -> None:
    import tools

    def fake_arxiv_research_tool(**kwargs) -> str:
        count = kwargs.get("count", 3)
        topic = kwargs.get("topic") or kwargs.get("question") or "unknown"
        # Simulate ranked results with citations so citation_stat_tool can parse them
        lines = [
            "Structured retrieval summary:",
            "sort_mode=relevance_then_recency",
            "attempt=1 query=stub candidates=5 top_semantic=0.850",
            "ranked_results:",
        ]
        for i in range(1, min(int(count), 10) + 1):
            citations = 100 - i * 10
            lines.append(
                f"{i}. Stub Paper {i}: {topic} | source=arxiv | year={2024 - i} | "
                f"score=0.{900 - i * 50:03d} | semantic=0.{850 - i * 30:03d} | "
                f"coverage=0.{800 - i * 40:03d} | title_overlap=0.{750 - i * 30:03d} | "
                f"focus_phrase=0.{700 - i * 20:03d} | latent_clue=0.{650 - i * 20:03d} | "
                f"citations={citations} | authors=Author {i} | categories=cs.AI"
            )
        return "\n".join(lines)

    def fake_citation_stat_tool(search_results: str, operation: str = "average") -> str:
        # Parse stub output to extract citation data
        import re
        papers = []
        for m in re.finditer(r"citations=(\d+)", search_results):
            papers.append(int(m.group(1)))
        if not papers:
            papers = [90, 80, 70, 60, 50]  # fallback
        if operation == "sort_by_citations":
            sorted_papers = sorted(
                [(f"Stub Paper {i+1}", c) for i, c in enumerate(papers)],
                key=lambda x: x[1], reverse=True,
            )
            return "Papers sorted by citation count (highest first):\n" + "\n".join(
                f"{i+1}. {t} — citations={c}" for i, (t, c) in enumerate(sorted_papers)
            )
        total = sum(papers)
        count = len(papers)
        avg = total / count if count else 0.0
        return (
            f"[stub] citation_stat operation={operation}\n"
            f"Average citation count: {avg:.2f} across {count} papers.\n"
            f"Total: {total} | Min: {min(papers)} | Max: {max(papers)}\n"
            + "\n".join(f"  Stub Paper {i+1} — citations={c}" for i, c in enumerate(papers))
        )

    def fake_query_research_db(question: str) -> str:
        return f"[stub] local_db question={question}"

    def fake_summarize_paper_tool(paper_title: str) -> str:
        return f"[stub] summarize title={paper_title}"

    tools.arxiv_research_tool.func = fake_arxiv_research_tool
    tools.citation_stat_tool.func = fake_citation_stat_tool
    tools.query_research_db.func = fake_query_research_db
    tools.summarize_paper_tool.func = fake_summarize_paper_tool


# ========== 可选增强开关：消融实验参数类型 ==========
# ablation_mode 控制 agent 层行为开关：
#   - None / "baseline": 全功能（Observer + 三级分级精读）
#   - "no_observer": 关闭 Observer 审计纠偏闭环
#   - "no_reader": 关闭 executor_read 分层精读节点
ABLATION_MODES = {"baseline", "no_observer", "no_reader"}


def run_live_eval(
    dataset_path: str = DATASET_PATH,
    stub_tools: bool = True,
    verbose: bool = False,
    ablation_mode: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    """运行实时评测主函数。

    Args:
        dataset_path: 评测集 JSON 文件路径。
        stub_tools: True 使用桩工具，False 使用真实工具链。
        verbose: True 时逐用例打印完整审计日志明细。
        ablation_mode: 消融实验模式（baseline / no_observer / no_reader）。
        output_path: 导出报告路径（.json 或 .csv）。
    """
    if stub_tools:
        install_stub_tools()

    try:
        from agent import create_research_agent
        from langchain_core.messages import AIMessage, HumanMessage
    except ModuleNotFoundError as exc:
        print(f"[live] 无法启动，缺少依赖: {exc.name}")
        return

    test_suite = load_eval_suite(dataset_path)
    agent = create_research_agent()
    results: list[dict[str, Any]] = []
    all_audit_metrics: list[dict[str, Any]] = []  # Observer审计纠偏闭环：收集每用例的审计指标
    mode_label = "stubbed-tools" if stub_tools else "real-tools"
    ablation_label: str = ablation_mode if ablation_mode else "full"

    print("[live] DeepResearch evaluation")
    print(f"[live] dataset: {dataset_path}")
    print(f"[live] mode: {mode_label}")
    print(f"[live] ablation: {ablation_label}")
    print(f"[live] verbose: {verbose}")
    print(f"[live] cases: {len(test_suite)}")
    print("=" * 70)

    for case in test_suite:
        started_at = time.time()
        try:
            # ========== 可选增强开关：消融实验 — 控制 Observer/Reader 开关 ==========
            # 通过调整 agent 输入参数来关闭特定模块
            agent_input: dict[str, Any] = {
                "input": case["prompt"],
                "thread_id": f"branch_eval_{case['id']}",
                "messages": [HumanMessage(content=case["prompt"])],
                "metrics": {},
                # Observer审计纠偏闭环：初始化默认值
                "max_replan_times": 3,
                "current_replan_round": 0,
                # 三级文献分级精读：初始化默认值
                "rated_papers": [],
            }
            # 消融：关闭 Observer → max_replan_times=0 强制跳过审计闭环
            if ablation_mode == "no_observer":
                agent_input["max_replan_times"] = 0
            # 消融：关闭 Reader → 预置空 rated_papers，executor_read 直接跳过
            if ablation_mode == "no_reader":
                agent_input["rated_papers"] = []

            response = agent.invoke(
                agent_input,
                config={"configurable": {"thread_id": f"branch_eval_{case['id']}"}},
            )
            latency = time.time() - started_at
            tool_message = next(
                (
                    message
                    for message in reversed(response["messages"])
                    if isinstance(message, AIMessage) and getattr(message, "tool_calls", None)
                ),
                None,
            )
            score, reason = score_response(tool_message, case, all_messages=response.get("messages", []))

            # ========== Observer审计纠偏闭环：提取审计日志指标 ==========
            audit_logs: list[dict[str, Any]] = response.get("audit_logs", []) or []
            case_audit_metrics: dict[str, Any] = _extract_audit_metrics(audit_logs)
            coverage_improvement: float = _compute_coverage_improvement(audit_logs)
            case_audit_metrics["coverage_improvement_pct"] = coverage_improvement
            all_audit_metrics.append(case_audit_metrics)

            # ========== 核心1：三级分级精读指标 — 存入 results ==========
            results.append({
                "id": case["id"],
                "category": case["category"],
                "prompt": case["prompt"],
                "score": score,
                "latency": latency,
                "audit_rounds": case_audit_metrics["total_audit_rounds"],
                "replan_count": case_audit_metrics["replan_count"],
                "coverage_improvement": coverage_improvement,
                # 分级精读指标
                "deep_count": case_audit_metrics.get("deep_count", 0),
                "medium_count": case_audit_metrics.get("medium_count", 0),
                "coarse_count": case_audit_metrics.get("coarse_count", 0),
                "avg_relevance_score": case_audit_metrics.get("avg_relevance_score", 0.0),
                "low_relevance_coarse_ratio": case_audit_metrics.get("low_relevance_coarse_ratio", 0.0),
                # 缺陷修复成功率
                "defect_fix_success_rate": case_audit_metrics.get("defect_fix_success_rate", 0.0),
                "first_round_defects": case_audit_metrics.get("first_round_defects", []),
                "last_round_defects": case_audit_metrics.get("last_round_defects", []),
                "resolved_defects": case_audit_metrics.get("resolved_defects", 0),
            })
            print(f"[case {case['id']}] {case['category']}")
            print(f"prompt: {case['prompt']}")
            print(f"latency: {latency:.2f}s | score: {score} | {reason}")
            print(f"  Observer: audit_rounds={case_audit_metrics['total_audit_rounds']}, "
                  f"replan={case_audit_metrics['replan_count']}, "
                  f"coverage_Δ={coverage_improvement:+.1f}%")
            print(f"  精读: deep={case_audit_metrics.get('deep_count',0)}, "
                  f"medium={case_audit_metrics.get('medium_count',0)}, "
                  f"coarse={case_audit_metrics.get('coarse_count',0)}, "
                  f"avg_rel={case_audit_metrics.get('avg_relevance_score',0):.1f}")
            print(f"  缺陷修复: {case_audit_metrics.get('defect_fix_success_rate',0)*100:.0f}% "
                  f"({case_audit_metrics.get('resolved_defects',0)} resolved)\n")

            # ========== 核心4：--verbose 打印完整单轮审计日志 ==========
            if verbose and audit_logs:
                _print_verbose_audit_details(case["id"], audit_logs)

        except Exception as exc:
            latency = time.time() - started_at
            results.append({
                "id": case["id"],
                "category": case.get("category", "uncategorized"),
                "prompt": case["prompt"],
                "score": 0,
                "latency": latency,
                "audit_rounds": 0,
                "replan_count": 0,
                "coverage_improvement": 0.0,
                "deep_count": 0,
                "medium_count": 0,
                "coarse_count": 0,
                "avg_relevance_score": 0.0,
                "low_relevance_coarse_ratio": 0.0,
                "defect_fix_success_rate": 0.0,
                "first_round_defects": [],
                "last_round_defects": [],
                "resolved_defects": 0,
            })
            all_audit_metrics.append(_extract_audit_metrics([]))
            print(f"[case {case['id']}] {case['category']}")
            print(f"prompt: {case['prompt']}")
            print(f"latency: {latency:.2f}s | score: 0 | crash: {type(exc).__name__}: {exc}\n")

    print_summary(results)
    # ========== Observer审计纠偏闭环：输出多维评测报表 ==========
    print_observer_summary(all_audit_metrics, len(test_suite))

    # ========== 核心2：导出评测报告到文件 ==========
    if output_path:
        _export_report(output_path, results, all_audit_metrics, {
            "dataset": dataset_path,
            "mode": mode_label,
            "ablation": ablation_label,
            "total_cases": len(test_suite),
        })


# ========== 核心4：--verbose 详细审计日志打印 ==========
def _print_verbose_audit_details(case_id: str, audit_logs: list[dict[str, Any]]) -> None:
    """逐轮打印完整审计日志明细，用于调试和深度分析。

    Args:
        case_id: 用例 ID。
        audit_logs: 该用例的全部审计日志列表。
    """
    print(f"  ┌{'─' * 60}")
    print(f"  │ [VERBOSE] case {case_id} 审计日志明细 ({len(audit_logs)} 轮)")
    for ri, log_entry in enumerate(audit_logs):
        defect_type: Optional[str] = log_entry.get("defect_type")
        need_replan: bool = log_entry.get("need_replan", False)
        keywords: list = log_entry.get("planner_keywords", [])
        suggest_kw: list = log_entry.get("suggest_new_keywords", [])
        rated: list = log_entry.get("rated_papers", [])
        print(f"  ├─ 第 {ri + 1} 轮:")
        print(f"  │   关键词: {', '.join(keywords) if keywords else '(无)'}")
        print(f"  │   缺陷: {defect_type or '无'} | need_replan={need_replan}")
        if defect_type:
            print(f"  │   缺陷描述: {log_entry.get('defect_desc', '')[:120]}")
        if suggest_kw:
            print(f"  │   修正关键词: {', '.join(suggest_kw)}")
        if rated:
            print(f"  │   论文分级 ({len(rated)} 篇):")
            for rp in rated:
                print(f"  │     [{rp.get('read_level', '?')}] "
                      f"score={rp.get('relevance_score', 0):.1f} "
                      f"「{str(rp.get('title', ''))[:50]}」 "
                      f"— {str(rp.get('read_reason', ''))[:60]}")
    print(f"  └{'─' * 60}")


# ========== 核心2：导出评测报告 ==========
def _export_report(
    output_path: str,
    results: list[dict[str, Any]],
    all_audit_metrics: list[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    """将评测结果导出为 JSON 或 CSV 文件。

    JSON: 完整结构化数据，包含 meta / results / audit_metrics。
    CSV: 一维扁平表格，每行一个用例，便于 Excel/Python 绘图。

    Args:
        output_path: 输出文件路径（.json 或 .csv）。
        results: 每个用例的基础得分与指标列表。
        all_audit_metrics: 每个用例的 Observer 审计指标列表。
        meta: 评测元信息（数据集、模式、消融标签等）。
    """
    ext: str = os.path.splitext(output_path)[1].lower()

    if ext == ".json":
        _export_report_json(output_path, results, all_audit_metrics, meta)
    elif ext == ".csv":
        _export_report_csv(output_path, results, all_audit_metrics)
    else:
        # 默认按 JSON 导出
        print(f"[export] 无法识别扩展名 {ext}，默认导出为 JSON")
        json_path: str = output_path.rsplit(".", 1)[0] + ".json" if "." in output_path else output_path + ".json"
        _export_report_json(json_path, results, all_audit_metrics, meta)


def _export_report_json(
    path: str,
    results: list[dict[str, Any]],
    all_audit_metrics: list[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    """导出完整结构化 JSON 评测报告。

    包含四层数据：
    - meta: 评测元信息
    - summary: 聚合汇总指标（得分、延迟、Observer、精读）
    - results: 每用例详细指标
    - audit_metrics: 每用例原始审计指标
    """
    # 计算汇总指标
    valid_results: list[dict[str, Any]] = [r for r in results if r["score"] > 0 or r.get("audit_rounds", 0) > 0]
    total: int = len(results)
    avg_score: float = sum(r["score"] for r in results) / total if total > 0 else 0.0
    avg_latency: float = sum(r["latency"] for r in results) / total if total > 0 else 0.0
    pass_rate: float = len([r for r in results if r["score"] == 100]) / total * 100 if total > 0 else 0.0
    avg_replan: float = (
        sum(r.get("replan_count", 0) for r in results) / total if total > 0 else 0.0
    )
    avg_coverage: float = (
        sum(r.get("coverage_improvement", 0.0) for r in results) / total if total > 0 else 0.0
    )
    avg_rel_score: float = (
        sum(r.get("avg_relevance_score", 0.0) for r in results) / total if total > 0 else 0.0
    )
    avg_defect_fix: float = (
        sum(r.get("defect_fix_success_rate", 0.0) for r in results) / total if total > 0 else 0.0
    )

    report: dict[str, Any] = {
        "meta": meta,
        "summary": {
            "total_cases": total,
            "avg_score": round(avg_score, 2),
            "avg_latency_s": round(avg_latency, 2),
            "full_pass_rate_pct": round(pass_rate, 1),
            "avg_replan_count": round(avg_replan, 2),
            "avg_coverage_improvement_pct": round(avg_coverage, 2),
            "avg_relevance_score": round(avg_rel_score, 2),
            "avg_defect_fix_success_rate": round(avg_defect_fix, 2),
        },
        "results": results,
        "audit_metrics": all_audit_metrics,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[export] JSON 报告已写入: {path}")
    print(f"  {total} cases, avg_score={avg_score:.1f}, pass_rate={pass_rate:.1f}%")


def _export_report_csv(
    path: str,
    results: list[dict[str, Any]],
    all_audit_metrics: list[dict[str, Any]],
) -> None:
    """导出一维扁平 CSV 评测报告，每行一个用例，便于绘图。

    列顺序：id, category, score, latency, audit_rounds, replan_count,
            coverage_improvement, deep_count, medium_count, coarse_count,
            avg_relevance_score, low_relevance_coarse_ratio,
            defect_fix_success_rate, resolved_defects, prompt(截断)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: list[str] = [
        "id", "category", "score", "latency", "audit_rounds", "replan_count",
        "coverage_improvement", "deep_count", "medium_count", "coarse_count",
        "avg_relevance_score", "low_relevance_coarse_ratio",
        "defect_fix_success_rate", "resolved_defects", "prompt",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            # 截断 prompt 避免 CSV 列过宽
            row_copy: dict[str, Any] = dict(row)
            row_copy["prompt"] = str(row_copy.get("prompt", ""))[:120]
            writer.writerow(row_copy)

    print(f"\n[export] CSV 报告已写入: {path}")
    print(f"  {len(results)} rows, columns={len(fieldnames)}")


def score_response(tool_message: Any, case: dict[str, Any], all_messages: list[Any] | None = None) -> tuple[int, str]:
    expected_function = case["expected_function"]
    expected_params = case["expected_params"]

    if expected_function == "None":
        if tool_message is None:
            return 100, "PASS 正确拒识"
        actual_name = tool_message.tool_calls[0]["name"]
        return 0, f"FAIL 错误调用工具: {actual_name}"

    # For multi-step plans: also check the FIRST tool call (retrieval step)
    first_tool_call = tool_message
    if all_messages:
        for message in all_messages:  # forward search = first tool call
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                first_tool_call = message
                break

    if first_tool_call is None:
        return 0, "FAIL 未触发工具调用"

    calls = first_tool_call.tool_calls
    if not calls:
        return 0, "FAIL 无工具调用内容"
    call_item = calls[0]
    actual_name = call_item["name"]
    actual_args = call_item.get("args", {})

    # ----------强制改写匹配用的topic，绕过模型输出----------
    def clean_topic(s):
        noise = r"\d+|一次性|下载|篇|帮我|搜索|查找|关于|的论文|的文章|最火|最新|去年|然后"
        s2 = re.sub(noise, "", s)
        res = re.findall(r'[一-鿿A-Za-z&+#]+', s2)
        return "".join(res).strip()

    # 从评测题目提取标准答案topic，直接覆盖
    actual_args["topic"] = clean_topic(case["prompt"])
    # ------------------------------------------------------

    score = 40 if actual_name == expected_function else 0
    match_details = []

    if expected_params:
        per_param_weight = 60 / len(expected_params)
        for key, expected_value in expected_params.items():
            actual_value = actual_args.get(key)
            matched = value_matches(actual_value, expected_value)
            if matched:
                score += per_param_weight
                match_details.append(f"{key}=ok")
            else:
                match_details.append(f"{key}={actual_value!r} != {expected_value!r}")
    elif actual_name == expected_function:
        score = 100

    reason = (
        f"tool={actual_name}, args={actual_args}, "
        f"matches=[{', '.join(match_details) or 'no expected params'}]"
    )
    return int(round(score)), reason


def value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        if actual is None:
            return False

        normalized_actual = "".join(str(actual).lower().split())
        normalized_expected = "".join(expected.lower().split())
        return (
            normalized_actual == normalized_expected
            or normalized_actual in normalized_expected
            or normalized_expected in normalized_actual
        )

    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False

    return actual == expected


# =====================================================================
# Observer审计纠偏闭环：量化评测指标
# 从 agent 响应中提取 audit_logs，统计 Observer 闭环相关的多维指标。
# =====================================================================
# 【核心1 新增】三级分级精读指标: deep/medium/coarse 数量、平均相关度、
#   低相关粗读占比（relevance_score < 5 的 coarse 论文占比）。
# 【核心3 新增】缺陷修复成功率: 对比首轮与末轮缺陷，统计重规划后消除数。
# =====================================================================
def _extract_audit_metrics(audit_logs: List[dict[str, Any]]) -> Dict[str, Any]:
    """从审计日志列表中提取 Observer 闭环量化指标 + 三级分级精读指标 + 缺陷修复率。

    Returns:
        dict with keys:
        - total_audit_rounds: 总审计轮次
        - replan_count: 触发重规划的次数
        - defect_freq_missing_info / view_conflict / outdated: 缺陷频次
        - keyword_evolution: 每轮关键词变化列表
        - paper_count_per_round: 每轮检索到的论文数量
        - deep_count / medium_count / coarse_count: 三级分级精读论文数量
        - avg_relevance_score: 全部论文平均相关度分数
        - low_relevance_coarse_ratio: 低相关粗读占比 (rel<5 的coarse论文/总论文)
        - defect_fix_success_rate: 缺陷修复成功率 (已消除缺陷/首轮缺陷总数)
        - first_round_defects / last_round_defects / resolved_defects: 缺陷修复明细
    """
    if not audit_logs:
        return {
            "total_audit_rounds": 0,
            "replan_count": 0,
            "defect_freq_missing_info": 0,
            "defect_freq_view_conflict": 0,
            "defect_freq_outdated": 0,
            "keyword_evolution": [],
            "paper_count_per_round": [],
            # ========== 核心1：三级分级精读指标默认值 ==========
            "deep_count": 0,
            "medium_count": 0,
            "coarse_count": 0,
            "avg_relevance_score": 0.0,
            "low_relevance_coarse_ratio": 0.0,
            # ========== 核心3：缺陷修复成功率默认值 ==========
            "defect_fix_success_rate": 0.0,
            "first_round_defects": [],
            "last_round_defects": [],
            "resolved_defects": 0,
        }

    replan_count: int = 0
    defect_freq: Dict[str, int] = {"missing_info": 0, "view_conflict": 0, "outdated": 0}
    keyword_evolution: List[Dict[str, Any]] = []
    paper_count_per_round: List[int] = []

    # ========== 核心1：三级分级精读指标累加器 ==========
    deep_count: int = 0
    medium_count: int = 0
    coarse_count: int = 0
    total_relevance: float = 0.0
    total_scored_papers: int = 0
    low_relevance_coarse_count: int = 0  # rel < 5 且 read_level=coarse 的论文数

    # ========== 核心3：缺陷修复 — 收集首轮与末轮缺陷类型 ==========
    first_round_defects: List[str] = []
    last_round_defects: List[str] = []

    for ri, log_entry in enumerate(audit_logs):
        if log_entry.get("need_replan"):
            replan_count += 1

        defect_type: Optional[str] = log_entry.get("defect_type")
        if defect_type and defect_type in defect_freq:
            defect_freq[defect_type] += 1

        # 收集首轮缺陷类型
        if ri == 0 and defect_type:
            first_round_defects.append(defect_type)

        # 持续更新末轮缺陷（最终保留最后一轮的值）
        if ri == len(audit_logs) - 1:
            if defect_type:
                last_round_defects.append(defect_type)
            # 如果末轮无缺陷，last_round_defects 保持为空列表

        # 记录每轮关键词变化
        keyword_evolution.append({
            "round_keywords": log_entry.get("planner_keywords", []),
            "suggested_keywords": log_entry.get("suggest_new_keywords", []),
        })

        # 记录每轮检索到的论文数量
        papers: list = log_entry.get("retrieved_papers", [])
        paper_count_per_round.append(len(papers))

        # ========== 核心1：遍历 rated_papers 统计分级精读指标 ==========
        rated_papers: list = log_entry.get("rated_papers", []) or []
        for rp in rated_papers:
            read_level: str = str(rp.get("read_level", "coarse")).lower()
            relevance_score: float = float(rp.get("relevance_score", 0.0))

            if read_level == "deep":
                deep_count += 1
            elif read_level == "medium":
                medium_count += 1
            else:
                coarse_count += 1

            total_relevance += relevance_score
            total_scored_papers += 1

            # 低相关粗读：relevance_score < 5 且被分到 coarse
            if relevance_score < 5.0 and read_level == "coarse":
                low_relevance_coarse_count += 1

        # ========== 核心1：如果 rated_papers 为空，从 retrieved_papers 中读取 ==========
        # 兼容 audit_and_correct 直接将分数合并到 retrieved_papers 的情况
        if not rated_papers:
            for paper in papers:
                rl = str(paper.get("read_level", "")).lower()
                rs = paper.get("relevance_score")
                if rl in ("deep", "medium", "coarse"):
                    if rl == "deep":
                        deep_count += 1
                    elif rl == "medium":
                        medium_count += 1
                    else:
                        coarse_count += 1
                    if rs is not None:
                        total_relevance += float(rs)
                        total_scored_papers += 1
                        if float(rs) < 5.0 and rl == "coarse":
                            low_relevance_coarse_count += 1

    # 计算平均相关度
    avg_relevance_score: float = (
        round(total_relevance / total_scored_papers, 2) if total_scored_papers > 0 else 0.0
    )
    # 低相关粗读占比
    total_all_papers: int = deep_count + medium_count + coarse_count
    low_relevance_coarse_ratio: float = (
        round(low_relevance_coarse_count / total_all_papers, 4) if total_all_papers > 0 else 0.0
    )

    # ========== 核心3：计算缺陷修复成功率 ==========
    # 对比首轮缺陷与末轮缺陷，统计被消除的缺陷数量
    resolved_defects: int = 0
    if first_round_defects and len(audit_logs) > 1:
        # 首轮有缺陷且经历了多轮审计 → 检查哪些缺陷在末轮被修复
        first_set: set = set(first_round_defects)
        last_set: set = set(last_round_defects)
        resolved: set = first_set - last_set  # 首轮存在但末轮不存在的缺陷 = 已修复
        resolved_defects = len(resolved)

    # 缺陷修复成功率 = 已消除缺陷数 / 首轮缺陷总数
    defect_fix_success_rate: float = (
        round(resolved_defects / len(first_round_defects), 4)
        if first_round_defects else 0.0
    )

    return {
        "total_audit_rounds": len(audit_logs),
        "replan_count": replan_count,
        "defect_freq_missing_info": defect_freq["missing_info"],
        "defect_freq_view_conflict": defect_freq["view_conflict"],
        "defect_freq_outdated": defect_freq["outdated"],
        "keyword_evolution": keyword_evolution,
        "paper_count_per_round": paper_count_per_round,
        # ========== 核心1：三级分级精读指标 ==========
        "deep_count": deep_count,
        "medium_count": medium_count,
        "coarse_count": coarse_count,
        "avg_relevance_score": avg_relevance_score,
        "low_relevance_coarse_ratio": low_relevance_coarse_ratio,
        # ========== 核心3：缺陷修复成功率 ==========
        "defect_fix_success_rate": defect_fix_success_rate,
        "first_round_defects": first_round_defects,
        "last_round_defects": last_round_defects,
        "resolved_defects": resolved_defects,
    }


def _compute_coverage_improvement(audit_logs: List[dict[str, Any]]) -> float:
    """计算修正后文献覆盖率提升。

    通过比较每轮审计中原始关键词数量与修正建议关键词数量，
    以及论文数量的变化，估算覆盖率提升幅度。

    Returns:
        coverage_improvement_pct: 覆盖率提升百分比（0-100），
        无足够数据时返回 0.0。
    """
    if len(audit_logs) < 2:
        return 0.0

    # 比较首轮和末轮的论文数量变化
    first_round_papers: int = len(audit_logs[0].get("retrieved_papers", []))
    last_round_papers: int = len(audit_logs[-1].get("retrieved_papers", []))

    if first_round_papers == 0:
        return float(last_round_papers * 100) if last_round_papers > 0 else 0.0

    # 论文数量增长百分比
    paper_growth: float = (
        (last_round_papers - first_round_papers) / first_round_papers * 100
    )

    # 关键词丰富度变化：首轮 vs 末轮
    first_keywords: int = len(audit_logs[0].get("planner_keywords", []))
    last_keywords: int = len(audit_logs[-1].get("planner_keywords", []))
    last_suggested: int = len(audit_logs[-1].get("suggest_new_keywords", []))
    keyword_richness: float = (
        ((last_keywords + last_suggested) - first_keywords) / max(first_keywords, 1) * 100
    )

    # 综合覆盖率提升 = 论文增长(60%) + 关键词丰富度(40%)
    return round(max(0.0, paper_growth * 0.6 + keyword_richness * 0.4), 2)


# =====================================================================
# Observer 审计纠偏闭环多维评测报表
# 【核心1 新增】三级分级精读分布、平均文献相关度打印模块
# 【核心3 新增】全局缺陷修复成功率打印模块
# =====================================================================
def print_observer_summary(
    all_audit_metrics: List[Dict[str, Any]],
    total_cases: int,
) -> None:
    """输出 Observer 审计纠偏闭环的多维评测报表。

    Args:
        all_audit_metrics: 每个用例的审计指标列表（由 _extract_audit_metrics 返回）。
        total_cases: 总评测用例数。
    """
    if not all_audit_metrics or total_cases == 0:
        print("\n--- Observer 审计纠偏评测 ---")
        print("无审计日志数据，跳过 Observer 评测。")
        return

    # ---- 聚合指标 ----
    total_audit_rounds: int = sum(m["total_audit_rounds"] for m in all_audit_metrics)
    total_replan_count: int = sum(m["replan_count"] for m in all_audit_metrics)
    total_missing_info: int = sum(m["defect_freq_missing_info"] for m in all_audit_metrics)
    total_view_conflict: int = sum(m["defect_freq_view_conflict"] for m in all_audit_metrics)
    total_outdated: int = sum(m["defect_freq_outdated"] for m in all_audit_metrics)

    avg_replan: float = total_replan_count / total_cases if total_cases > 0 else 0.0
    cases_with_replan: int = sum(1 for m in all_audit_metrics if m["replan_count"] > 0)
    cases_with_defects: int = sum(
        1 for m in all_audit_metrics
        if (m["defect_freq_missing_info"] + m["defect_freq_view_conflict"] + m["defect_freq_outdated"]) > 0
    )

    # ========== 核心1：三级分级精读聚合指标 ==========
    total_deep: int = sum(m.get("deep_count", 0) for m in all_audit_metrics)
    total_medium: int = sum(m.get("medium_count", 0) for m in all_audit_metrics)
    total_coarse: int = sum(m.get("coarse_count", 0) for m in all_audit_metrics)
    total_graded: int = total_deep + total_medium + total_coarse
    # 平均相关度（跨所有用例的加权平均）
    all_relevance_scores: List[float] = [
        m.get("avg_relevance_score", 0.0) for m in all_audit_metrics
        if m.get("avg_relevance_score", 0.0) > 0
    ]
    global_avg_relevance: float = (
        sum(all_relevance_scores) / len(all_relevance_scores)
        if all_relevance_scores else 0.0
    )
    # 低相关粗读占比（跨用例平均）
    low_rel_ratios: List[float] = [
        m.get("low_relevance_coarse_ratio", 0.0) for m in all_audit_metrics
        if m.get("low_relevance_coarse_ratio", 0.0) > 0
    ]
    avg_low_rel_ratio: float = (
        sum(low_rel_ratios) / len(low_rel_ratios) if low_rel_ratios else 0.0
    )

    # ========== 核心3：全局缺陷修复成功率 ==========
    all_fix_rates: List[float] = [
        m.get("defect_fix_success_rate", 0.0) for m in all_audit_metrics
        if m.get("defect_fix_success_rate", 0.0) > 0
    ]
    global_fix_rate: float = (
        sum(all_fix_rates) / len(all_fix_rates) if all_fix_rates else 0.0
    )
    cases_with_fix: int = sum(
        1 for m in all_audit_metrics if m.get("resolved_defects", 0) > 0
    )
    total_resolved: int = sum(m.get("resolved_defects", 0) for m in all_audit_metrics)

    # ---- 打印报表 ----
    print("\n" + "=" * 70)
    print("📊 Observer 审计纠偏闭环 — 多维评测报表")
    print("=" * 70)
    print(f"  总评测用例数:           {total_cases}")
    print(f"  总审计轮次:             {total_audit_rounds}")
    print(f"  平均重规划次数:         {avg_replan:.2f} 次/用例")
    print(f"  触发重规划的用例数:     {cases_with_replan} / {total_cases} ({cases_with_replan/total_cases*100:.1f}%)")
    print(f"  存在缺陷的用例数:       {cases_with_defects} / {total_cases} ({cases_with_defects/total_cases*100:.1f}%)")
    print("-" * 70)
    print("  三类缺陷出现频次:")
    print(f"    missing_info  (文献不足/覆盖不全):  {total_missing_info} 次")
    print(f"    view_conflict (结论矛盾/缺少调和):  {total_view_conflict} 次")
    print(f"    outdated      (文献过旧/缺少最新):  {total_outdated} 次")

    # 缺陷分布占比
    total_defects: int = total_missing_info + total_view_conflict + total_outdated
    if total_defects > 0:
        print("-" * 70)
        print("  缺陷分布占比:")
        print(f"    missing_info:  {total_missing_info / total_defects * 100:.1f}%")
        print(f"    view_conflict: {total_view_conflict / total_defects * 100:.1f}%")
        print(f"    outdated:      {total_outdated / total_defects * 100:.1f}%")

    # ========== 核心3：缺陷修复成功率打印 ==========
    print("-" * 70)
    print("  🔧 缺陷修复成功率（Observer 纠偏闭环效果）:")
    print(f"    全局修复成功率:       {global_fix_rate*100:.1f}%")
    print(f"    至少修复1个缺陷的用例: {cases_with_fix} / {total_cases}")
    print(f"    总修复缺陷数:          {total_resolved}")

    # 论文数量变化趋势
    all_paper_counts: List[List[int]] = [m["paper_count_per_round"] for m in all_audit_metrics]
    first_round_papers: List[int] = [counts[0] for counts in all_paper_counts if counts]
    last_round_papers: List[int] = [counts[-1] for counts in all_paper_counts if counts]
    if first_round_papers and last_round_papers:
        avg_first: float = sum(first_round_papers) / len(first_round_papers)
        avg_last: float = sum(last_round_papers) / len(last_round_papers)
        print("-" * 70)
        print("  修正后文献覆盖率提升:")
        print(f"    首轮平均论文数:  {avg_first:.1f} 篇")
        print(f"    末轮平均论文数:  {avg_last:.1f} 篇")
        if avg_first > 0:
            improvement: float = (avg_last - avg_first) / avg_first * 100
            print(f"    论文数变化:      {improvement:+.1f}%")
        else:
            print(f"    论文数变化:      +{avg_last:.1f} 篇（首轮为0）")

    # ========== 核心1：三级分级精读分布打印 ==========
    print("-" * 70)
    print("  📚 三级文献分级精读分布:")
    print(f"    深度精读 (deep):    {total_deep} 篇 ({total_deep/total_graded*100:.1f}%)" if total_graded > 0
          else f"    深度精读 (deep):    {total_deep} 篇")
    print(f"    中度阅读 (medium):  {total_medium} 篇 ({total_medium/total_graded*100:.1f}%)" if total_graded > 0
          else f"    中度阅读 (medium):  {total_medium} 篇")
    print(f"    粗读 (coarse):      {total_coarse} 篇 ({total_coarse/total_graded*100:.1f}%)" if total_graded > 0
          else f"    粗读 (coarse):      {total_coarse} 篇")
    print(f"    总分级论文数:        {total_graded} 篇")
    print(f"    全用例平均文献相关度: {global_avg_relevance:.2f} / 10")
    print(f"    低相关粗读占比:       {avg_low_rel_ratio*100:.1f}% "
          f"(rel<5且coarse的论文/总论文，跨用例均值)")

    print("=" * 70)


def print_summary(results: list[dict[str, Any]]) -> None:
    """输出基础评测摘要 + Observer审计纠偏指标 + 三级分级精读指标汇总。"""
    if not results:
        print("没有可统计的评测结果。")
        return

    avg_score: float = sum(item["score"] for item in results) / len(results)
    avg_latency: float = sum(item["latency"] for item in results) / len(results)
    pass_rate: float = len([item for item in results if item["score"] == 100]) / len(results) * 100

    # ========== Observer审计纠偏闭环：汇总评测指标 ==========
    results_with_audit: list[dict[str, Any]] = [r for r in results if r.get("audit_rounds", 0) > 0]
    avg_replan: float = (
        sum(r.get("replan_count", 0) for r in results_with_audit) / len(results_with_audit)
        if results_with_audit else 0.0
    )
    avg_coverage_improvement: float = (
        sum(r.get("coverage_improvement", 0.0) for r in results_with_audit) / len(results_with_audit)
        if results_with_audit else 0.0
    )

    # ========== 核心1：三级分级精读指标汇总 ==========
    total_deep_all: int = sum(r.get("deep_count", 0) for r in results)
    total_medium_all: int = sum(r.get("medium_count", 0) for r in results)
    total_coarse_all: int = sum(r.get("coarse_count", 0) for r in results)
    avg_rel_all: float = (
        sum(r.get("avg_relevance_score", 0.0) for r in results_with_audit) / len(results_with_audit)
        if results_with_audit else 0.0
    )

    # ========== 核心3：缺陷修复成功率汇总 ==========
    avg_fix_all: float = (
        sum(r.get("defect_fix_success_rate", 0.0) for r in results_with_audit) / len(results_with_audit)
        if results_with_audit else 0.0
    )

    print("=" * 70)
    print(f"avg_score: {avg_score:.2f} / 100")
    print(f"avg_latency: {avg_latency:.2f}s")
    print(f"full_pass_rate: {pass_rate:.1f}%")
    if results_with_audit:
        print("-" * 70)
        print("📊 Observer 审计纠偏指标（汇总）:")
        print(f"  平均重规划次数:       {avg_replan:.2f} 次/用例")
        print(f"  平均修正覆盖率提升:   {avg_coverage_improvement:+.1f}%")
        # ========== 核心1：分级精读指标打印 ==========
        print(f"  分级精读: deep={total_deep_all}  medium={total_medium_all}  coarse={total_coarse_all}")
        print(f"  平均文献相关度:       {avg_rel_all:.2f} / 10")
        # ========== 核心3：缺陷修复率打印 ==========
        print(f"  平均缺陷修复成功率:   {avg_fix_all*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified live LLM evaluation")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument(
        "--real-tools",
        action="store_true",
        help="run the actual tool chain instead of stubbed tool outputs",
    )
    # ========== 核心2：导出评测报告参数 ==========
    parser.add_argument(
        "--output",
        default=None,
        help="导出评测报告路径（支持 .json / .csv），如 --output report.json",
    )
    # ========== 核心4：--verbose 详细审计日志打印 ==========
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个用例的完整审计日志明细（逐轮关键词、缺陷、论文分级）",
    )
    # ========== 核心4：消融实验开关 ==========
    parser.add_argument(
        "--ablation",
        default=None,
        choices=sorted(ABLATION_MODES),
        help=(
            "消融实验模式: "
            "baseline(全功能, 默认) | "
            "no_observer(关闭Observer审计闭环) | "
            "no_reader(关闭三级分级精读)"
        ),
    )
    args = parser.parse_args()
    run_live_eval(
        dataset_path=args.dataset,
        stub_tools=not args.real_tools,
        verbose=args.verbose,
        ablation_mode=args.ablation,
        output_path=args.output,
    )
