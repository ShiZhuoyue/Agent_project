from __future__ import annotations


import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from langfuse_config import langfuse_client, langfuse_callback

ROOT_DIR = Path(__file__).resolve().parent
LITSEARCH_DIR = ROOT_DIR / "eval_dataset" / "litsearch"
RAW_QUERY_FILE = LITSEARCH_DIR / "query" / "full-00000-of-00001.parquet"
ENRICHED_QUERY_FILE = LITSEARCH_DIR / "litsearch_queries_enriched.jsonl"
GOLD_CACHE_FILE = LITSEARCH_DIR / "litsearch_gold_titles.json"
REPORT_DIR = LITSEARCH_DIR / "reports"
RUNTIME_DIR = LITSEARCH_DIR / "runtime"

DEFAULT_PROBE_K = 20
DEFAULT_BATCH_SIZE = 100
S2_FIELDS = "title,year,externalIds"
RANKED_TITLE_PATTERN = re.compile(r"^\d+\.\s+(.*?)\s+\|\s+(?:source=[^|]+\s+\|\s+)?year=", re.MULTILINE)
RESOURCE_PATTERN = re.compile(
    r"\b(dataset|datasets|resource|resources|corpus|corpora|benchmark|benchmarks|tool|tools|"
    r"lexicon|dictionary|annotation|annotated|comments|translation resources?)\b",
    re.IGNORECASE,
)
BENCHMARK_PATTERN = re.compile(
    r"\b(benchmark|benchmarks|evaluation|evaluate|metric|metrics|leaderboard)\b",
    re.IGNORECASE,
)
ANALYSIS_PATTERN = re.compile(
    r"\b(survey|review|reviews|analysis|analyze|study|studies|explore|explores|"
    r"investigate|understand|detection|detect|identify|identification)\b",
    re.IGNORECASE,
)
COMPLEXITY_PATTERN = re.compile(
    r"\b(with|that|using|both|contain|contains|including|through|without|under|"
    r"specifically|focused on|at both|in addition|additional|manually|augmented)\b",
    re.IGNORECASE,
)


@dataclass
class CaseOutcome:
    query_id: str
    query: str
    query_set: str
    specificity: int
    quality: int
    specificity_bucket: str
    query_source_bucket: str
    quality_bucket: str
    capability_type: str
    complexity_bucket: str
    complexity_score: int
    tool_called: bool
    wrong_tool_name: str | None
    crashed: bool
    latency_seconds: float
    no_result: bool
    rewrite_used: bool
    top1_hit: float
    top3_hit: float
    recall_at_5: float
    recall_at_20: float
    mrr_at_20: float
    gold_count: int
    planner_args: dict[str, Any]
    gold_titles: list[str]
    default_titles: list[str]
    probe_titles: list[str]
    tool_output: str
    final_answer: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_set": self.query_set,
            "specificity": self.specificity,
            "quality": self.quality,
            "specificity_bucket": self.specificity_bucket,
            "query_source_bucket": self.query_source_bucket,
            "quality_bucket": self.quality_bucket,
            "capability_type": self.capability_type,
            "complexity_bucket": self.complexity_bucket,
            "complexity_score": self.complexity_score,
            "tool_called": self.tool_called,
            "wrong_tool_name": self.wrong_tool_name,
            "crashed": self.crashed,
            "latency_seconds": round(self.latency_seconds, 3),
            "no_result": self.no_result,
            "rewrite_used": self.rewrite_used,
            "top1_hit": self.top1_hit,
            "top3_hit": self.top3_hit,
            "recall_at_5": self.recall_at_5,
            "recall_at_20": self.recall_at_20,
            "mrr_at_20": self.mrr_at_20,
            "gold_count": self.gold_count,
            "planner_args": self.planner_args,
            "gold_titles": self.gold_titles,
            "default_titles": self.default_titles,
            "probe_titles": self.probe_titles,
            "tool_output": self.tool_output,
            "final_answer": self.final_answer,
            "error_message": self.error_message,
        }


def ensure_litsearch_query_downloaded() -> Path:
    LITSEARCH_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_QUERY_FILE.exists():
        return RAW_QUERY_FILE

    hf_hub_download(
        repo_id="princeton-nlp/LitSearch",
        repo_type="dataset",
        filename="query/full-00000-of-00001.parquet",
        local_dir=str(LITSEARCH_DIR),
    )
    return RAW_QUERY_FILE


def _query_source_bucket(query_set: str) -> str:
    return "inline_citation" if str(query_set).startswith("inline_") else "author_written"


def _complexity_score(query: str) -> int:
    text = str(query or "").strip()
    lowered = text.lower()
    score = len(COMPLEXITY_PATTERN.findall(lowered))
    word_count = len(lowered.split())
    if word_count >= 18:
        score += 1
    if word_count >= 28:
        score += 1
    if "?" in text:
        score += 1
    return score


def _complexity_bucket(score: int) -> str:
    if score >= 4:
        return "hard"
    if score >= 2:
        return "medium"
    return "simple"


def _primary_capability_type(query: str, specificity: int, complexity_score: int) -> str:
    if RESOURCE_PATTERN.search(query):
        return "resource_lookup"
    if BENCHMARK_PATTERN.search(query):
        return "benchmark_eval_lookup"
    if ANALYSIS_PATTERN.search(query):
        return "analysis_or_detection_lookup"
    if specificity == 1 and complexity_score >= 3:
        return "constraint_heavy_lookup"
    return "method_topic_lookup"


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(title or "").lower())


def _titles_match(left: str, right: str) -> bool:
    normalized_left = _normalize_title(left)
    normalized_right = _normalize_title(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return True
    return SequenceMatcher(None, normalized_left, normalized_right).ratio() >= 0.94


def _extract_ranked_titles(tool_output: str) -> list[str]:
    return [match.group(1).strip() for match in RANKED_TITLE_PATTERN.finditer(str(tool_output or ""))]


def _max_attempt_number(tool_output: str) -> int:
    attempts = [
        int(match.group(1))
        for match in re.finditer(r"attempt=(\d+)", str(tool_output or ""))
    ]
    return max(attempts) if attempts else 0


def _match_ranks(predicted_titles: list[str], gold_titles: list[str]) -> list[int]:
    matched_ranks: list[int] = []
    remaining_gold = list(gold_titles)
    for rank, predicted_title in enumerate(predicted_titles, start=1):
        match_index = next(
            (
                index
                for index, gold_title in enumerate(remaining_gold)
                if _titles_match(predicted_title, gold_title)
            ),
            None,
        )
        if match_index is None:
            continue
        matched_ranks.append(rank)
        remaining_gold.pop(match_index)
    return matched_ranks


def _query_recall_at_k(predicted_titles: list[str], gold_titles: list[str], k: int) -> float:
    if not gold_titles:
        return 0.0
    matched = _match_ranks(predicted_titles[:k], gold_titles)
    return len(matched) / len(gold_titles)


def _hit_at_k(predicted_titles: list[str], gold_titles: list[str], k: int) -> float:
    return 1.0 if _match_ranks(predicted_titles[:k], gold_titles) else 0.0


def _mrr_at_k(predicted_titles: list[str], gold_titles: list[str], k: int) -> float:
    ranks = _match_ranks(predicted_titles[:k], gold_titles)
    return 1.0 / min(ranks) if ranks else 0.0


def _batched(items: list[int], size: int) -> list[list[int]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def ensure_gold_cache(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    LITSEARCH_DIR.mkdir(parents=True, exist_ok=True)
    if GOLD_CACHE_FILE.exists():
        with GOLD_CACHE_FILE.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
    else:
        cache = {}

    requested_ids = sorted({int(corpus_id) for ids in frame["corpusids"] for corpus_id in ids})
    missing_ids = [corpus_id for corpus_id in requested_ids if str(corpus_id) not in cache]
    if not missing_ids:
        return cache

    session = requests.Session()
    for batch in _batched(missing_ids, DEFAULT_BATCH_SIZE):
        payload = {"ids": [f"CorpusID:{corpus_id}" for corpus_id in batch]}
        for attempt in range(3):
            try:
                response = session.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": S2_FIELDS},
                    json=payload,
                    timeout=40,
                )
                if response.status_code == 429:
                    time.sleep(2 + attempt)
                    continue
                response.raise_for_status()
                rows = response.json()
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    external_ids = row.get("externalIds") or {}
                    corpus_id = external_ids.get("CorpusId")
                    if corpus_id is None:
                        continue
                    cache[str(corpus_id)] = {
                        "title": row.get("title") or "",
                        "year": row.get("year"),
                        "externalIds": external_ids,
                    }
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 + attempt)

    with GOLD_CACHE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)
    return cache


def build_enriched_queries() -> list[dict[str, Any]]:
    parquet_path = ensure_litsearch_query_downloaded()
    frame = pd.read_parquet(parquet_path)
    gold_cache = ensure_gold_cache(frame)

    records: list[dict[str, Any]] = []
    with ENRICHED_QUERY_FILE.open("w", encoding="utf-8") as handle:
        for index, row in frame.reset_index(drop=True).iterrows():
            query = str(row["query"])
            complexity_score = _complexity_score(query)
            gold_titles = [
                str(gold_cache.get(str(int(corpus_id)), {}).get("title") or "").strip()
                for corpus_id in row["corpusids"]
            ]
            gold_titles = [title for title in gold_titles if title]
            record = {
                "query_id": f"litsearch_{index:04d}",
                "query": query,
                "query_set": str(row["query_set"]),
                "query_source_bucket": _query_source_bucket(str(row["query_set"])),
                "specificity": int(row["specificity"]),
                "specificity_bucket": "specific" if int(row["specificity"]) == 1 else "broad",
                "quality": int(row["quality"]),
                "quality_bucket": f"quality_{int(row['quality'])}",
                "complexity_score": complexity_score,
                "complexity_bucket": _complexity_bucket(complexity_score),
                "capability_type": _primary_capability_type(
                    query,
                    int(row["specificity"]),
                    complexity_score,
                ),
                "gold_corpusids": [int(corpus_id) for corpus_id in row["corpusids"]],
                "gold_titles": gold_titles,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


def _select_cases(
    records: list[dict[str, Any]],
    max_cases: int,
    max_cases_per_capability: int,
    seed: int,
) -> list[dict[str, Any]]:
    chosen = list(records)
    rng = random.Random(seed)

    if max_cases_per_capability > 0:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in chosen:
            grouped[record["capability_type"]].append(record)

        chosen = []
        for capability in sorted(grouped):
            bucket = list(grouped[capability])
            rng.shuffle(bucket)
            chosen.extend(bucket[:max_cases_per_capability])

    if max_cases > 0 and len(chosen) > max_cases:
        rng.shuffle(chosen)
        chosen = chosen[:max_cases]

    return sorted(chosen, key=lambda item: item["query_id"])


def _prepare_runtime_environment() -> None:
    load_dotenv()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["STORAGE_BACKEND"] = "sqlite"
    os.environ["SQLITE_STORAGE_DB_PATH"] = str(RUNTIME_DIR / "litsearch_eval_storage.db")
    os.environ["AGENT_CHECKPOINT_DB_PATH"] = str(RUNTIME_DIR / "litsearch_eval_checkpoint.db")
    os.environ["VECTOR_DB_DIR"] = str(RUNTIME_DIR / "vector_db_storage")


def _build_agent_for_eval():
    _prepare_runtime_environment()

    import conversation_memory

    conversation_memory.load_turn_memories = lambda *args, **kwargs: []
    conversation_memory.search_turn_memories = lambda *args, **kwargs: []
    conversation_memory.store_turn_memory = lambda *args, **kwargs: None

    import tools

    tools.paper_already_indexed = lambda *args, **kwargs: True
    tools.download_pdf = lambda *args, **kwargs: None

    from agent import create_research_agent

    return create_research_agent(), tools


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content or "")


def _probe_structured_retrieval(
    tools_module: Any,
    planner_args: dict[str, Any],
    probe_k: int,
) -> tuple[list[str], bool]:
    bounded_count = max(1, int(planner_args.get("count") or 1), probe_k)
    # bounded_rewrite_limit = max(0, int(planner_args.get("rewrite_limit") or 0))
    bounded_rewrite_limit = max(0, int(planner_args.get("rewrite_limit") or 3))
    min_relevance = float(planner_args.get("min_relevance_score") or 0.20)
#=============原代码=========================
    # base_filters = {
    #     "question": str(planner_args.get("question") or "").strip(),
    #     "topic": str(planner_args.get("topic") or "").strip() or None,
    #     "paper_title": str(planner_args.get("paper_title") or "").strip() or None,
    #     "author": str(planner_args.get("author") or "").strip() or None,
    #     "year": int(planner_args["year"]) if planner_args.get("year") else None,
    #     "category": str(planner_args.get("category") or "").strip() or None,
    #     "category_strict": bool(planner_args.get("category_strict")),
    #     "citation_count": max(0, int(planner_args.get("citation_count") or 0)),
    #     "count": bounded_count,
    #     "sort_mode": str(planner_args.get("sort_mode") or "relevance_then_recency"),
    #     "rewrite_limit": bounded_rewrite_limit,
    #     "min_relevance_score": min_relevance,
    #     "query_variants": [],
    #     "latent_clues": [],
    # }
    # base_filters = tools_module._bootstrap_search_filters(base_filters)
#=========================================================

    #========新===================
    # 新增中文判断工具
    def has_chinese(text: str) -> bool:
        return any("\u4e00" <= c <= "\u9fff" for c in str(text or ""))

    # 取出原始topic
    raw_topic = str(planner_args.get("topic") or "").strip() or None
    search_topic = raw_topic

    # 中文topic自动翻译成英文用于检索（仅用户中文输入场景才会走到这里）
    if raw_topic and has_chinese(raw_topic):
        # 导入翻译函数与planner llm
        from agent import translate_cn_topic_to_en, planner_llm
        en_topic = translate_cn_topic_to_en(raw_topic, planner_llm)
        search_topic = en_topic
        print(f"[TRANSLATE LOG] CN topic → EN search keyword | raw:{raw_topic} | trans:{en_topic}")

    # 构建检索过滤器，topic使用翻译后的英文关键词
    base_filters = {
        "question": str(planner_args.get("question") or "").strip(),
        "topic": search_topic,
        "paper_title": str(planner_args.get("paper_title") or "").strip() or None,
        "author": str(planner_args.get("author") or "").strip() or None,
        "year": int(planner_args["year"]) if planner_args.get("year") else None,
        "category": str(planner_args.get("category") or "").strip() or None,
        "category_strict": bool(planner_args.get("category_strict")),
        "citation_count": max(0, int(planner_args.get("citation_count") or 0)),
        "count": bounded_count,
        "sort_mode": str(planner_args.get("sort_mode") or "relevance_then_recency"),
        "rewrite_limit": bounded_rewrite_limit,
        "min_relevance_score": min_relevance,
        "query_variants": [],
        "latent_clues": [],
    }

    base_filters = tools_module._bootstrap_search_filters(base_filters)

    # ========== 这里插入两段优化代码 ==========
    q_text = base_filters["question"].lower()

    # 1. 自动填充NLP/LLM分类，兼容单复数
    llm_nlp_keywords = ["llm", "language model", "explanation", "finetuning", "nlp", "natural language"]
    has_llm_key = any(k in q_text for k in llm_nlp_keywords)
    if has_llm_key and not base_filters["category"]:
        base_filters["category"] = "Natural language processing, Large language model"

    # 2. 评测测量类问题强制纯相关性排序，弱化年份
    eval_keywords = ["measure", "evaluation", "helpful", "faithful", "objective evaluation"]
    is_eval_query = any(k in q_text for k in eval_keywords)
    if is_eval_query:
        base_filters["sort_mode"] = "relevance_only"
        # 关键新增：评测查询清空分类，避免arxiv超长查询503报错
        base_filters["category"] = None
    # ==========================================

    print(f"[PROBE_FILTER_DEBUG] hit_nlp={has_llm_key}, hit_eval={is_eval_query}, final_category={base_filters['category']}, final_sort={base_filters['sort_mode']}")
    pool_size = tools_module._candidate_pool_size(bounded_count)
    if tools_module._is_title_seeking_question(base_filters["question"]):
        pool_size = max(pool_size, 24)

    search_filters = deepcopy(base_filters)

    final_ranked: list[dict[str, Any]] = []
    rewrite_used = False

    for attempt in range(bounded_rewrite_limit + 1):
        _, candidates = tools_module._search_candidates(search_filters, pool_size)
        if not candidates and search_filters["category"] and not search_filters.get("category_strict"):
            relaxed_filters = deepcopy(search_filters)
            relaxed_filters["category"] = None
            _, candidates = tools_module._search_candidates(relaxed_filters, pool_size)
            if candidates:
                search_filters = relaxed_filters
        if search_filters["citation_count"] > 0:
            tools_module._enrich_candidates_with_citations(
                candidates,
                limit=min(len(candidates), 10),
            )

        ranked = tools_module._score_candidates(
            base_filters["question"],
            search_filters,
            candidates,
        )
        final_ranked = ranked
        if not tools_module._should_rewrite(ranked, search_filters, min_relevance):
            break
        if attempt >= bounded_rewrite_limit:
            break

        rewritten_filters = tools_module._rewrite_search_filters(
            base_filters["question"],
            search_filters,
            ranked,
        )
        if rewritten_filters == search_filters:
            break
        search_filters = rewritten_filters
        rewrite_used = True

    selected = tools_module._finalize_ranked_candidates(
        base_filters["question"],
        search_filters,
        final_ranked,
        bounded_count,
    )
    return [candidate["title"] for candidate in selected], rewrite_used


def _invoke_agent_case(
    agent: Any,
    tools_module: Any,
    record: dict[str, Any],
    probe_k: int,
    run_id: str,
    enable_probe: bool = True,
) -> CaseOutcome:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    started_at = time.perf_counter()
    planner_args: dict[str, Any] = {}
    tool_output = ""
    final_answer = ""
    default_titles: list[str] = []
    probe_titles: list[str] = []
    wrong_tool_name: str | None = None

    try:
        # 评测专用元数据，区分批量测试用例
        eval_meta = {
            "langfuse_tags": ["LitSearch-批量评测用例"],
            "langfuse_user_id": "eval_offline",
            "langfuse_session_id": f"eval_run_{run_id}",
            "query_id": record["query_id"],
            "capability_type": record["capability_type"],
            "gold_count": len(record["gold_titles"])
        }
        # 合并原有config + 追踪回调
        trace_config = {
            "configurable": {"thread_id": f"{record['query_id']}__{run_id}"},
            "callbacks": [langfuse_callback],
            "metadata": eval_meta
        }

        response = agent.invoke(
            {
                "input": record["query"],
                "user_id": "litsearch_eval",
                "thread_id": f"{record['query_id']}__{run_id}",
                "messages": [HumanMessage(content=record["query"])],
                "metrics": {},
            },
            config=trace_config,
        )
        latency = time.perf_counter() - started_at

        messages = list(response.get("messages", []))
        tool_call_message = next(
            (
                message
                for message in messages
                if isinstance(message, AIMessage) and getattr(message, "tool_calls", None)
            ),
            None,
        )
        tool_message = next(
            (
                message
                for message in messages
                if isinstance(message, ToolMessage)
            ),
            None,
        )
        final_ai_message = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None)
            ),
            None,
        )

        tool_called = False
        if tool_call_message is not None:
            first_call = tool_call_message.tool_calls[0]
            #==================
            # 安全兜底，不存在args就给空字典
            raw_args = first_call.get("args", {})
            #==================
            planner_args = dict(raw_args) if raw_args is not None else {}

            # 2. name 单独兜底，不存在name就赋值空字符串
            call_name = first_call.get("name", "")

            tool_called = call_name == "arxiv_research_tool"
            wrong_tool_name = None if tool_called else call_name

        tool_output = _stringify_content(getattr(tool_message, "content", ""))
        final_answer = _stringify_content(getattr(final_ai_message, "content", ""))
        default_titles = _extract_ranked_titles(tool_output)

        rewrite_used = _max_attempt_number(tool_output) > 1
        if enable_probe and tool_called and planner_args:
            probe_titles, probe_rewrite_used = _probe_structured_retrieval(
                tools_module,
                planner_args,
                probe_k=probe_k,
            )
            rewrite_used = rewrite_used or probe_rewrite_used
        else:
            probe_titles = list(default_titles)

        # ========== 替换成这段方案B上报代码，删掉get_current_trace相关 ==========
        # 提前一次性计算指标，复用
        hit1 = _hit_at_k(default_titles, record["gold_titles"], 1)
        hit3 = _hit_at_k(default_titles, record["gold_titles"], 3)
        r5 = _query_recall_at_k(probe_titles, record["gold_titles"], 5)
        r20 = _query_recall_at_k(probe_titles, record["gold_titles"], 20)
        mrr20 = _mrr_at_k(probe_titles, record["gold_titles"], 20)

        # 手动绑定唯一trace ID，和agent thread_id统一
        trace_id = f"{record['query_id']}__{run_id}"

        # 上传所有评测分数
        langfuse_client.create_score(trace_id=trace_id, name="top1_hit", value=hit1)
        langfuse_client.create_score(trace_id=trace_id, name="top3_hit", value=hit3)
        langfuse_client.create_score(trace_id=trace_id, name="recall_at_5", value=r5)
        langfuse_client.create_score(trace_id=trace_id, name="recall_at_20", value=r20)
        langfuse_client.create_score(trace_id=trace_id, name="mrr_at_20", value=mrr20)
        langfuse_client.create_score(trace_id=trace_id, name="latency_seconds", value=latency)
        langfuse_client.create_score(trace_id=trace_id, name="rewrite_used", value=1.0 if rewrite_used else 0.0)
        langfuse_client.create_score(trace_id=trace_id, name="search_tool_called", value=1.0 if tool_called else 0.0)

        # =======================================================

        return CaseOutcome(
            query_id=record["query_id"],
            query=record["query"],
            query_set=record["query_set"],
            specificity=record["specificity"],
            quality=record["quality"],
            specificity_bucket=record["specificity_bucket"],
            query_source_bucket=record["query_source_bucket"],
            quality_bucket=record["quality_bucket"],
            capability_type=record["capability_type"],
            complexity_bucket=record["complexity_bucket"],
            complexity_score=record["complexity_score"],
            tool_called=tool_called,
            wrong_tool_name=wrong_tool_name,
            crashed=False,
            latency_seconds=latency,
            no_result="No papers were found" in tool_output or not default_titles,
            rewrite_used=rewrite_used,
            top1_hit=hit1,
            top3_hit=hit3,
            recall_at_5=r5,
            recall_at_20=r20,
            mrr_at_20=mrr20,
            gold_count=len(record["gold_titles"]),
            planner_args=planner_args,
            gold_titles=record["gold_titles"],
            default_titles=default_titles,
            probe_titles=probe_titles[:probe_k],
            tool_output=tool_output,
            final_answer=final_answer,
        )

    except Exception as exc:
        latency = time.perf_counter() - started_at
        trace_id = f"{record['query_id']}__{run_id}"
        # 标记崩溃
        langfuse_client.create_score(trace_id=trace_id, name="crashed", value=1.0)

        return CaseOutcome(
            query_id=record["query_id"],
            query=record["query"],
            query_set=record["query_set"],
            specificity=record["specificity"],
            quality=record["quality"],
            specificity_bucket=record["specificity_bucket"],
            query_source_bucket=record["query_source_bucket"],
            quality_bucket=record["quality_bucket"],
            capability_type=record["capability_type"],
            complexity_bucket=record["complexity_score"],
            complexity_score=record["complexity_score"],
            tool_called=False,
            wrong_tool_name=None,
            crashed=True,
            latency_seconds=latency,
            no_result=True,
            rewrite_used=False,
            top1_hit=0.0,
            top3_hit=0.0,
            recall_at_5=0.0,
            recall_at_20=0.0,
            mrr_at_20=0.0,
            gold_count=len(record["gold_titles"]),
            planner_args={},
            gold_titles=record["gold_titles"],
            default_titles=[],
            probe_titles=[],
            tool_output="",
            final_answer="",
            error_message=f"{type(exc).__name__}: {exc}",
        )

def _aggregate(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    count = len(outcomes)
    if count == 0:
        return {"cases": 0}

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "cases": count,
        "search_tool_rate": mean([1.0 if item.tool_called else 0.0 for item in outcomes]),
        "crash_rate": mean([1.0 if item.crashed else 0.0 for item in outcomes]),
        "no_result_rate": mean([1.0 if item.no_result else 0.0 for item in outcomes]),
        "rewrite_usage_rate": mean([1.0 if item.rewrite_used else 0.0 for item in outcomes]),
        "top1_hit_rate": mean([item.top1_hit for item in outcomes]),
        "top3_hit_rate": mean([item.top3_hit for item in outcomes]),
        "recall_at_5": mean([item.recall_at_5 for item in outcomes]),
        "recall_at_20": mean([item.recall_at_20 for item in outcomes]),
        "mrr_at_20": mean([item.mrr_at_20 for item in outcomes]),
        "avg_latency_seconds": mean([item.latency_seconds for item in outcomes]),
    }


def _group_summary(outcomes: list[CaseOutcome], field_name: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[CaseOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[str(getattr(outcome, field_name))].append(outcome)
    return {key: _aggregate(value) for key, value in sorted(grouped.items())}


def _print_summary(label: str, summary: dict[str, Any]) -> None:
    print(f"\n[{label}]")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def run_eval(
    max_cases: int,
    max_cases_per_capability: int,
    seed: int,
    probe_k: int,
    enable_probe: bool = True,
) -> dict[str, Any]:
    run_id = _timestamp()
    records = build_enriched_queries()
    selected_records = _select_cases(
        records,
        max_cases=max_cases,
        max_cases_per_capability=max_cases_per_capability,
        seed=seed,
    )

    print(f"[litsearch] prepared_records={len(records)}")
    print(f"[litsearch] selected_records={len(selected_records)}")
    print(f"[litsearch] enriched_queries={ENRICHED_QUERY_FILE}")
    print(f"[litsearch] gold_cache={GOLD_CACHE_FILE}")

    agent, tools_module = _build_agent_for_eval()
    outcomes: list[CaseOutcome] = []

    for index, record in enumerate(selected_records[:1], start=1):
        print(
            f"[case {index}/{len(selected_records)}] start {record['query_id']} "
            f"bucket={record['capability_type']} probe={enable_probe}",
            flush=True,
        )
        outcome = _invoke_agent_case(
            agent,
            tools_module,
            record,
            probe_k=probe_k,
            run_id=run_id,
            enable_probe=enable_probe,
        )
        outcomes.append(outcome)
        print(
            f"[case {index}/{len(selected_records)}] {record['query_id']} "
            f"tool={outcome.tool_called} top3={outcome.top3_hit:.0f} "
            f"r20={outcome.recall_at_20:.2f} latency={outcome.latency_seconds:.2f}s "
            f"bucket={record['capability_type']} probe={enable_probe}"
        )

    overall = _aggregate(outcomes)
    by_specificity = _group_summary(outcomes, "specificity_bucket")
    by_query_source = _group_summary(outcomes, "query_source_bucket")
    by_query_set = _group_summary(outcomes, "query_set")
    by_quality = _group_summary(outcomes, "quality_bucket")
    by_capability = _group_summary(outcomes, "capability_type")
    by_complexity = _group_summary(outcomes, "complexity_bucket")

    report = {
        "generated_at": run_id,
        "dataset": {
            "raw_query_file": str(RAW_QUERY_FILE),
            "enriched_query_file": str(ENRICHED_QUERY_FILE),
            "gold_cache_file": str(GOLD_CACHE_FILE),
            "selected_cases": len(selected_records),
            "total_cases": len(records),
        },
        "eval_mode": {
            "storage_backend": "sqlite",
            "checkpoint_path": str(RUNTIME_DIR / "litsearch_eval_checkpoint.db"),
            "storage_path": str(RUNTIME_DIR / "litsearch_eval_storage.db"),
            "vector_db_dir": str(RUNTIME_DIR / "vector_db_storage"),
            "non_ingest_mode": True,
            "memory_disabled": True,
            "probe_k": probe_k,
            "enable_probe": enable_probe,
        },
        "overall": overall,
        "by_specificity": by_specificity,
        "by_query_source": by_query_source,
        "by_query_set": by_query_set,
        "by_quality": by_quality,
        "by_capability_type": by_capability,
        "by_complexity": by_complexity,
        "cases": [outcome.to_dict() for outcome in outcomes],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = report["generated_at"]
    report_json = REPORT_DIR / f"litsearch_eval_report_{timestamp}.json"
    report_jsonl = REPORT_DIR / f"litsearch_eval_cases_{timestamp}.jsonl"
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with report_jsonl.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome.to_dict(), ensure_ascii=False) + "\n")

    print(f"\n[litsearch] report_json={report_json}")
    print(f"[litsearch] report_jsonl={report_jsonl}")
    _print_summary("overall", overall)
    _print_summary("by_specificity", by_specificity)
    _print_summary("by_capability_type", by_capability)

    # 新增：批量评测全部跑完强制上传所有Trace
    langfuse_client.flush()
    return report



def main() -> None:
    parser = argparse.ArgumentParser(description="LitSearch live evaluation for the research agent")
    parser.add_argument("--prepare-only", action="store_true", help="download and enrich LitSearch without running the agent")
    parser.add_argument("--max-cases", type=int, default=0, help="overall cap, 0 means use all selected cases")
    parser.add_argument(
        "--max-cases-per-capability",
        type=int,
        default=0,
        help="balanced cap per capability bucket, 0 means disabled",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probe-k", type=int, default=DEFAULT_PROBE_K)
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="skip the extra retrieval probe and score only from the agent-returned ranked results",
    )
    args = parser.parse_args()

    records = build_enriched_queries()
    print(f"[litsearch] raw_query_file={RAW_QUERY_FILE}")
    print(f"[litsearch] enriched_query_file={ENRICHED_QUERY_FILE}")
    print(f"[litsearch] gold_cache_file={GOLD_CACHE_FILE}")
    print(f"[litsearch] records={len(records)}")

    if args.prepare_only:
        return

    run_eval(
        max_cases=max(0, args.max_cases),
        max_cases_per_capability=max(0, args.max_cases_per_capability),
        seed=args.seed,
        probe_k=max(5, args.probe_k),
        enable_probe=not args.lightweight,
    )


if __name__ == "__main__":
    main()
