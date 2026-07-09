import json
import math
import os
import re
from collections import Counter
from typing import Any

import requests
from langchain_core.documents import Document
from pypdf import PdfReader
from sqlalchemy import text

from database import DATA_DIR,get_vector_db
from research_harness import CURRENT_YEAR
from storage import ensure_structured_storage_ready, get_structured_store_engine, now_utc_timestamp

DOWNLOAD_TIMEOUT_SECONDS = 15
MIN_FULLTEXT_LENGTH = 800
MIN_SENTENCES_PER_CHUNK = 2
MAX_SENTENCES_PER_CHUNK = 10
MIN_CHUNK_CHARS = 450
MAX_CHUNK_CHARS = 2200
SEMANTIC_BOUNDARY_FLOOR = 0.42
SEMANTIC_DROP_MARGIN = 0.10
EMBED_BATCH_SIZE = 48

DENSE_RETRIEVAL_K = 12
SPARSE_RETRIEVAL_K = 12
FUSED_RETRIEVAL_K = 10
FINAL_RETRIEVAL_K = 4
RRF_K = 60

WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")


def _safe_filename(title: str) -> str:
    return "".join(char for char in title if char.isalnum() or char in " -_").strip() or "paper"


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_PATTERN.findall(str(text or ""))]


def _token_overlap_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    text_tokens = set(_tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalize_score_map(candidates: list[dict[str, Any]], field_name: str) -> dict[str, float]:
    values = [float(candidate.get(field_name, 0.0) or 0.0) for candidate in candidates]
    if not values:
        return {}
    high = max(values)
    low = min(values)
    if math.isclose(high, low):
        return {candidate["chunk_id"]: (1.0 if high > 0 else 0.0) for candidate in candidates}
    return {
        candidate["chunk_id"]: (float(candidate.get(field_name, 0.0) or 0.0) - low) / (high - low)
        for candidate in candidates
    }


def _recency_signal(year_value: Any) -> float:
    try:
        year = int(year_value)
    except (TypeError, ValueError):
        return 0.0
    age = max(CURRENT_YEAR - year, 0)
    return max(0.0, 1.0 - min(age, 12) / 12.0)


def _embed_texts_in_batches(embeddings: Any, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors.extend(embeddings.embed_documents(batch))
    return vectors


def download_pdf(candidate: dict[str, Any]) -> dict[str, Any] | None:
    result = candidate.get("result")
    pdf_url = candidate.get("pdf_url") or getattr(result, "pdf_url", None)
    if not pdf_url:
        return None
    safe_title = _safe_filename(candidate["title"])
    path = os.path.join(DATA_DIR, f"{safe_title}.pdf")
    try:
        if not os.path.exists(path):
            response = requests.get(pdf_url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
            if response.status_code != 200:
                return None
            if not bytes(response.content[:5]).startswith(b"%PDF-"):
                return None
            with open(path, "wb") as handle:
                handle.write(response.content)
        return {"candidate": candidate, "path": path, "safe_title": safe_title}
    except Exception:
        return None


def _normalize_pdf_text(text: str) -> str:
    normalized = str(text or "").replace("\r", "\n")
    normalized = re.sub(r"-\s*\n\s*", "", normalized)
    normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized)
    normalized = re.sub(r"\n{2,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\s+\n", "\n", normalized)
    normalized = re.sub(r"\n\s+", "\n", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized.strip()


def _extract_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        return ""
    return _normalize_pdf_text("\n".join(page for page in pages if page))


def _split_text_into_sentences(text: str) -> list[str]:
    if not text:
        return []
    working = text.replace("\n\n", ". ")
    pieces = SENTENCE_SPLIT_PATTERN.split(working)
    sentences: list[str] = []
    for piece in pieces:
        cleaned = " ".join(piece.split()).strip()
        if not cleaned:
            continue
        if len(cleaned) < 35 and sentences:
            sentences[-1] = f"{sentences[-1]} {cleaned}".strip()
            continue
        sentences.append(cleaned)
    return sentences


def _merge_small_chunks(chunks: list[str]) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        cleaned = " ".join(chunk.split()).strip()
        if not cleaned:
            continue
        if merged and len(cleaned) < MIN_CHUNK_CHARS // 2:
            merged[-1] = f"{merged[-1]} {cleaned}"
            continue
        if merged and len(merged[-1]) < MIN_CHUNK_CHARS // 2:
            merged[-1] = f"{merged[-1]} {cleaned}"
            continue
        merged.append(cleaned)
    return merged


def _semantic_chunk_sentences(sentences: list[str], embeddings: Any) -> list[str]:
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    sentence_vectors = _embed_texts_in_batches(embeddings, sentences)
    adjacent_scores = [
        _cosine_similarity(sentence_vectors[index - 1], sentence_vectors[index])
        for index in range(1, len(sentence_vectors))
    ]

    mean_similarity = _average(adjacent_scores)
    variance = _average([(score - mean_similarity) ** 2 for score in adjacent_scores])
    std_similarity = math.sqrt(variance)
    adaptive_threshold = max(
        SEMANTIC_BOUNDARY_FLOOR,
        min(0.86, mean_similarity - max(0.05, std_similarity * 0.35)),
    )
    drop_margin = max(SEMANTIC_DROP_MARGIN, std_similarity * 0.65)

    chunks: list[str] = []
    current_chunk = [sentences[0]]

    for index in range(1, len(sentences)):
        similarity = adjacent_scores[index - 1]
        next_sentence = sentences[index]
        current_text = " ".join(current_chunk)
        rolling_window = adjacent_scores[max(0, index - 4) : index]
        rolling_average = _average(rolling_window) if rolling_window else mean_similarity

        should_force_split = (
            len(current_chunk) >= MAX_SENTENCES_PER_CHUNK
            or len(current_text) + 1 + len(next_sentence) > MAX_CHUNK_CHARS
        )
        should_semantic_split = (
            len(current_chunk) >= MIN_SENTENCES_PER_CHUNK
            and len(current_text) >= MIN_CHUNK_CHARS
            and (similarity < adaptive_threshold or similarity < rolling_average - drop_margin)
        )

        if should_force_split or should_semantic_split:
            chunks.append(current_text)
            current_chunk = [next_sentence]
            continue

        current_chunk.append(next_sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return _merge_small_chunks(chunks)


def _chunk_metadata(candidate: dict[str, Any], safe_title: str, path: str, chunk_id: str, chunk_index: int) -> dict[str, Any]:
    return {
        "record_type": "fulltext_chunk",
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "source": safe_title,
        "paper_title": candidate["title"],
        "arxiv_id": candidate["arxiv_id"],
        "year": candidate["year"] or "",
        "authors": ", ".join(candidate["authors"]),
        "categories": ", ".join(candidate["categories"]),
        "pdf_path": path,
        "score": round(candidate["overall_score"], 4),
        "semantic_score": round(candidate["semantic_score"], 4),
        "citation_count": candidate["citation_count"] if candidate["citation_count"] is not None else "",
    }


def build_fulltext_documents(candidate: dict[str, Any], path: str, safe_title: str) -> tuple[list[Document], list[dict[str, Any]]]:
    _, embeddings = get_vector_db()
    full_text = _extract_pdf_text(path)
    if len(full_text) < MIN_FULLTEXT_LENGTH:
        return [], []

    sentences = _split_text_into_sentences(full_text)
    if not sentences:
        return [], []

    chunks = _semantic_chunk_sentences(sentences, embeddings)
    documents: list[Document] = []
    records: list[dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        cleaned_chunk = " ".join(chunk.split()).strip()
        if not cleaned_chunk:
            continue
        chunk_id = f"{candidate['arxiv_id']}::chunk::{chunk_index}"
        metadata = _chunk_metadata(candidate, safe_title, path, chunk_id, chunk_index)
        documents.append(Document(page_content=cleaned_chunk, metadata=metadata))
        records.append(
            {
                "record_type": "fulltext_chunk",
                "chunk_id": chunk_id,
                "arxiv_id": candidate["arxiv_id"],
                "page_content": cleaned_chunk,
                "metadata": metadata,
            }
        )

    return documents, records


def load_sparse_records() -> list[dict[str, Any]]:
    ensure_structured_storage_ready()
    with get_structured_store_engine().connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT chunk_id, arxiv_id, record_type, page_content, metadata
                FROM hybrid_chunks
                WHERE record_type = :record_type
                ORDER BY created_at ASC, chunk_id ASC
                """
            ),
            {"record_type": "fulltext_chunk"},
        ).fetchall()

    records_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        chunk_id = str(row[0] or "")
        if not chunk_id:
            continue
        records_by_id[chunk_id] = {
            "chunk_id": chunk_id,
            "arxiv_id": str(row[1] or ""),
            "record_type": str(row[2] or "fulltext_chunk"),
            "page_content": str(row[3] or ""),
            "metadata": json.loads(row[4]) if row[4] else {},
        }

    return list(records_by_id.values())


def append_sparse_records(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0

    ensure_structured_storage_ready()
    existing_ids = {record["chunk_id"] for record in load_sparse_records()}
    new_records = [record for record in records if record["chunk_id"] not in existing_ids]
    if not new_records:
        return 0

    payloads = [
        {
            "chunk_id": str(record["chunk_id"]),
            "arxiv_id": str(record.get("arxiv_id") or ""),
            "record_type": str(record.get("record_type") or "fulltext_chunk"),
            "page_content": str(record.get("page_content") or ""),
            "metadata": json.dumps(record.get("metadata", {}), ensure_ascii=False),
            "created_at": now_utc_timestamp(),
        }
        for record in new_records
    ]

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

    return len(new_records)


def paper_already_indexed(arxiv_id: str) -> bool:
    ensure_structured_storage_ready()
    with get_structured_store_engine().connect() as connection:
        count = connection.execute(
            text(
                """
                SELECT count(*)
                FROM hybrid_chunks
                WHERE arxiv_id = :arxiv_id AND record_type = :record_type
                """
            ),
            {"arxiv_id": arxiv_id, "record_type": "fulltext_chunk"},
        ).scalar_one()
    return int(count) > 0


def _bm25_score(query_tokens: list[str], document_tokens: list[str], document_frequency: dict[str, int], corpus_size: int, avg_doc_length: float) -> float:
    if not query_tokens or not document_tokens or corpus_size <= 0:
        return 0.0

    term_frequency = Counter(document_tokens)
    document_length = len(document_tokens)
    k1 = 1.5
    b = 0.75
    score = 0.0

    for token in set(query_tokens):
        frequency = term_frequency.get(token, 0)
        if frequency <= 0:
            continue
        df_value = document_frequency.get(token, 0)
        idf = math.log(1.0 + (corpus_size - df_value + 0.5) / (df_value + 0.5))
        denominator = frequency + k1 * (1 - b + b * document_length / max(avg_doc_length, 1.0))
        score += idf * (frequency * (k1 + 1) / denominator)

    return score


def _dense_retrieve(question: str) -> list[dict[str, Any]]:
    db, _ = get_vector_db()
    records_by_id = {record["chunk_id"]: record for record in load_sparse_records()}

    try:
        raw_results = db.similarity_search_with_score(
            question,
            k=DENSE_RETRIEVAL_K,
            filter={"record_type": "fulltext_chunk"},
        )
    except TypeError:
        raw_results = db.similarity_search_with_score(question, k=DENSE_RETRIEVAL_K)

    dense_results: list[dict[str, Any]] = []
    for rank, (doc, distance) in enumerate(raw_results, start=1):
        if doc.metadata.get("record_type") != "fulltext_chunk":
            continue
        chunk_id = str(doc.metadata.get("chunk_id") or "")
        if not chunk_id:
            continue
        record = records_by_id.get(chunk_id)
        dense_results.append(
            {
                "chunk_id": chunk_id,
                "page_content": record.get("page_content", doc.page_content) if record else doc.page_content,
                "metadata": record.get("metadata", doc.metadata) if record else doc.metadata,
                "dense_score": 1.0 / (1.0 + max(float(distance), 0.0)),
                "sparse_score": 0.0,
                "fusion_score": 0.0,
                "dense_rank": rank,
            }
        )

    return dense_results


def _sparse_retrieve(question: str) -> list[dict[str, Any]]:
    records = load_sparse_records()
    query_tokens = _tokenize(question)
    if not records or not query_tokens:
        return []

    tokenized = {record["chunk_id"]: _tokenize(record["page_content"]) for record in records}
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized.values():
        document_frequency.update(set(tokens))
    avg_doc_length = sum(len(tokens) for tokens in tokenized.values()) / len(tokenized)

    scored_results: list[dict[str, Any]] = []
    for record in records:
        chunk_id = record["chunk_id"]
        score = _bm25_score(query_tokens, tokenized.get(chunk_id, []), dict(document_frequency), len(records), avg_doc_length)
        if score <= 0:
            continue
        scored_results.append(
            {
                "chunk_id": chunk_id,
                "page_content": record["page_content"],
                "metadata": record["metadata"],
                "dense_score": 0.0,
                "sparse_score": score,
                "fusion_score": 0.0,
            }
        )

    return sorted(scored_results, key=lambda item: item["sparse_score"], reverse=True)[:SPARSE_RETRIEVAL_K]


def _fuse_retrievals(dense_results: list[dict[str, Any]], sparse_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(dense_results, start=1):
        candidate = fused.setdefault(
            item["chunk_id"],
            {
                "chunk_id": item["chunk_id"],
                "page_content": item["page_content"],
                "metadata": item["metadata"],
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "fusion_score": 0.0,
            },
        )
        candidate["dense_score"] = max(candidate["dense_score"], item["dense_score"])
        candidate["fusion_score"] += 1.0 / (RRF_K + rank)

    for rank, item in enumerate(sparse_results, start=1):
        candidate = fused.setdefault(
            item["chunk_id"],
            {
                "chunk_id": item["chunk_id"],
                "page_content": item["page_content"],
                "metadata": item["metadata"],
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "fusion_score": 0.0,
            },
        )
        candidate["sparse_score"] = max(candidate["sparse_score"], item["sparse_score"])
        candidate["fusion_score"] += 1.0 / (RRF_K + rank)

    return sorted(fused.values(), key=lambda item: item["fusion_score"], reverse=True)[:FUSED_RETRIEVAL_K]


def hybrid_retrieve(question: str, limit: int = FINAL_RETRIEVAL_K) -> list[dict[str, Any]]:
    dense_results = _dense_retrieve(question)
    sparse_results = _sparse_retrieve(question)
    candidates = _fuse_retrievals(dense_results, sparse_results)
    if not candidates:
        return []

    _, embeddings = get_vector_db()
    rerank_texts = [
        f"title: {candidate['metadata'].get('paper_title', '')}\nchunk: {candidate['page_content']}"
        for candidate in candidates
    ]
    try:
        query_vector = embeddings.embed_query(question)
        candidate_vectors = _embed_texts_in_batches(embeddings, rerank_texts)
        semantic_scores = [_cosine_similarity(query_vector, vector) for vector in candidate_vectors]
    except Exception:
        semantic_scores = [_token_overlap_score(question, text) for text in rerank_texts]

    dense_norm = _normalize_score_map(candidates, "dense_score")
    sparse_norm = _normalize_score_map(candidates, "sparse_score")
    fusion_norm = _normalize_score_map(candidates, "fusion_score")

    for candidate, semantic_score in zip(candidates, semantic_scores):
        chunk_id = candidate["chunk_id"]
        candidate["semantic_rerank_score"] = semantic_score
        candidate["final_score"] = round(
            semantic_score * 0.52
            + sparse_norm.get(chunk_id, 0.0) * 0.18
            + dense_norm.get(chunk_id, 0.0) * 0.14
            + fusion_norm.get(chunk_id, 0.0) * 0.08
            + _token_overlap_score(question, candidate["page_content"]) * 0.05
            + _recency_signal(candidate["metadata"].get("year")) * 0.03,
            4,
        )

    return sorted(candidates, key=lambda item: item["final_score"], reverse=True)[:limit]


def format_retrieved_chunk(candidate: dict[str, Any], rank: int) -> str:
    metadata = candidate["metadata"]
    title = metadata.get("paper_title") or metadata.get("source") or "unknown"
    preview = " ".join(candidate["page_content"].split())
    if len(preview) > 700:
        preview = preview[:700].rstrip() + " ..."
    return (
        f"{rank}. {title} | chunk={metadata.get('chunk_index', 'n/a')} | year={metadata.get('year', 'n/a')} | "
        f"final={candidate['final_score']:.3f} | semantic={candidate['semantic_rerank_score']:.3f} | "
        f"dense={candidate.get('dense_score', 0.0):.3f} | sparse={candidate.get('sparse_score', 0.0):.3f}\n"
        f"{preview}"
    )
