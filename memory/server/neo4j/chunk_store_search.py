"""Vector search helpers for Chunk keyword embeddings."""

from __future__ import annotations

from typing import Any

from neo4j import ManagedTransaction

from server.embedding_keywords import embed_keywords, keyword_embedding_index_name


def keyword_vector_search(
    transaction: ManagedTransaction,
    *,
    query_keywords: list[str],
    run_id: str | None = None,
    document_name: str | None = None,
    limit: int = 20,
    embedding_config_override: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    vectors = embed_keywords(query_keywords, config_override=embedding_config_override)
    if not vectors:
        return []
    index_name = keyword_embedding_index_name(config_override=embedding_config_override)
    candidate_k = max(20, int(limit) * 5)
    rows: list[dict[str, object]] = []
    for query_keyword, vector in zip(query_keywords, vectors, strict=False):
        result = transaction.run(
            f"""
            CALL db.index.vector.queryNodes($index_name, $candidate_k, $vector)
            YIELD node, score
            WHERE node:KeywordNode
              AND ($run_id IS NULL OR node.run_id = $run_id)
              AND ($document_name IS NULL OR node.document_name = $document_name)
            MATCH (chunk:Chunk)-[:HAS_KEYWORD]->(node)
            WHERE chunk.run_id = node.run_id
              AND chunk.document_name = node.document_name
            WITH chunk, max(score) AS score, collect(DISTINCT node.keyword) AS matched_keywords
            RETURN
                chunk.run_id AS run_id,
                chunk.document_name AS document_name,
                coalesce(chunk.chunk_id, chunk.id) AS chunk_id,
                chunk.chunk_index AS chunk_index,
                chunk.summary AS summary,
                matched_keywords AS matched_keywords,
                score
            ORDER BY score DESC
            LIMIT $limit
            """,
            index_name=index_name,
            limit=max(1, int(limit)),
            candidate_k=candidate_k,
            vector=[float(value) for value in vector],
            run_id=run_id,
            document_name=document_name,
        )
        for record in result:
            rows.append(
                {
                    "query_keyword": str(query_keyword),
                    "run_id": str(record["run_id"] or ""),
                    "document_name": str(record["document_name"] or ""),
                    "id": str(record["chunk_id"] or ""),
                    "chunk_index": int(record["chunk_index"] or 0),
                    "summary": str(record["summary"] or ""),
                    "matched_keywords": [str(keyword) for keyword in (record["matched_keywords"] or [])],
                    "score": float(record["score"] or 0.0),
                }
            )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows[: max(1, int(limit))]
