"""Read helpers for the migrated Neo4j chunk catalog store."""

from __future__ import annotations

from neo4j import ManagedTransaction


def document_exists(
    transaction: ManagedTransaction,
    *,
    document_name: str,
    run_id: str | None = None,
) -> bool:
    record = transaction.run(
        """
        MATCH (chunk:Chunk {document_name: $document_name})
        WHERE $run_id IS NULL OR chunk.run_id = $run_id
        RETURN 1 AS found
        LIMIT 1
        """,
        document_name=document_name,
        run_id=run_id,
    ).single()
    return record is not None


def list_document_names(transaction: ManagedTransaction, *, run_id: str) -> list[str]:
    result = transaction.run(
        """
        MATCH (chunk:Chunk {run_id: $run_id})
        RETURN DISTINCT chunk.document_name AS document_name
        ORDER BY document_name
        """,
        run_id=run_id,
    )
    return [str(record["document_name"]) for record in result]


def list_chunks(transaction: ManagedTransaction, *, run_id: str, document_name: str) -> list[dict[str, object]]:
    result = transaction.run(
        """
        MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name})
        OPTIONAL MATCH (chunk)-[:HAS_KEYWORD]->(keyword:KeywordNode {run_id: $run_id, document_name: $document_name})
        WITH chunk, keyword
        ORDER BY keyword.keyword_index
        WITH chunk,
             collect(CASE
                 WHEN keyword IS NULL THEN NULL
                 ELSE {
                     keyword_index: keyword.keyword_index,
                     keyword: keyword.keyword,
                     embedding: keyword.embedding
                 }
             END) AS keyword_items
        RETURN
            chunk.chunk_id AS chunk_id,
            chunk.document_name AS document_name,
            chunk.chunk_index AS chunk_index,
            chunk.summary AS summary,
            chunk.keywords AS keywords,
            keyword_items,
            chunk.text AS text,
            chunk.char_start AS char_start,
            chunk.char_end AS char_end,
            chunk.created_at AS created_at
        ORDER BY chunk.chunk_index
        """,
        run_id=run_id,
        document_name=document_name,
    )
    rows: list[dict[str, object]] = []
    for record in result:
        keyword_items = [item for item in list(record["keyword_items"] or []) if item is not None]
        keywords = [str(item["keyword"]) for item in keyword_items]
        keyword_vectors = [[float(value) for value in list(item["embedding"] or [])] for item in keyword_items]
        if not keywords:
            keywords = [str(item) for item in list(record["keywords"] or [])]
        rows.append(
            {
                "id": str(record["chunk_id"]),
                "document_name": str(record["document_name"]),
                "chunk_index": int(record["chunk_index"]),
                "summary": str(record["summary"]),
                "keywords": keywords,
                "keyword_vectors": keyword_vectors,
                "text": str(record["text"]),
                "char_start": int(record["char_start"]),
                "char_end": int(record["char_end"]),
                "created_at": str(record["created_at"]),
            }
        )
    return rows
