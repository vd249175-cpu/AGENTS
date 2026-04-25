"""Read helpers for document-side Chunk records."""

from __future__ import annotations

from neo4j import ManagedTransaction

DEFAULT_EDGE_DISTANCE = 0.3


def normalize_document_name(document_name: str) -> str:
    normalized = document_name.strip().rsplit("/", maxsplit=1)[-1].rsplit("\\", maxsplit=1)[-1]
    if "." in normalized:
        normalized = normalized.rsplit(".", maxsplit=1)[0]
    if not normalized:
        raise ValueError("document_name is required")
    return normalized


def document_exists(transaction: ManagedTransaction, *, document_name: str, run_id: str | None = None) -> bool:
    record = transaction.run(
        """
        MATCH (chunk:Chunk {document_name: $document_name})
        WHERE $run_id IS NULL OR chunk.run_id = $run_id
        RETURN 1 AS found
        LIMIT 1
        """,
        document_name=normalize_document_name(document_name),
        run_id=run_id,
    ).single()
    return record is not None


def document_run_id(transaction: ManagedTransaction, *, document_name: str) -> str | None:
    record = transaction.run(
        """
        MATCH (chunk:Chunk {document_name: $document_name})
        WITH chunk.run_id AS run_id, max(coalesce(chunk.updated_at, chunk.created_at, '')) AS updated_at
        RETURN run_id
        ORDER BY updated_at DESC, run_id
        LIMIT 1
        """,
        document_name=normalize_document_name(document_name),
    ).single()
    if record is None or record["run_id"] is None:
        return None
    return str(record["run_id"])


def list_documents(transaction: ManagedTransaction, *, run_id: str | None = None) -> list[dict[str, object]]:
    result = transaction.run(
        """
        MATCH (chunk:Chunk)
        WHERE $run_id IS NULL OR chunk.run_id = $run_id
        WITH chunk.document_name AS document_name,
             collect(DISTINCT chunk.run_id) AS run_ids,
             count(chunk) AS chunk_count,
             min(chunk.created_at) AS created_at,
             max(coalesce(chunk.updated_at, chunk.created_at)) AS updated_at
        RETURN document_name, run_ids, chunk_count, created_at, updated_at
        ORDER BY updated_at DESC, document_name
        """,
        run_id=run_id,
    )
    return [
        {
            "document_name": str(record["document_name"]),
            "run_ids": [str(item) for item in list(record["run_ids"] or []) if item is not None],
            "chunk_count": int(record["chunk_count"]),
            "created_at": str(record["created_at"] or ""),
            "updated_at": str(record["updated_at"] or ""),
        }
        for record in result
    ]


def list_chunks(transaction: ManagedTransaction, *, document_name: str, run_id: str | None = None) -> list[dict[str, object]]:
    result = transaction.run(
        """
        MATCH (chunk:Chunk {document_name: $document_name})
        WHERE $run_id IS NULL OR chunk.run_id = $run_id
        OPTIONAL MATCH (chunk)-[:HAS_KEYWORD]->(keyword:KeywordNode)
        WHERE keyword.document_name = chunk.document_name
          AND keyword.owner_id = coalesce(chunk.chunk_id, chunk.id)
          AND ($run_id IS NULL OR keyword.run_id = chunk.run_id)
        WITH chunk, keyword
        ORDER BY keyword.keyword_index
        WITH chunk,
             collect(CASE
                 WHEN keyword IS NULL THEN NULL
                 ELSE {
                     keyword_index: keyword.keyword_index,
                     keyword: keyword.keyword,
                     embedding: keyword.embedding,
                     embedding_provider: keyword.embedding_provider,
                     embedding_model: keyword.embedding_model,
                     embedding_dimensions: coalesce(keyword.embedding_dimensions, keyword.dimension)
                 }
             END) AS keyword_items
        OPTIONAL MATCH (prev:Chunk)-[prev_rel]->(chunk)
        WHERE type(prev_rel) <> 'HAS_KEYWORD'
          AND ($run_id IS NULL OR prev.run_id = chunk.run_id)
        WITH chunk,
             keyword_items,
             collect(DISTINCT CASE
                 WHEN prev IS NULL THEN NULL
                 ELSE {
                     relation: type(prev_rel),
                     source_node_id: coalesce(prev.chunk_id, prev.id),
                     source_position: prev.chunk_index,
                     target_node_id: coalesce(chunk.chunk_id, chunk.id),
                     target_position: chunk.chunk_index,
                     dist: coalesce(prev_rel.dist, prev_rel.distance, $default_dist)
                 }
             END) AS incoming_edges
        OPTIONAL MATCH (chunk)-[next_rel]->(next:Chunk)
        WHERE type(next_rel) <> 'HAS_KEYWORD'
          AND ($run_id IS NULL OR next.run_id = chunk.run_id)
        WITH chunk,
             keyword_items,
             incoming_edges,
             collect(DISTINCT CASE
                 WHEN next IS NULL THEN NULL
                 ELSE {
                     relation: type(next_rel),
                     source_node_id: coalesce(chunk.chunk_id, chunk.id),
                     source_position: chunk.chunk_index,
                     target_node_id: coalesce(next.chunk_id, next.id),
                     target_position: next.chunk_index,
                     dist: coalesce(next_rel.dist, next_rel.distance, $default_dist)
                 }
             END) AS outgoing_edges
        RETURN
            chunk.run_id AS run_id,
            coalesce(chunk.chunk_id, chunk.id) AS id,
            chunk.document_name AS document_name,
            chunk.chunk_index AS chunk_index,
            chunk.summary AS summary,
            coalesce(chunk.body, chunk.text, '') AS body,
            chunk.keywords AS keywords,
            keyword_items,
            incoming_edges,
            outgoing_edges,
            chunk.char_start AS char_start,
            chunk.char_end AS char_end,
            chunk.created_at AS created_at,
            chunk.updated_at AS updated_at
        ORDER BY chunk.chunk_index
        """,
        document_name=normalize_document_name(document_name),
        run_id=run_id,
        default_dist=DEFAULT_EDGE_DISTANCE,
    )
    records: list[dict[str, object]] = []
    for record in result:
        keyword_items = [item for item in list(record["keyword_items"] or []) if item is not None]
        keywords = [str(item["keyword"]) for item in keyword_items]
        keyword_vectors = [[float(value) for value in list(item["embedding"] or [])] for item in keyword_items]
        embedding_provider = str(keyword_items[0].get("embedding_provider") or "") if keyword_items else ""
        embedding_model = str(keyword_items[0].get("embedding_model") or "") if keyword_items else ""
        embedding_dimensions = int(keyword_items[0].get("embedding_dimensions") or 0) if keyword_items else 0
        if not keywords:
            keywords = [str(item) for item in list(record["keywords"] or [])]
        records.append(
            {
            "run_id": str(record["run_id"] or ""),
            "id": str(record["id"]),
            "document_name": str(record["document_name"]),
            "chunk_index": int(record["chunk_index"]),
            "summary": str(record["summary"] or ""),
            "body": str(record["body"] or ""),
            "keywords": keywords,
            "keyword_vectors": keyword_vectors,
            "keyword_embedding_provider": embedding_provider,
            "keyword_embedding_model": embedding_model,
            "keyword_embedding_dimensions": embedding_dimensions,
            "edges": _normalize_edges(list(record["incoming_edges"] or []), list(record["outgoing_edges"] or [])),
            "char_start": int(record["char_start"] or 0),
            "char_end": int(record["char_end"] or 0),
            "created_at": str(record["created_at"] or ""),
            "updated_at": str(record["updated_at"] or ""),
        }
        )
    return records


def _normalize_edges(incoming_edges: list[object], outgoing_edges: list[object]) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_edge in [*incoming_edges, *outgoing_edges]:
        if not raw_edge:
            continue
        edge = dict(raw_edge)
        relation = str(edge.get("relation") or "")
        source_id = str(edge.get("source_node_id") or "")
        target_id = str(edge.get("target_node_id") or "")
        if not relation or not source_id or not target_id:
            continue
        key = (relation, source_id, target_id)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "relation": relation,
                "source_node_id": source_id,
                "source_position": int(edge.get("source_position") or 0),
                "target_node_id": target_id,
                "target_position": int(edge.get("target_position") or 0),
                "dist": float(edge.get("dist") or DEFAULT_EDGE_DISTANCE),
            }
        )
    return edges
