"""Write helpers for document-side Chunk records."""

from __future__ import annotations

from datetime import datetime, timezone

from neo4j import ManagedTransaction


def clear_document(transaction: ManagedTransaction, *, document_name: str) -> None:
    transaction.run(
        """
        MATCH (chunk:Chunk {document_name: $document_name})-[:HAS_KEYWORD]->
              (keyword:KeywordNode {document_name: $document_name})
        DETACH DELETE keyword
        """,
        document_name=document_name,
    )
    transaction.run(
        """
        MATCH (chunk:Chunk {document_name: $document_name})
        DETACH DELETE chunk
        """,
        document_name=document_name,
    )


def delete_chunks_not_in_ids(transaction: ManagedTransaction, *, run_id: str, document_name: str, keep_chunk_ids: list[str]) -> None:
    transaction.run(
        """
        MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name})-[:HAS_KEYWORD]->
              (keyword:KeywordNode {run_id: $run_id, document_name: $document_name})
        WHERE NOT coalesce(chunk.chunk_id, chunk.id, '') IN $keep_chunk_ids
        DETACH DELETE keyword
        """,
        run_id=run_id,
        document_name=document_name,
        keep_chunk_ids=keep_chunk_ids,
    )
    transaction.run(
        """
        MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name})
        WHERE NOT coalesce(chunk.chunk_id, chunk.id, '') IN $keep_chunk_ids
        DETACH DELETE chunk
        """,
        run_id=run_id,
        document_name=document_name,
        keep_chunk_ids=keep_chunk_ids,
    )


def upsert_chunk(
    transaction: ManagedTransaction,
    *,
    run_id: str,
    document_name: str,
    chunk_id: str,
    chunk_index: int,
    summary: str,
    body: str,
    keywords: list[str],
    char_start: int,
    char_end: int,
    created_at: str | None,
    keyword_vectors: list[list[float]] | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    chunk_key = f"{run_id}:{document_name}:{chunk_id}"
    transaction.run(
        """
        MERGE (chunk:Chunk {chunk_key: $chunk_key})
        SET
            chunk.run_id = $run_id,
            chunk.document_name = $document_name,
            chunk.chunk_id = $chunk_id,
            chunk.chunk_index = $chunk_index,
            chunk.summary = $summary,
            chunk.body = $body,
            chunk.text = $body,
            chunk.keywords = $keywords,
            chunk.char_start = $char_start,
            chunk.char_end = $char_end,
            chunk.created_at = coalesce(chunk.created_at, $created_at, $now),
            chunk.updated_at = $now
        """,
        chunk_key=chunk_key,
        run_id=run_id,
        document_name=document_name,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        summary=summary,
        body=body,
        keywords=keywords,
        char_start=char_start,
        char_end=char_end,
        created_at=created_at,
        now=now,
    )
    _replace_keyword_nodes(
        transaction,
        run_id=run_id,
        document_name=document_name,
        chunk_id=chunk_id,
        keywords=keywords,
        keyword_vectors=keyword_vectors or [],
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )


def _replace_keyword_nodes(
    transaction: ManagedTransaction,
    *,
    run_id: str,
    document_name: str,
    chunk_id: str,
    keywords: list[str],
    keyword_vectors: list[list[float]],
    embedding_provider: str | None,
    embedding_model: str | None,
    embedding_dimensions: int | None,
) -> None:
    if keyword_vectors and len(keywords) != len(keyword_vectors):
        raise ValueError(
            "keyword vector count does not match keyword count: "
            f"{len(keyword_vectors)}/{len(keywords)}"
        )
    transaction.run(
        """
        MATCH (:Chunk {run_id: $run_id, document_name: $document_name, chunk_id: $chunk_id})-[edge:HAS_KEYWORD]->
              (keyword:KeywordNode {run_id: $run_id, document_name: $document_name, owner_id: $chunk_id})
        DELETE edge, keyword
        """,
        run_id=run_id,
        document_name=document_name,
        chunk_id=chunk_id,
    )
    if not keyword_vectors:
        return
    for keyword_index, (keyword, embedding) in enumerate(zip(keywords, keyword_vectors, strict=True)):
        transaction.run(
            """
            MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name, chunk_id: $chunk_id})
            MERGE (keyword:KeywordNode {
                run_id: $run_id,
                document_name: $document_name,
                owner_id: $chunk_id,
                keyword_index: $keyword_index
            })
            SET
                keyword.run_id = $run_id,
                keyword.document_name = $document_name,
                keyword.owner_id = $chunk_id,
                keyword.keyword_index = $keyword_index,
                keyword.keyword = $keyword,
                keyword.embedding = $embedding,
                keyword.dimension = $dimension,
                keyword.embedding_provider = $embedding_provider,
                keyword.embedding_model = $embedding_model,
                keyword.embedding_dimensions = $embedding_dimensions
            MERGE (chunk)-[edge:HAS_KEYWORD {keyword_index: $keyword_index}]->(keyword)
            SET edge.run_id = $run_id
            """,
            run_id=run_id,
            document_name=document_name,
            chunk_id=chunk_id,
            keyword_index=keyword_index,
            keyword=keyword,
            embedding=[float(value) for value in embedding],
            dimension=len(embedding),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=int(embedding_dimensions or len(embedding)),
        )


def link_chunks(
    transaction: ManagedTransaction,
    *,
    run_id: str,
    left_chunk_id: str,
    right_chunk_id: str,
    dist: float,
) -> None:
    transaction.run(
        """
        MATCH (left:Chunk {run_id: $run_id, chunk_id: $left_chunk_id})
        MATCH (right:Chunk {run_id: $run_id, chunk_id: $right_chunk_id})
        WITH left, right,
             CASE
                 WHEN left.chunk_id <= right.chunk_id THEN left.chunk_id + '|' + right.chunk_id
                 ELSE right.chunk_id + '|' + left.chunk_id
             END AS edge_key,
             CASE WHEN left.chunk_id <= right.chunk_id THEN left ELSE right END AS source,
             CASE WHEN left.chunk_id <= right.chunk_id THEN right ELSE left END AS target
        MERGE (source)-[edge:LINKS {edge_key: edge_key}]->(target)
        SET
            edge.run_id = $run_id,
            edge.u = source.chunk_id,
            edge.v = target.chunk_id,
            edge.dist = $dist,
            edge.distance = $dist,
            edge.edge_kind = coalesce(edge.edge_kind, 'normal_graph')
        """,
        run_id=run_id,
        left_chunk_id=left_chunk_id,
        right_chunk_id=right_chunk_id,
        dist=float(dist),
    )


def unlink_chunks(
    transaction: ManagedTransaction,
    *,
    run_id: str,
    left_chunk_id: str,
    right_chunk_id: str,
) -> None:
    transaction.run(
        """
        MATCH (left:Chunk {run_id: $run_id, chunk_id: $left_chunk_id})-[edge:LINKS]-(right:Chunk {run_id: $run_id, chunk_id: $right_chunk_id})
        DELETE edge
        """,
        run_id=run_id,
        left_chunk_id=left_chunk_id,
        right_chunk_id=right_chunk_id,
    )


def unlink_extra_edges(
    transaction: ManagedTransaction,
    *,
    run_id: str,
    chunk_id: str,
    protected_chunk_ids: list[str],
) -> None:
    transaction.run(
        """
        MATCH (chunk:Chunk {run_id: $run_id, chunk_id: $chunk_id})-[edge:LINKS]-(neighbor:Chunk {run_id: $run_id})
        WHERE NOT neighbor.chunk_id IN $protected_chunk_ids
        DELETE edge
        """,
        run_id=run_id,
        chunk_id=chunk_id,
        protected_chunk_ids=protected_chunk_ids,
    )
