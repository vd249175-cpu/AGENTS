"""Write helpers for the migrated Neo4j chunk catalog store."""

from __future__ import annotations

from datetime import datetime, timezone

from neo4j import ManagedTransaction

from server.embedding_keywords import keyword_embedding_dimensions, keyword_embedding_index_name


def create_schema(
    transaction: ManagedTransaction,
    *,
    statements: list[str],
    embedding_config_override: dict[str, object] | None = None,
) -> None:
    for statement in statements:
        transaction.run(statement)
    index_name = keyword_embedding_index_name(config_override=embedding_config_override)
    transaction.run(
        f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (keyword:KeywordNode) ON (keyword.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: $dimensions,
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """,
        dimensions=keyword_embedding_dimensions(config_override=embedding_config_override),
    )


def clear_document(transaction: ManagedTransaction, *, run_id: str, document_name: str) -> None:
    transaction.run(
        """
        MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name})-[:HAS_KEYWORD]->
              (keyword:KeywordNode {run_id: $run_id, document_name: $document_name})
        DETACH DELETE keyword
        """,
        run_id=run_id,
        document_name=document_name,
    )
    transaction.run(
        """
        MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name})
        DETACH DELETE chunk
        """,
        run_id=run_id,
        document_name=document_name,
    )


def upsert_chunk(
    transaction: ManagedTransaction,
    *,
    run_id: str,
    document_name: str,
    chunk_id: str,
    chunk_index: int,
    summary: str,
    keywords: list[str],
    text: str,
    char_start: int,
    char_end: int,
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
            chunk.keywords = $keywords,
            chunk.body = $text,
            chunk.text = $text,
            chunk.char_start = $char_start,
            chunk.char_end = $char_end,
            chunk.created_at = coalesce(chunk.created_at, $now),
            chunk.updated_at = $now
        """,
        chunk_key=chunk_key,
        run_id=run_id,
        document_name=document_name,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        summary=summary,
        keywords=keywords,
        text=text,
        char_start=char_start,
        char_end=char_end,
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
