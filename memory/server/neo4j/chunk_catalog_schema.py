"""Schema helpers for the migrated Neo4j chunk catalog store."""

SCHEMA_STATEMENTS = [
    """
    CREATE CONSTRAINT chunk_key_unique IF NOT EXISTS
    FOR (chunk:Chunk)
    REQUIRE chunk.chunk_key IS UNIQUE
    """,
    """
    CREATE INDEX chunk_document_name_index IF NOT EXISTS
    FOR (chunk:Chunk)
    ON (chunk.document_name)
    """,
    """
    CREATE INDEX chunk_document_run_order_index IF NOT EXISTS
    FOR (chunk:Chunk)
    ON (chunk.run_id, chunk.document_name, chunk.chunk_index)
    """,
    """
    CREATE CONSTRAINT chunk_keyword_unique IF NOT EXISTS
    FOR (keyword:KeywordNode)
    REQUIRE (keyword.run_id, keyword.document_name, keyword.owner_id, keyword.keyword_index) IS UNIQUE
    """,
]
