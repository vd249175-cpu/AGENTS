"""Real Neo4j-backed chunk catalog store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neo4j import Driver, GraphDatabase

from .chunk_catalog_schema import SCHEMA_STATEMENTS
from .chunk_store_reads import document_exists, list_chunks, list_document_names
from .chunk_store_search import keyword_vector_search
from .chunk_store_writes import clear_document, create_schema, upsert_chunk
from .database_config import get_neo4j_config
from .document_edges import DEFAULT_DOCUMENT_EDGE_DISTANCE, rebuild_document_order_edges
from server.embedding_keywords import (
    embed_keywords,
    keyword_embedding_dimensions,
    keyword_embedding_index_name,
    keyword_embedding_profile,
)


class ChunkStore:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        driver: Driver | None = None,
        document_edge_distance: float = DEFAULT_DOCUMENT_EDGE_DISTANCE,
        persist_keyword_embeddings: bool = True,
        embedding_config_override: dict[str, Any] | None = None,
    ) -> None:
        config = get_neo4j_config(path=config_path)
        resolved_uri = uri or str(config.get("uri") or "")
        resolved_username = username or str(config.get("username") or "")
        resolved_password = password or str(config.get("password") or "")
        resolved_database = database if database is not None else (str(config.get("database")) if config.get("database") else None)
        if not resolved_uri or not resolved_username or not resolved_password:
            raise ValueError("Missing Neo4j configuration. Set workspace/config/database_config.json or pass explicit credentials.")
        self.database = resolved_database
        self.driver = driver or GraphDatabase.driver(resolved_uri, auth=(resolved_username, resolved_password))
        self.document_edge_distance = float(document_edge_distance)
        self.persist_keyword_embeddings = persist_keyword_embeddings
        self.embedding_config_override = dict(embedding_config_override or {}) or None
        self.keyword_index_name = keyword_embedding_index_name(config_override=self.embedding_config_override)
        self.keyword_dimensions = keyword_embedding_dimensions(config_override=self.embedding_config_override)
        self.keyword_profile = keyword_embedding_profile(config_override=self.embedding_config_override)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                create_schema,
                statements=SCHEMA_STATEMENTS,
                embedding_config_override=self.embedding_config_override,
            )

    def clear_document(self, *, run_id: str, document_name: str) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(clear_document, run_id=run_id, document_name=document_name)

    def rebuild_document_edges(self, *, run_id: str, document_name: str) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                rebuild_document_order_edges,
                run_id=run_id,
                document_name=document_name,
                distance=self.document_edge_distance,
            )

    def upsert_chunk(
        self,
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
    ) -> None:
        keyword_vectors = self._embed_keywords(keywords, salt=document_name) if self.persist_keyword_embeddings and keywords else []
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                upsert_chunk,
                run_id=run_id,
                document_name=document_name,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                summary=summary,
                keywords=keywords,
                text=text,
                char_start=char_start,
                char_end=char_end,
                keyword_vectors=keyword_vectors,
                embedding_provider=self.keyword_profile["provider"],
                embedding_model=self.keyword_profile["model"],
                embedding_dimensions=self.keyword_profile["dimensions"],
            )

    def list_document_names(self, *, run_id: str) -> list[str]:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(list_document_names, run_id=run_id)

    def document_exists(self, *, document_name: str, run_id: str | None = None) -> bool:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(document_exists, document_name=document_name, run_id=run_id)

    def list_chunks(self, *, run_id: str, document_name: str) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(list_chunks, run_id=run_id, document_name=document_name)

    def list_chunks_with_keywords(self, *, run_id: str, document_name: str) -> list[dict[str, Any]]:
        return self.list_chunks(run_id=run_id, document_name=document_name)

    def keyword_vector_search(
        self,
        *,
        query_keywords: list[str],
        run_id: str | None = None,
        document_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(
                keyword_vector_search,
                query_keywords=query_keywords,
                run_id=run_id,
                document_name=document_name,
                limit=limit,
                embedding_config_override=self.embedding_config_override,
            )

    def _embed_keywords(self, keywords: list[str], *, salt: str = "") -> list[list[float]]:
        if self.embedding_config_override is None:
            return embed_keywords(keywords, salt=salt)
        return embed_keywords(keywords, salt=salt, config_override=self.embedding_config_override)

    def close(self) -> None:
        self.driver.close()
