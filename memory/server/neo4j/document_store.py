"""Document-side store backed by Neo4j Chunk nodes."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from secrets import token_hex
from typing import Any

from neo4j import Driver, GraphDatabase

from .chunk_catalog_schema import SCHEMA_STATEMENTS
from .chunk_store_writes import create_schema
from .database_config import get_neo4j_config
from .document_store_reads import document_exists, document_run_id, list_chunks, list_documents, normalize_document_name
from .document_edges import DEFAULT_DOCUMENT_EDGE_DISTANCE, rebuild_document_order_edges
from .document_store_writes import delete_chunks_not_in_ids, link_chunks, unlink_chunks, unlink_extra_edges, upsert_chunk
from server.embedding_keywords import (
    embed_keywords,
    keyword_embedding_dimensions,
    keyword_embedding_index_name,
    keyword_embedding_profile,
)


def _normalize_keywords(keywords: list[str] | None) -> list[str]:
    values: list[str] = []
    for keyword in keywords or []:
        cleaned = str(keyword).strip()
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values


def _apply_keyword_ops(base_keywords: list[str], keyword_ops: list[object] | None) -> list[str]:
    keywords = _normalize_keywords(base_keywords)
    for raw_op in keyword_ops or []:
        if isinstance(raw_op, dict):
            op = str(raw_op.get("op") or "").strip()
            raw_keywords = raw_op.get("keywords") or []
        else:
            op = str(getattr(raw_op, "op", "") or "").strip()
            raw_keywords = getattr(raw_op, "keywords", []) or []
        op_keywords = _normalize_keywords(list(raw_keywords))
        if op == "+":
            for keyword in op_keywords:
                if keyword not in keywords:
                    keywords.append(keyword)
        elif op == "-":
            remove_set = set(op_keywords)
            keywords = [keyword for keyword in keywords if keyword not in remove_set]
        elif op == "replace":
            keywords = op_keywords
        else:
            raise ValueError(f"unsupported keyword op: {op}")
    return keywords


def _action_value(action: dict[str, object], key: str) -> object | None:
    value = action.get(key)
    return value if value is not None else None


def _build_run_id(document_name: str) -> str:
    digest = sha1(document_name.encode("utf-8")).hexdigest()[:12]
    return f"document-run-{digest}"


class DocumentStore:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        driver: Driver | None = None,
        run_id: str | None = None,
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
        self.run_id = run_id
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

    def document_exists(self, *, document_name: str, run_id: str | None = None) -> bool:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(document_exists, document_name=document_name, run_id=run_id)

    def list_documents(self, *, run_id: str | None = None) -> list[dict[str, object]]:
        active_run_id = run_id if run_id is not None else self.run_id
        with self.driver.session(database=self.database) as session:
            return session.execute_read(list_documents, run_id=active_run_id)

    def list_chunks(self, *, document_name: str, run_id: str | None = None) -> list[dict[str, object]]:
        normalized_name = normalize_document_name(document_name)
        active_run_id = self._resolve_read_run_id(document_name=normalized_name, run_id=run_id)
        with self.driver.session(database=self.database) as session:
            return session.execute_read(list_chunks, document_name=normalized_name, run_id=active_run_id)

    def query_positions(self, *, document_name: str, positions: list[int] | None = None, run_id: str | None = None) -> list[dict[str, object]]:
        chunks = self.list_chunks(document_name=document_name, run_id=run_id)
        if positions is None:
            return chunks
        wanted = {int(position) for position in positions}
        return [chunk for chunk in chunks if int(chunk["chunk_index"]) in wanted]

    def list_chunks_with_keywords(self, *, document_name: str | None = None, run_id: str | None = None) -> list[dict[str, object]]:
        if document_name is not None:
            return self.list_chunks(document_name=document_name, run_id=run_id)
        active_run_id = run_id if run_id is not None else self.run_id
        records: list[dict[str, object]] = []
        for document in self.list_documents(run_id=active_run_id):
            records.extend(self.list_chunks(document_name=str(document["document_name"]), run_id=active_run_id))
        return records

    def create_document(
        self,
        *,
        document_name: str,
        summary: str,
        body: str,
        keywords: list[str] | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        normalized_name = normalize_document_name(document_name)
        if self.document_exists(document_name=normalized_name):
            return {"ok": False, "document_name": normalized_name, "message": "当前文档已经在memory中"}
        active_run_id = run_id or self.run_id or _build_run_id(normalized_name)
        chunk = {
            "run_id": active_run_id,
            "id": token_hex(4),
            "document_name": normalized_name,
            "chunk_index": 0,
            "summary": summary.strip(),
            "body": body,
            "keywords": _normalize_keywords(keywords),
            "char_start": 0,
            "char_end": len(body),
            "created_at": None,
        }
        self.replace_document(document_name=normalized_name, run_id=active_run_id, chunks=[chunk])
        return {"ok": True, "document_name": normalized_name, "run_id": active_run_id, "chunks": self.list_chunks(document_name=normalized_name, run_id=active_run_id)}

    def replace_document(self, *, document_name: str, run_id: str, chunks: list[dict[str, object]]) -> None:
        normalized_name = normalize_document_name(document_name)
        with self.driver.session(database=self.database) as session:
            keep_chunk_ids = [str(chunk.get("id") or token_hex(4)) for chunk in chunks]
            session.execute_write(delete_chunks_not_in_ids, run_id=run_id, document_name=normalized_name, keep_chunk_ids=keep_chunk_ids)
            cursor = 0
            for chunk_index, chunk in enumerate(chunks):
                body = str(chunk.get("body") or chunk.get("text") or "")
                char_start = cursor
                char_end = cursor + len(body)
                cursor = char_end
                chunk_id = str(chunk.get("id") or keep_chunk_ids[chunk_index])
                keywords = _normalize_keywords(list(chunk.get("keywords") or []))
                keyword_vectors = self._resolve_keyword_vectors(chunk=chunk, keywords=keywords, salt=normalized_name)
                session.execute_write(
                    upsert_chunk,
                    run_id=run_id,
                    document_name=normalized_name,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    summary=str(chunk.get("summary") or "").strip(),
                    body=body,
                    keywords=keywords,
                    char_start=char_start,
                    char_end=char_end,
                    created_at=str(chunk.get("created_at") or "") or None,
                    keyword_vectors=keyword_vectors,
                    embedding_provider=self.keyword_profile["provider"],
                    embedding_model=self.keyword_profile["model"],
                    embedding_dimensions=self.keyword_profile["dimensions"],
                )
            session.execute_write(
                rebuild_document_order_edges,
                run_id=run_id,
                document_name=normalized_name,
                distance=self.document_edge_distance,
            )

    def _embed_keywords(self, keywords: list[str], *, salt: str = "") -> list[list[float]]:
        if self.embedding_config_override is None:
            return embed_keywords(keywords, salt=salt)
        return embed_keywords(keywords, salt=salt, config_override=self.embedding_config_override)

    def _resolve_keyword_vectors(self, *, chunk: dict[str, object], keywords: list[str], salt: str) -> list[list[float]]:
        if not self.persist_keyword_embeddings or not keywords:
            return []
        existing_vectors = [
            [float(value) for value in vector]
            for vector in list(chunk.get("keyword_vectors") or [])
            if isinstance(vector, list)
        ]
        if self._chunk_embedding_profile_matches_current(chunk) and len(existing_vectors) == len(keywords):
            return existing_vectors
        return self._embed_keywords(keywords, salt=salt)

    def _chunk_embedding_profile_matches_current(self, chunk: dict[str, object]) -> bool:
        provider = str(chunk.get("keyword_embedding_provider") or "").strip()
        model = str(chunk.get("keyword_embedding_model") or "").strip()
        dimensions = int(chunk.get("keyword_embedding_dimensions") or 0)
        return (
            provider == str(self.keyword_profile["provider"])
            and model == str(self.keyword_profile["model"])
            and dimensions == int(self.keyword_profile["dimensions"])
        )

    def manage_chunks(self, *, document_name: str, actions: list[dict[str, object]], run_id: str | None = None) -> dict[str, object]:
        normalized_name = normalize_document_name(document_name)
        active_run_id = self._resolve_read_run_id(document_name=normalized_name, run_id=run_id)
        if active_run_id is None:
            return {"ok": False, "document_name": normalized_name, "message": f"文档 {normalized_name} 不存在"}
        chunks = self.list_chunks(document_name=normalized_name, run_id=active_run_id)
        if not chunks:
            return {"ok": False, "document_name": normalized_name, "message": f"文档 {normalized_name} 不存在"}
        working = [dict(chunk) for chunk in chunks]
        operation_results: list[dict[str, object]] = []
        deferred_edge_ops: list[tuple[str, list[object]]] = []
        for action in actions:
            op = str(action.get("op") or "").strip().lower()
            chunk_count_before = len(working)
            if op == "insert":
                insert_after = action.get("insert_after")
                target_index = len(working) if insert_after is None else int(insert_after) + 1
                target_index = max(0, min(target_index, len(working)))
                keywords = _normalize_keywords(list(action.get("keywords") or []))
                if action.get("keyword_ops"):
                    keywords = _apply_keyword_ops(keywords, list(action.get("keyword_ops") or []))
                chunk_id = str(action.get("id") or token_hex(4))
                working.insert(
                    target_index,
                    {
                        "run_id": active_run_id,
                        "id": chunk_id,
                        "document_name": normalized_name,
                        "chunk_index": target_index,
                        "summary": str(action.get("summary") or "").strip(),
                        "body": str(action.get("body") or ""),
                        "keywords": keywords,
                        "created_at": None,
                    },
                )
                if action.get("edge_ops"):
                    deferred_edge_ops.append((chunk_id, list(action.get("edge_ops") or [])))
                operation_results.append(
                    {
                        "operation": "insert",
                        "status": "success",
                        "id": chunk_id,
                        "chunk_count_before": chunk_count_before,
                        "chunk_count_after": len(working),
                        "message": "Inserted chunk. Re-read before further index-based edits to avoid offset.",
                    }
                )
            elif op == "update":
                target = self._find_target(working, action)
                if target is None:
                    raise ValueError("update target chunk not found")
                if action.get("summary") is not None:
                    target["summary"] = str(action.get("summary") or "").strip()
                if action.get("body") is not None:
                    target["body"] = str(action.get("body") or "")
                if action.get("keywords") is not None:
                    replaced_keywords = _normalize_keywords(list(action.get("keywords") or []))
                    target["keywords"] = replaced_keywords
                    target["keyword_vectors"] = self._embed_keywords(replaced_keywords, salt=normalized_name) if replaced_keywords else []
                    self._apply_current_embedding_profile(target)
                elif action.get("keyword_ops"):
                    next_keywords, next_vectors = self._apply_incremental_keyword_ops(
                        chunk=target,
                        keyword_ops=list(action.get("keyword_ops") or []),
                        salt=normalized_name,
                    )
                    target["keywords"] = next_keywords
                    target["keyword_vectors"] = next_vectors
                    self._apply_current_embedding_profile(target)
                elif not self._chunk_embedding_profile_matches_current(target):
                    current_keywords = _normalize_keywords(list(target.get("keywords") or []))
                    target["keyword_vectors"] = self._embed_keywords(current_keywords, salt=normalized_name) if current_keywords else []
                    self._apply_current_embedding_profile(target)
                chunk_id = str(target["id"])
                if action.get("edge_ops"):
                    deferred_edge_ops.append((chunk_id, list(action.get("edge_ops") or [])))
                operation_results.append(
                    {
                        "operation": "update",
                        "status": "success",
                        "id": chunk_id,
                        "chunk_count_before": chunk_count_before,
                        "chunk_count_after": len(working),
                        "message": "Updated chunk. Document order remained stable; keyword and extra edge changes were applied.",
                    }
                )
            elif op == "delete":
                target = self._find_target(working, action)
                if target is None:
                    raise ValueError("delete target chunk not found")
                chunk_id = str(target["id"])
                working = [chunk for chunk in working if chunk is not target]
                operation_results.append(
                    {
                        "operation": "delete",
                        "status": "success",
                        "id": chunk_id,
                        "chunk_count_before": chunk_count_before,
                        "chunk_count_after": len(working),
                        "message": "Deleted chunk. Document order changed; re-read before further index-based edits to avoid offset.",
                    }
                )
            else:
                raise ValueError(f"unsupported chunk op: {op}")
        self.replace_document(document_name=normalized_name, run_id=active_run_id, chunks=working)
        refreshed = self.list_chunks(document_name=normalized_name, run_id=active_run_id)
        for chunk_id, edge_ops in deferred_edge_ops:
            self._apply_edge_ops(run_id=active_run_id, document_name=normalized_name, chunk_id=chunk_id, edge_ops=edge_ops)
        if deferred_edge_ops:
            refreshed = self.list_chunks(document_name=normalized_name, run_id=active_run_id)
        return {
            "ok": True,
            "document_name": normalized_name,
            "run_id": active_run_id,
            "results": operation_results,
            "chunk_count_before": len(chunks),
            "chunk_count_after": len(refreshed),
            "chunks": refreshed,
        }

    def keyword_vector_search(
        self,
        *,
        query_keywords: list[str],
        run_id: str | None = None,
        document_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, object]]:
        from .chunk_store_search import keyword_vector_search

        with self.driver.session(database=self.database) as session:
            return session.execute_read(
                keyword_vector_search,
                query_keywords=query_keywords,
                run_id=run_id,
                document_name=normalize_document_name(document_name) if document_name is not None else None,
                limit=limit,
            )

    def _document_run_id(self, *, document_name: str) -> str | None:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(document_run_id, document_name=document_name)

    def _resolve_read_run_id(self, *, document_name: str, run_id: str | None = None) -> str | None:
        if run_id is not None:
            return run_id
        if self.run_id is not None:
            return self.run_id
        return self._document_run_id(document_name=document_name)

    def _apply_edge_ops(self, *, run_id: str, document_name: str, chunk_id: str, edge_ops: list[object]) -> None:
        chunks = self.list_chunks(document_name=document_name, run_id=run_id)
        chunk_ids = [str(chunk["id"]) for chunk in chunks]
        if chunk_id not in chunk_ids:
            raise ValueError(f"chunk {chunk_id} not found")
        protected_ids = self._document_neighbor_ids(chunks, chunk_id)
        for raw_op in edge_ops:
            op = raw_op if isinstance(raw_op, dict) else getattr(raw_op, "__dict__", {})
            action = str(op.get("op") or "").strip()
            targets = op.get("targets")
            if action not in {"+", "-"}:
                raise ValueError(f"unsupported edge op: {action}")
            if targets == "all":
                if action != "-":
                    raise ValueError('targets="all" is only supported for delete operations')
                with self.driver.session(database=self.database) as session:
                    session.execute_write(unlink_extra_edges, run_id=run_id, chunk_id=chunk_id, protected_chunk_ids=sorted(protected_ids))
                continue
            if targets is None:
                raise ValueError("edge targets are required")
            target_values = [targets] if isinstance(targets, (int, str)) else list(targets)
            target_ids = self._resolve_edge_targets(run_id=run_id, document_name=document_name, targets=target_values)
            for target_id in target_ids:
                if target_id == chunk_id:
                    raise ValueError("self loops are not allowed")
                if target_id in protected_ids:
                    raise ValueError("document order edges cannot be modified through edge_ops")
                with self.driver.session(database=self.database) as session:
                    if action == "+":
                        session.execute_write(
                            link_chunks,
                            run_id=run_id,
                            left_chunk_id=chunk_id,
                            right_chunk_id=target_id,
                            dist=float(op.get("dist") or self.document_edge_distance),
                        )
                    else:
                        session.execute_write(unlink_chunks, run_id=run_id, left_chunk_id=chunk_id, right_chunk_id=target_id)

    def _resolve_edge_targets(self, *, run_id: str, document_name: str, targets: list[object]) -> list[str]:
        document_chunks = self.list_chunks(document_name=document_name, run_id=run_id)
        by_index = {int(chunk["chunk_index"]): str(chunk["id"]) for chunk in document_chunks}
        all_ids = {str(chunk["id"]) for chunk in self.list_chunks_with_keywords(run_id=run_id)}
        resolved: list[str] = []
        for target in targets:
            if isinstance(target, int):
                if target not in by_index:
                    raise KeyError(f"chunk index {target} not found")
                resolved.append(by_index[target])
            elif isinstance(target, str):
                cleaned = target.strip()
                if cleaned not in all_ids:
                    raise KeyError(f"chunk {cleaned} not found")
                resolved.append(cleaned)
            elif isinstance(target, list) and len(target) == 2 and all(isinstance(item, int) for item in target):
                start, end = int(target[0]), int(target[1])
                if start > end:
                    raise ValueError("range end must be greater than or equal to start")
                for index in range(start, end + 1):
                    if index not in by_index:
                        raise KeyError(f"chunk index {index} not found")
                    resolved.append(by_index[index])
            elif isinstance(target, list) and all(isinstance(item, str) for item in target):
                for item in target:
                    cleaned = item.strip()
                    if cleaned not in all_ids:
                        raise KeyError(f"chunk {cleaned} not found")
                    resolved.append(cleaned)
            else:
                raise ValueError("unsupported edge target type")
        return list(dict.fromkeys(resolved))

    @staticmethod
    def _document_neighbor_ids(chunks: list[dict[str, object]], chunk_id: str) -> set[str]:
        for index, chunk in enumerate(chunks):
            if str(chunk["id"]) != chunk_id:
                continue
            neighbors: set[str] = set()
            if index > 0:
                neighbors.add(str(chunks[index - 1]["id"]))
            if index + 1 < len(chunks):
                neighbors.add(str(chunks[index + 1]["id"]))
            return neighbors
        return set()

    @staticmethod
    def _find_target(chunks: list[dict[str, object]], action: dict[str, object]) -> dict[str, object] | None:
        if action.get("id") is not None:
            wanted_id = str(action["id"])
            for chunk in chunks:
                if str(chunk.get("id")) == wanted_id:
                    return chunk
        if action.get("chunk_index") is not None:
            wanted_index = int(action["chunk_index"])
            for chunk in chunks:
                if int(chunk.get("chunk_index", -1)) == wanted_index:
                    return chunk
        return None

    def _apply_incremental_keyword_ops(
        self,
        *,
        chunk: dict[str, object],
        keyword_ops: list[object],
        salt: str,
    ) -> tuple[list[str], list[list[float]]]:
        base_keywords = _normalize_keywords(list(chunk.get("keywords") or []))
        if not self.persist_keyword_embeddings:
            return _apply_keyword_ops(base_keywords, keyword_ops), []
        if not keyword_ops:
            return base_keywords, [
                [float(value) for value in vector]
                for vector in list(chunk.get("keyword_vectors") or [])
                if isinstance(vector, list)
            ]
        if not self._chunk_embedding_profile_matches_current(chunk):
            next_keywords = _apply_keyword_ops(base_keywords, keyword_ops)
            return next_keywords, self._embed_keywords(next_keywords, salt=salt) if next_keywords else []
        replace_mode = False
        current_map: dict[str, list[float]] = {}
        existing_vectors = list(chunk.get("keyword_vectors") or [])
        for index, keyword in enumerate(base_keywords):
            vector = existing_vectors[index] if index < len(existing_vectors) and isinstance(existing_vectors[index], list) else None
            current_map[keyword] = [float(value) for value in vector] if isinstance(vector, list) else []
        keywords = list(base_keywords)
        for raw_op in keyword_ops:
            if isinstance(raw_op, dict):
                op = str(raw_op.get("op") or "").strip()
                raw_keywords = raw_op.get("keywords") or []
            else:
                op = str(getattr(raw_op, "op", "") or "").strip()
                raw_keywords = getattr(raw_op, "keywords", []) or []
            op_keywords = _normalize_keywords(list(raw_keywords))
            if op == "+":
                new_keywords = [keyword for keyword in op_keywords if keyword not in current_map]
                if new_keywords:
                    new_vectors = self._embed_keywords(new_keywords, salt=salt)
                    if len(new_keywords) != len(new_vectors):
                        raise ValueError(
                            "embedded keyword vector count does not match new keyword count: "
                            f"{len(new_vectors)}/{len(new_keywords)}"
                        )
                    for keyword, vector in zip(new_keywords, new_vectors, strict=True):
                        current_map[keyword] = [float(value) for value in vector]
                keywords = _apply_keyword_ops(keywords, [raw_op])
            elif op == "-":
                keywords = _apply_keyword_ops(keywords, [raw_op])
                current_map = {keyword: current_map.get(keyword, []) for keyword in keywords}
            elif op == "replace":
                replace_mode = True
                keywords = op_keywords
                break
            else:
                raise ValueError(f"unsupported keyword op: {op}")
        if replace_mode:
            return keywords, self._embed_keywords(keywords, salt=salt) if keywords else []
        vectors = [current_map.get(keyword, []) for keyword in keywords]
        if any(len(vector) == 0 for vector in vectors if keywords):
            return keywords, self._embed_keywords(keywords, salt=salt) if keywords else []
        return keywords, vectors

    def _apply_current_embedding_profile(self, chunk: dict[str, object]) -> None:
        chunk["keyword_embedding_provider"] = self.keyword_profile["provider"]
        chunk["keyword_embedding_model"] = self.keyword_profile["model"]
        chunk["keyword_embedding_dimensions"] = self.keyword_profile["dimensions"]

    def close(self) -> None:
        self.driver.close()
