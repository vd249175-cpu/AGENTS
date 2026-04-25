"""Neo4j-backed graph store for GraphNode and Chunk graph operations."""

from datetime import datetime, timezone
import heapq
from pathlib import Path
from secrets import token_hex
from typing import Any, Iterable, Mapping

from neo4j import Driver, GraphDatabase

from server.embedding_keywords import (
    embed_keywords,
    keyword_embedding_dimensions,
    keyword_embedding_index_name,
    keyword_embedding_profile,
)
from .chunk_catalog_schema import SCHEMA_STATEMENTS
from .chunk_store_writes import create_schema
from .database_config import get_neo4j_config
from .document_store_reads import normalize_document_name


DEFAULT_GRAPH_EDGE_DISTANCE = 0.3
GRAPH_RELATIONSHIPS = ("LINKS", "DOCUMENT_NEXT")


def _normalize_keywords(keywords: Iterable[str] | None) -> list[str]:
    normalized: list[str] = []
    for keyword in keywords or []:
        cleaned = str(keyword).strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _apply_keyword_ops(base_keywords: list[str], keyword_ops: list[object] | None) -> list[str]:
    keywords = _normalize_keywords(base_keywords)
    for raw_op in keyword_ops or []:
        payload = raw_op if isinstance(raw_op, Mapping) else getattr(raw_op, "__dict__", {})
        op = str(payload.get("op") or "").strip()
        op_keywords = _normalize_keywords(payload.get("keywords") or [])
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GraphStore:
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
        default_edge_distance: float = DEFAULT_GRAPH_EDGE_DISTANCE,
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
        self.default_edge_distance = float(default_edge_distance)
        self.persist_keyword_embeddings = persist_keyword_embeddings
        self.embedding_config_override = dict(embedding_config_override or {}) or None
        self.keyword_index_name = self._keyword_embedding_index_name()
        self.keyword_dimensions = self._keyword_embedding_dimensions()
        self.keyword_profile = keyword_embedding_profile(config_override=self.embedding_config_override)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                create_schema,
                statements=SCHEMA_STATEMENTS,
                embedding_config_override=self.embedding_config_override,
            )
            session.execute_write(self._create_graph_schema)

    @staticmethod
    def _create_graph_schema(transaction) -> None:
        transaction.run(
            """
            CREATE CONSTRAINT graph_node_identity IF NOT EXISTS
            FOR (node:GraphNode)
            REQUIRE (node.run_id, node.node_id) IS UNIQUE
            """
        )
        transaction.run(
            """
            CREATE CONSTRAINT graph_keyword_owner_identity IF NOT EXISTS
            FOR (keyword:KeywordNode)
            REQUIRE (keyword.run_id, keyword.owner_label, keyword.owner_id, keyword.keyword_index) IS UNIQUE
            """
        )

    def manage_nodes(self, *, actions: list[dict[str, Any]], run_id: str | None = None) -> dict[str, Any]:
        active_run_id = self._resolve_run_id(run_id)
        if not actions:
            return {"ok": False, "run_id": active_run_id, "message": "items must not be empty.", "results": []}
        results: list[dict[str, Any]] = []
        overall_ok = True
        for action in actions:
            try:
                result = self._apply_action(run_id=active_run_id, action=action)
            except Exception as exc:  # noqa: BLE001
                overall_ok = False
                result = {
                    "operation": str(action.get("operation") or action.get("op") or "unknown"),
                    "status": "error",
                    "message": str(exc),
                    "ids": action.get("ids") or [],
                    "results": [],
                }
            if result.get("status") != "success":
                overall_ok = False
            results.append(result)
        return {
            "ok": overall_ok,
            "run_id": active_run_id,
            "message": "Batch completed successfully." if overall_ok else "Batch completed with errors.",
            "results": results,
        }

    def read_nodes(
        self,
        *,
        ids: list[str] | None,
        run_id: str | None = None,
        detail_mode: str = "summary",
    ) -> dict[str, Any]:
        active_run_id = self._resolve_run_id(run_id)
        if ids is None:
            return {
                "ok": True,
                "run_id": active_run_id,
                "message": "No ids were provided. Supply ids to read graph nodes.",
                "ids": [],
                "missing_ids": [],
                "results": [],
            }
        requested_ids = [str(node_id).strip() for node_id in ids if str(node_id).strip()]
        if not requested_ids:
            raise ValueError("ids must not be empty when provided")
        found: list[dict[str, Any]] = []
        missing: list[str] = []
        for node_id in requested_ids:
            payload = self.fetch_node(run_id=active_run_id, node_id=node_id, detail_mode=detail_mode)
            if payload is None:
                missing.append(node_id)
            else:
                found.append(payload)
        return {
            "ok": not missing,
            "run_id": active_run_id,
            "message": f"Read {len(found)} node(s)." if not missing else f"Read {len(found)} node(s), {len(missing)} missing.",
            "ids": requested_ids,
            "missing_ids": missing,
            "results": found,
        }

    def fetch_node(self, *, run_id: str, node_id: str, detail_mode: str = "detail") -> dict[str, Any] | None:
        with self.driver.session(database=self.database) as session:
            node = session.execute_read(self._read_node, run_id, node_id)
        if node is None:
            return None
        return self._decorate_node(node, detail_mode=detail_mode)

    def recall_nodes_by_keywords(
        self,
        *,
        query_keywords: list[str],
        run_id: str | None = None,
        top_k: int = 5,
        detail_mode: str = "summary",
        blocked_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        active_run_id = self._resolve_run_id(run_id)
        keywords = _normalize_keywords(query_keywords)
        if not keywords:
            raise ValueError("query_keywords must not be empty")
        top_k = max(1, int(top_k))
        vectors = self._embed_keywords(keywords) if self.persist_keyword_embeddings else []
        aggregated: dict[str, dict[str, Any]] = {}
        if vectors:
            index_name = self._keyword_embedding_index_name()
            with self.driver.session(database=self.database) as session:
                for query_index, (query_keyword, vector) in enumerate(zip(keywords, vectors, strict=False)):
                    records = session.execute_read(
                        self._keyword_vector_query,
                        active_run_id,
                        index_name,
                        query_keyword,
                        [float(value) for value in vector],
                        top_k,
                        max(top_k * 5, 10),
                    )
                    for rank, record in enumerate(records):
                        node_id = str(record["node_id"])
                        item = aggregated.setdefault(
                            node_id,
                            {
                                "score": 0.0,
                                "matched_keywords": [],
                                "matched_count": 0,
                                "first_order": (query_index, rank),
                            },
                        )
                        item["score"] = float(item["score"]) + float(record["score"])
                        item["matched_count"] = int(item["matched_count"]) + 1
                        item["first_order"] = min(tuple(item["first_order"]), (query_index, rank))
                        for matched_keyword in record.get("matched_keywords") or []:
                            matched = str(matched_keyword)
                            if matched and matched not in item["matched_keywords"]:
                                item["matched_keywords"].append(matched)
        blocked = blocked_ids or set()
        ranked: list[dict[str, Any]] = []
        for node_id, item in aggregated.items():
            if node_id in blocked:
                continue
            payload = self.fetch_node(run_id=active_run_id, node_id=node_id, detail_mode=detail_mode)
            if payload is None:
                continue
            matched_count = int(item.get("matched_count") or 0)
            payload["score"] = float(item["score"]) + max(0, matched_count - 1) * 0.1
            payload["matched_keywords"] = list(item["matched_keywords"])
            payload["_keyword_recall_order"] = tuple(item["first_order"])
            ranked.append(payload)
        ranked.sort(
            key=lambda item: (
                -float(item.get("score") or 0.0),
                tuple(item.get("_keyword_recall_order") or (0, 0)),
                str(item.get("node_id") or ""),
            )
        )
        for item in ranked:
            item.pop("_keyword_recall_order", None)
        results = ranked[:top_k]
        return {
            "ok": True,
            "run_id": active_run_id,
            "query_keywords": keywords,
            "detail_mode": detail_mode,
            "message": f"Found {len(results)} candidate(s).",
            "candidate_count": len(results),
            "results": results,
        }

    def distance_recall(
        self,
        *,
        anchor_node_id: str,
        run_id: str | None = None,
        max_distance: float = 1.0,
        top_k: int = 5,
        detail_mode: str = "summary",
        blocked_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        active_run_id = self._resolve_run_id(run_id)
        anchor = str(anchor_node_id).strip()
        if not anchor:
            raise ValueError("anchor_node_id is required")
        graph = self._load_graph_snapshot(run_id=active_run_id)
        if anchor not in graph["nodes"]:
            return {
                "ok": False,
                "run_id": active_run_id,
                "anchor_node_id": anchor,
                "message": f"Node {anchor} not found.",
                "candidate_count": 0,
                "results": [],
            }
        distances = self._dijkstra(graph["edges"], anchor=anchor, max_distance=float(max_distance))
        blocked = blocked_ids or set()
        ranked_ids = [
            node_id
            for node_id, distance in sorted(distances.items(), key=lambda item: (item[1], item[0]))
            if node_id != anchor and node_id not in blocked
        ][: max(1, int(top_k))]
        results: list[dict[str, Any]] = []
        for node_id in ranked_ids:
            payload = self.fetch_node(run_id=active_run_id, node_id=node_id, detail_mode=detail_mode)
            if payload is None:
                continue
            payload["distance"] = distances[node_id]
            results.append(payload)
        return {
            "ok": True,
            "run_id": active_run_id,
            "anchor_node_id": anchor,
            "max_distance": float(max_distance),
            "detail_mode": detail_mode,
            "candidate_count": len(results),
            "results": results,
        }

    def _apply_action(self, *, run_id: str, action: dict[str, Any]) -> dict[str, Any]:
        operation = str(action.get("operation") or action.get("op") or "").strip().lower()
        if operation == "create":
            return self._create_nodes(run_id=run_id, action=action)
        if operation == "update":
            return self._update_node(run_id=run_id, action=action)
        if operation == "delete":
            return self._delete_nodes(run_id=run_id, action=action)
        raise ValueError(f"unsupported graph operation: {operation}")

    def _create_nodes(self, *, run_id: str, action: dict[str, Any]) -> dict[str, Any]:
        summary = str(action.get("summary") or "").strip()
        body = str(action.get("body") or "").strip()
        if not summary:
            raise ValueError("summary is required for create")
        if not body:
            raise ValueError("body is required for create")
        keyword_ops = list(action.get("keyword_ops") or [])
        if not keyword_ops:
            raise ValueError("keyword_ops are required for create")
        node_ids = [str(node_id).strip() for node_id in (action.get("ids") or [self._new_node_id(run_id)]) if str(node_id).strip()]
        if not node_ids:
            raise ValueError("ids must not be empty when provided")
        existing = [node_id for node_id in node_ids if self.fetch_node(run_id=run_id, node_id=node_id, detail_mode="summary") is not None]
        if existing:
            raise ValueError(f"Node id already exists: {existing}")
        created: list[dict[str, Any]] = []
        for node_id in node_ids:
            with self.driver.session(database=self.database) as session:
                session.execute_write(self._upsert_graph_node, run_id, node_id, summary, body)
            keywords = _apply_keyword_ops([], keyword_ops)
            self._replace_graph_keywords(run_id=run_id, node_id=node_id, keywords=keywords)
            if action.get("edge_ops"):
                self._apply_edge_ops(run_id=run_id, source_node_id=node_id, edge_ops=list(action.get("edge_ops") or []))
            payload = self.fetch_node(run_id=run_id, node_id=node_id, detail_mode="detail")
            if payload is not None:
                created.append(payload)
        return {
            "operation": "create",
            "status": "success",
            "message": f"Created {len(created)} node(s).",
            "ids": node_ids,
            "created_ids": node_ids,
            "results": created,
        }

    def _update_node(self, *, run_id: str, action: dict[str, Any]) -> dict[str, Any]:
        ids = [str(node_id).strip() for node_id in (action.get("ids") or []) if str(node_id).strip()]
        if len(ids) != 1:
            raise ValueError("ids must contain exactly one id for update")
        node_id = ids[0]
        current = self.fetch_node(run_id=run_id, node_id=node_id, detail_mode="detail")
        if current is None:
            return {"operation": "update", "status": "error", "message": f"Node {node_id} not found.", "ids": ids, "results": []}
        is_chunk = current.get("node_label") == "Chunk"
        keyword_ops = list(action.get("keyword_ops") or [])
        edge_ops = list(action.get("edge_ops") or [])
        has_content_mutation = (
            action.get("summary") is not None
            or action.get("body") is not None
            or bool(keyword_ops)
        )
        if is_chunk and has_content_mutation:
            raise ValueError(f"Node {node_id} is document-backed; use document tools to edit chunk content or keywords.")
        if not is_chunk:
            summary = str(action.get("summary") if action.get("summary") is not None else current.get("summary") or "").strip()
            body = str(action.get("body") if action.get("body") is not None else current.get("body") or "").strip()
            with self.driver.session(database=self.database) as session:
                session.execute_write(self._upsert_graph_node, run_id, node_id, summary, body)
            if keyword_ops:
                keywords = _apply_keyword_ops(list(current.get("keywords") or []), keyword_ops)
                self._replace_graph_keywords(run_id=run_id, node_id=node_id, keywords=keywords)
            elif not self._graph_keyword_profile_matches_current(run_id=run_id, node_id=node_id):
                self._replace_graph_keywords(run_id=run_id, node_id=node_id, keywords=list(current.get("keywords") or []))
        if edge_ops:
            self._apply_edge_ops(run_id=run_id, source_node_id=node_id, edge_ops=edge_ops)
        payload = self.fetch_node(run_id=run_id, node_id=node_id, detail_mode="detail")
        return {
            "operation": "update",
            "status": "success",
            "message": f"Updated node {node_id}.",
            "ids": ids,
            "results": [payload] if payload else [],
        }

    def _delete_nodes(self, *, run_id: str, action: dict[str, Any]) -> dict[str, Any]:
        ids = [str(node_id).strip() for node_id in (action.get("ids") or []) if str(node_id).strip()]
        if not ids:
            raise ValueError("ids are required for delete")
        deleted: list[dict[str, Any]] = []
        errors: list[str] = []
        for node_id in ids:
            current = self.fetch_node(run_id=run_id, node_id=node_id, detail_mode="detail")
            if current is None:
                errors.append(f"Node {node_id} not found.")
                continue
            if current.get("node_label") == "Chunk":
                errors.append(f"Node {node_id} is document-backed; use document tools to remove chunk content.")
                continue
            with self.driver.session(database=self.database) as session:
                session.execute_write(self._delete_graph_node, run_id, node_id)
            deleted.append({"node_id": node_id, "status": "success", "mode": "physical", "before": current})
        return {
            "operation": "delete",
            "status": "success" if not errors else "error",
            "message": f"Physically deleted {len(deleted)} node(s)." if not errors else "Deleted some nodes but encountered errors: " + "; ".join(errors),
            "ids": ids,
            "deleted_ids": [item["node_id"] for item in deleted],
            "errors": errors,
            "results": deleted,
        }

    def _apply_edge_ops(self, *, run_id: str, source_node_id: str, edge_ops: list[object]) -> None:
        if self.fetch_node(run_id=run_id, node_id=source_node_id, detail_mode="summary") is None:
            raise KeyError(f"Node {source_node_id} not found.")
        for raw_op in edge_ops:
            op = raw_op if isinstance(raw_op, Mapping) else getattr(raw_op, "__dict__", {})
            action = str(op.get("op") or "").strip()
            if action not in {"+", "-"}:
                raise ValueError(f"unsupported edge op: {action}")
            targets = op.get("targets")
            if targets == "all":
                if action != "-":
                    raise ValueError('targets="all" is only supported for delete operations')
                with self.driver.session(database=self.database) as session:
                    session.execute_write(self._unlink_all_links, run_id, source_node_id)
                continue
            if targets is None:
                raise ValueError("edge targets are required")
            target_ids = self._resolve_edge_targets(run_id=run_id, raw_targets=targets, document_name=op.get("document_name"))
            for target_id in target_ids:
                if target_id == source_node_id:
                    raise ValueError("self loops are not allowed")
                if action == "+":
                    self._link_nodes(run_id=run_id, left_id=source_node_id, right_id=target_id, dist=float(op.get("dist") or self.default_edge_distance))
                else:
                    self._unlink_nodes(run_id=run_id, left_id=source_node_id, right_id=target_id)

    def _resolve_edge_targets(self, *, run_id: str, raw_targets: object, document_name: object | None) -> list[str]:
        target_values = [raw_targets] if isinstance(raw_targets, (int, str)) else list(raw_targets)  # type: ignore[arg-type]
        resolved: list[str] = []
        normalized_document_name = normalize_document_name(str(document_name)) if document_name is not None else None
        for target in target_values:
            if isinstance(target, int):
                resolved.append(self._chunk_id_by_index(run_id=run_id, document_name=normalized_document_name, chunk_index=target))
            elif isinstance(target, str):
                cleaned = target.strip()
                if self.fetch_node(run_id=run_id, node_id=cleaned, detail_mode="summary") is None:
                    raise KeyError(f"Node {cleaned} not found.")
                resolved.append(cleaned)
            elif isinstance(target, list) and len(target) == 2 and all(isinstance(item, int) for item in target):
                start, end = int(target[0]), int(target[1])
                if start > end:
                    raise ValueError("range end must be greater than or equal to start")
                for index in range(start, end + 1):
                    resolved.append(self._chunk_id_by_index(run_id=run_id, document_name=normalized_document_name, chunk_index=index))
            elif isinstance(target, list) and all(isinstance(item, str) for item in target):
                for item in target:
                    cleaned = item.strip()
                    if self.fetch_node(run_id=run_id, node_id=cleaned, detail_mode="summary") is None:
                        raise KeyError(f"Node {cleaned} not found.")
                    resolved.append(cleaned)
            else:
                raise ValueError("unsupported edge target type")
        return list(dict.fromkeys(resolved))

    def _chunk_id_by_index(self, *, run_id: str, document_name: str | None, chunk_index: int) -> str:
        if document_name is None:
            raise ValueError("document_name is required for chunk index targets")
        if chunk_index < 0:
            raise ValueError("chunk indexes must be non-negative")
        with self.driver.session(database=self.database) as session:
            record = session.run(
                """
                MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name, chunk_index: $chunk_index})
                RETURN coalesce(chunk.chunk_id, chunk.id) AS node_id
                """,
                run_id=run_id,
                document_name=document_name,
                chunk_index=int(chunk_index),
            ).single()
        if record is None:
            raise KeyError(f"chunk index {chunk_index} not found for {document_name}")
        return str(record["node_id"])

    def _link_nodes(self, *, run_id: str, left_id: str, right_id: str, dist: float) -> None:
        left_key, right_key = sorted((left_id, right_id))
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._merge_link, run_id, left_id, right_id, left_key, right_key, float(dist))

    def _unlink_nodes(self, *, run_id: str, left_id: str, right_id: str) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._delete_link, run_id, left_id, right_id)

    def _replace_graph_keywords(self, *, run_id: str, node_id: str, keywords: list[str]) -> None:
        vectors = self._embed_keywords(keywords, salt=f"{run_id}:{node_id}") if self.persist_keyword_embeddings and keywords else []
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                self._replace_keywords,
                run_id,
                node_id,
                keywords,
                vectors,
                self.keyword_profile["provider"],
                self.keyword_profile["model"],
                self.keyword_profile["dimensions"],
            )

    def _keyword_embedding_index_name(self) -> str:
        if self.embedding_config_override is None:
            return keyword_embedding_index_name()
        return keyword_embedding_index_name(config_override=self.embedding_config_override)

    def _keyword_embedding_dimensions(self) -> int:
        if self.embedding_config_override is None:
            return keyword_embedding_dimensions()
        return keyword_embedding_dimensions(config_override=self.embedding_config_override)

    def _embed_keywords(self, keywords: list[str], *, salt: str = "") -> list[list[float]]:
        if self.embedding_config_override is None:
            return embed_keywords(keywords, salt=salt)
        return embed_keywords(keywords, salt=salt, config_override=self.embedding_config_override)

    def _decorate_node(self, node: dict[str, Any], *, detail_mode: str) -> dict[str, Any]:
        node_id = str(node["node_id"])
        run_id = str(node["run_id"])
        payload: dict[str, Any] = {
            "run_id": run_id,
            "node_id": node_id,
            "node_label": node["node_label"],
            "summary": str(node.get("summary") or ""),
            "edges": self._read_edges(run_id=run_id, node_id=node_id),
        }
        if node.get("document_name") is not None:
            payload["document_name"] = str(node["document_name"])
        if node.get("chunk_index") is not None:
            payload["chunk_index"] = int(node["chunk_index"])
        if detail_mode == "detail":
            payload["body"] = str(node.get("body") or "")
            payload["keywords"] = self._read_keywords(run_id=run_id, node_id=node_id)
        elif detail_mode != "summary":
            raise ValueError("detail_mode must be summary or detail")
        return payload

    def _read_edges(self, *, run_id: str, node_id: str) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            records = session.execute_read(self._read_node_edges, run_id, node_id)
        return records

    def _read_keywords(self, *, run_id: str, node_id: str) -> list[str]:
        with self.driver.session(database=self.database) as session:
            records = session.execute_read(self._read_node_keywords, run_id, node_id)
        return [str(record["keyword"]) for record in records]

    def _graph_keyword_profile_matches_current(self, *, run_id: str, node_id: str) -> bool:
        with self.driver.session(database=self.database) as session:
            records = session.execute_read(self._read_node_keyword_profiles, run_id, node_id)
        if not records:
            return True
        provider = str(self.keyword_profile["provider"])
        model = str(self.keyword_profile["model"])
        dimensions = int(self.keyword_profile["dimensions"])
        return all(
            str(record.get("embedding_provider") or "").strip() == provider
            and str(record.get("embedding_model") or "").strip() == model
            and int(record.get("embedding_dimensions") or 0) == dimensions
            for record in records
        )

    def _load_graph_snapshot(self, *, run_id: str) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(self._read_graph_snapshot, run_id)

    def _new_node_id(self, run_id: str) -> str:
        while True:
            candidate = token_hex(4)
            if self.fetch_node(run_id=run_id, node_id=candidate, detail_mode="summary") is None:
                return candidate

    def _resolve_run_id(self, run_id: str | None) -> str:
        active_run_id = run_id or self.run_id
        if not active_run_id:
            raise ValueError("run_id is required for graph operations")
        return str(active_run_id)

    @staticmethod
    def _dijkstra(edges: dict[str, list[tuple[str, float]]], *, anchor: str, max_distance: float) -> dict[str, float]:
        distances: dict[str, float] = {anchor: 0.0}
        queue: list[tuple[float, str]] = [(0.0, anchor)]
        while queue:
            current_distance, node_id = heapq.heappop(queue)
            if current_distance > distances[node_id]:
                continue
            for neighbor_id, edge_distance in edges.get(node_id, []):
                candidate = current_distance + float(edge_distance)
                if candidate > max_distance:
                    continue
                if candidate < distances.get(neighbor_id, float("inf")):
                    distances[neighbor_id] = candidate
                    heapq.heappush(queue, (candidate, neighbor_id))
        return distances

    @staticmethod
    def _node_match_where(alias: str, parameter_name: str = "node_id") -> str:
        return (
            f"{alias}.run_id = $run_id AND "
            f"(({alias}:GraphNode AND coalesce({alias}.node_id, {alias}.id) = ${parameter_name}) OR "
            f"({alias}:Chunk AND coalesce({alias}.chunk_id, {alias}.id) = ${parameter_name}))"
        )

    @staticmethod
    def _read_node(transaction, run_id: str, node_id: str) -> dict[str, Any] | None:
        record = transaction.run(
            """
            MATCH (node)
            WHERE node.run_id = $run_id
              AND (
                (node:GraphNode AND coalesce(node.node_id, node.id) = $node_id)
                OR
                (node:Chunk AND coalesce(node.chunk_id, node.id) = $node_id)
              )
            RETURN
                labels(node) AS labels,
                node.run_id AS run_id,
                CASE WHEN node:Chunk THEN coalesce(node.chunk_id, node.id) ELSE coalesce(node.node_id, node.id) END AS node_id,
                CASE WHEN node:Chunk THEN 'Chunk' ELSE 'GraphNode' END AS node_label,
                node.document_name AS document_name,
                node.chunk_index AS chunk_index,
                node.summary AS summary,
                coalesce(node.body, node.text, '') AS body
            LIMIT 1
            """,
            run_id=run_id,
            node_id=node_id,
        ).single()
        return dict(record) if record is not None else None

    @staticmethod
    def _upsert_graph_node(transaction, run_id: str, node_id: str, summary: str, body: str) -> None:
        now = _now()
        transaction.run(
            """
            MERGE (node:GraphNode {run_id: $run_id, node_id: $node_id})
            SET
                node.id = $node_id,
                node.run_id = $run_id,
                node.summary = $summary,
                node.body = $body,
                node.created_at = coalesce(node.created_at, $now),
                node.updated_at = $now
            """,
            run_id=run_id,
            node_id=node_id,
            summary=summary,
            body=body,
            now=now,
        )

    @staticmethod
    def _delete_graph_node(transaction, run_id: str, node_id: str) -> None:
        transaction.run(
            """
            MATCH (node:GraphNode {run_id: $run_id, node_id: $node_id})
            OPTIONAL MATCH (node)-[keyword_edge:HAS_KEYWORD]->(keyword:KeywordNode {
                run_id: $run_id,
                owner_label: 'GraphNode',
                owner_id: $node_id
            })
            DELETE keyword_edge, keyword
            WITH node
            DETACH DELETE node
            """,
            run_id=run_id,
            node_id=node_id,
        )

    @staticmethod
    def _replace_keywords(
        transaction,
        run_id: str,
        node_id: str,
        keywords: list[str],
        vectors: list[list[float]],
        embedding_provider: str,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> None:
        transaction.run(
            """
            MATCH (node:GraphNode {run_id: $run_id, node_id: $node_id})-[edge:HAS_KEYWORD]->
                  (keyword:KeywordNode {run_id: $run_id, owner_label: 'GraphNode', owner_id: $node_id})
            DELETE edge, keyword
            """,
            run_id=run_id,
            node_id=node_id,
        )
        for keyword_index, keyword in enumerate(keywords):
            vector = vectors[keyword_index] if keyword_index < len(vectors) else []
            transaction.run(
                """
                MATCH (node:GraphNode {run_id: $run_id, node_id: $node_id})
                MERGE (keyword:KeywordNode {
                    run_id: $run_id,
                    owner_label: 'GraphNode',
                    owner_id: $node_id,
                    keyword_index: $keyword_index
                })
                SET
                    keyword.run_id = $run_id,
                    keyword.owner_label = 'GraphNode',
                    keyword.owner_id = $node_id,
                    keyword.document_name = '',
                    keyword.keyword_index = $keyword_index,
                    keyword.keyword = $keyword,
                    keyword.embedding = $embedding,
                    keyword.dimension = $dimension,
                    keyword.embedding_provider = $embedding_provider,
                    keyword.embedding_model = $embedding_model,
                    keyword.embedding_dimensions = $embedding_dimensions
                MERGE (node)-[edge:HAS_KEYWORD {keyword_index: $keyword_index}]->(keyword)
                SET edge.run_id = $run_id
                """,
                run_id=run_id,
                node_id=node_id,
                keyword_index=keyword_index,
                keyword=keyword,
                embedding=[float(value) for value in vector],
                dimension=len(vector),
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_dimensions=int(embedding_dimensions or len(vector)),
            )

    @staticmethod
    def _merge_link(transaction, run_id: str, left_id: str, right_id: str, left_key: str, right_key: str, dist: float) -> None:
        edge_key = f"{left_key}|{right_key}"
        transaction.run(
            """
            MATCH (left)
            WHERE left.run_id = $run_id
              AND (
                (left:GraphNode AND coalesce(left.node_id, left.id) = $left_id)
                OR
                (left:Chunk AND coalesce(left.chunk_id, left.id) = $left_id)
              )
            MATCH (right)
            WHERE right.run_id = $run_id
              AND (
                (right:GraphNode AND coalesce(right.node_id, right.id) = $right_id)
                OR
                (right:Chunk AND coalesce(right.chunk_id, right.id) = $right_id)
              )
            WITH
                CASE WHEN $left_id <= $right_id THEN left ELSE right END AS source,
                CASE WHEN $left_id <= $right_id THEN right ELSE left END AS target
            MERGE (source)-[edge:LINKS {run_id: $run_id, edge_key: $edge_key}]->(target)
            SET
                edge.run_id = $run_id,
                edge.edge_key = $edge_key,
                edge.u = $left_id,
                edge.v = $right_id,
                edge.dist = $dist,
                edge.distance = $dist,
                edge.edge_kind = 'normal_graph'
            """,
            run_id=run_id,
            left_id=left_id,
            right_id=right_id,
            edge_key=edge_key,
            dist=dist,
        )

    @staticmethod
    def _delete_link(transaction, run_id: str, left_id: str, right_id: str) -> None:
        transaction.run(
            """
            MATCH (left)-[edge:LINKS]-(right)
            WHERE left.run_id = $run_id
              AND right.run_id = $run_id
              AND (
                ((left:GraphNode AND coalesce(left.node_id, left.id) = $left_id) OR (left:Chunk AND coalesce(left.chunk_id, left.id) = $left_id))
                AND
                ((right:GraphNode AND coalesce(right.node_id, right.id) = $right_id) OR (right:Chunk AND coalesce(right.chunk_id, right.id) = $right_id))
                OR
                ((left:GraphNode AND coalesce(left.node_id, left.id) = $right_id) OR (left:Chunk AND coalesce(left.chunk_id, left.id) = $right_id))
                AND
                ((right:GraphNode AND coalesce(right.node_id, right.id) = $left_id) OR (right:Chunk AND coalesce(right.chunk_id, right.id) = $left_id))
              )
            DELETE edge
            """,
            run_id=run_id,
            left_id=left_id,
            right_id=right_id,
        )

    @staticmethod
    def _unlink_all_links(transaction, run_id: str, source_node_id: str) -> None:
        transaction.run(
            """
            MATCH (node)-[edge:LINKS]-(neighbor)
            WHERE node.run_id = $run_id
              AND neighbor.run_id = $run_id
              AND (
                (node:GraphNode AND coalesce(node.node_id, node.id) = $source_node_id)
                OR
                (node:Chunk AND coalesce(node.chunk_id, node.id) = $source_node_id)
              )
            DELETE edge
            """,
            run_id=run_id,
            source_node_id=source_node_id,
        )

    @staticmethod
    def _read_node_edges(transaction, run_id: str, node_id: str) -> list[dict[str, Any]]:
        result = transaction.run(
            """
            MATCH (node)-[edge:LINKS|DOCUMENT_NEXT]-(neighbor)
            WHERE node.run_id = $run_id
              AND neighbor.run_id = $run_id
              AND (
                (node:GraphNode AND coalesce(node.node_id, node.id) = $node_id)
                OR
                (node:Chunk AND coalesce(node.chunk_id, node.id) = $node_id)
              )
            RETURN DISTINCT
                CASE WHEN neighbor:Chunk THEN coalesce(neighbor.chunk_id, neighbor.id) ELSE coalesce(neighbor.node_id, neighbor.id) END AS neighbor_node_id,
                coalesce(edge.dist, edge.distance, $default_dist) AS dist
            ORDER BY neighbor_node_id
            """,
            run_id=run_id,
            node_id=node_id,
            default_dist=DEFAULT_GRAPH_EDGE_DISTANCE,
        )
        return [
            {
                "neighbor_node_id": str(record["neighbor_node_id"]),
                "dist": float(record["dist"] or DEFAULT_GRAPH_EDGE_DISTANCE),
            }
            for record in result
        ]

    @staticmethod
    def _read_node_keywords(transaction, run_id: str, node_id: str) -> list[dict[str, Any]]:
        result = transaction.run(
            """
            MATCH (node)-[:HAS_KEYWORD]->(keyword:KeywordNode)
            WHERE node.run_id = $run_id
              AND keyword.run_id = $run_id
              AND (
                (node:GraphNode AND coalesce(node.node_id, node.id) = $node_id)
                OR
                (node:Chunk AND coalesce(node.chunk_id, node.id) = $node_id)
              )
            RETURN keyword.keyword AS keyword, keyword.keyword_index AS keyword_index
            ORDER BY keyword.keyword_index
            """,
            run_id=run_id,
            node_id=node_id,
        )
        return [dict(record) for record in result]

    @staticmethod
    def _read_node_keyword_profiles(transaction, run_id: str, node_id: str) -> list[dict[str, Any]]:
        result = transaction.run(
            """
            MATCH (node)-[:HAS_KEYWORD]->(keyword:KeywordNode)
            WHERE node.run_id = $run_id
              AND keyword.run_id = $run_id
              AND (
                (node:GraphNode AND coalesce(node.node_id, node.id) = $node_id)
                OR
                (node:Chunk AND coalesce(node.chunk_id, node.id) = $node_id)
              )
            RETURN
                keyword.embedding_provider AS embedding_provider,
                keyword.embedding_model AS embedding_model,
                coalesce(keyword.embedding_dimensions, keyword.dimension) AS embedding_dimensions
            """
            ,
            run_id=run_id,
            node_id=node_id,
        )
        return [dict(record) for record in result]

    @staticmethod
    def _keyword_vector_query(
        transaction,
        run_id: str,
        index_name: str,
        query_keyword: str,
        vector: list[float],
        top_k: int,
        candidate_k: int,
    ) -> list[dict[str, Any]]:
        result = transaction.run(
            f"""
            CALL db.index.vector.queryNodes('{index_name}', $candidate_k, $vector)
            YIELD node, score
            WHERE node:KeywordNode
              AND node.run_id = $run_id
            MATCH (owner)-[:HAS_KEYWORD]->(node)
            WHERE owner.run_id = $run_id
              AND (owner:GraphNode OR owner:Chunk)
            WITH owner, max(score) AS score, collect(DISTINCT node.keyword) AS matched_keywords
            RETURN
                CASE WHEN owner:Chunk THEN coalesce(owner.chunk_id, owner.id) ELSE coalesce(owner.node_id, owner.id) END AS node_id,
                CASE WHEN owner:Chunk THEN 'Chunk' ELSE 'GraphNode' END AS node_label,
                score AS score,
                matched_keywords AS matched_keywords,
                $query_keyword AS query_keyword
            ORDER BY score DESC, node_id ASC
            LIMIT $top_k
            """,
            run_id=run_id,
            vector=vector,
            query_keyword=query_keyword,
            candidate_k=max(candidate_k, top_k),
            top_k=top_k,
        )
        return [dict(record) for record in result]

    @staticmethod
    def _read_graph_snapshot(transaction, run_id: str) -> dict[str, Any]:
        node_result = transaction.run(
            """
            MATCH (node)
            WHERE node.run_id = $run_id AND (node:GraphNode OR node:Chunk)
            RETURN CASE WHEN node:Chunk THEN coalesce(node.chunk_id, node.id) ELSE coalesce(node.node_id, node.id) END AS node_id
            """,
            run_id=run_id,
        )
        edge_result = transaction.run(
            """
            MATCH (source)-[edge:LINKS|DOCUMENT_NEXT]->(target)
            WHERE source.run_id = $run_id
              AND target.run_id = $run_id
              AND (source:GraphNode OR source:Chunk)
              AND (target:GraphNode OR target:Chunk)
            RETURN
                CASE WHEN source:Chunk THEN coalesce(source.chunk_id, source.id) ELSE coalesce(source.node_id, source.id) END AS source_id,
                CASE WHEN target:Chunk THEN coalesce(target.chunk_id, target.id) ELSE coalesce(target.node_id, target.id) END AS target_id,
                coalesce(edge.dist, edge.distance, $default_dist) AS dist
            """,
            run_id=run_id,
            default_dist=DEFAULT_GRAPH_EDGE_DISTANCE,
        )
        nodes = {str(record["node_id"]) for record in node_result}
        edges: dict[str, list[tuple[str, float]]] = {}
        for record in edge_result:
            source_id = str(record["source_id"])
            target_id = str(record["target_id"])
            dist = float(record["dist"] or DEFAULT_GRAPH_EDGE_DISTANCE)
            edges.setdefault(source_id, []).append((target_id, dist))
            edges.setdefault(target_id, []).append((source_id, dist))
        return {"nodes": nodes, "edges": edges}

    def close(self) -> None:
        self.driver.close()
