import json
import unittest
from pathlib import Path
from secrets import token_hex
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from server.neo4j import DocumentStore
from server.neo4j.graph_store import GraphStore
from tools.create_chunk_document import CreateChunkDocumentInput, CreateChunkDocumentToolConfig, build_create_chunk_document_tool
from tools.graph_distance_recall import GraphDistanceRecallInput, GraphDistanceRecallToolConfig, build_graph_distance_recall_tool
from tools.graph_manage_nodes import (
    GraphCreateNodeAction,
    GraphCreateNodesInput,
    GraphDeleteNodesInput,
    GraphEdgeOp,
    GraphKeywordOp,
    GraphManageNodesInput,
    GraphManageNodesToolConfig,
    GraphNodeAction,
    GraphUpdateNodeInput,
    build_graph_create_nodes_tool,
    build_graph_delete_nodes_tool,
    build_graph_manage_nodes_tool,
    build_graph_update_node_tool,
)
from tools.graph_mark_blocked import GraphMarkBlockedInput, GraphMarkBlockedToolConfig, build_graph_mark_blocked_tool
from tools.graph_mark_useful import GraphMarkUsefulInput, GraphMarkUsefulToolConfig, build_graph_mark_useful_tool
from tools.graph_clear_blocked import GraphClearBlockedInput, GraphClearBlockedToolConfig, build_graph_clear_blocked_tool
from tools.keyword_recall import KeywordRecallInput, KeywordRecallToolConfig, build_keyword_recall_tool
from tools.manage_chunks import ChunkManageAction, ManageChunksInput, ManageChunksToolConfig, build_manage_chunks_tool
from tools.read_nodes import ReadNodesInput, ReadNodesToolConfig, build_read_nodes_tool
from tools.recall_nodes_by_keywords import RecallNodesByKeywordsToolConfig, build_recall_nodes_by_keywords_tool


CONFIG_PATH = Path("/Users/apexwave/Desktop/memory/workspace/config/database_config.json")


def _normalize_result(result):
    if isinstance(result, Command):
        update = dict(result.update or {})
        payload = {}
        for message in update.get("messages") or []:
            if isinstance(message, ToolMessage) and isinstance(message.content, str):
                payload = json.loads(message.content)
                break
        state_update = {key: value for key, value in update.items() if key != "messages"}
        if state_update:
            payload["state_update"] = state_update
        return payload
    return result


class GraphToolTests(unittest.TestCase):
    def test_keyword_recall_aggregates_owner_hits_with_stored_keywords(self) -> None:
        run_id = f"graph_keyword_logic_{token_hex(4)}"
        store = object.__new__(GraphStore)
        store.run_id = run_id
        store.database = None
        store.persist_keyword_embeddings = True
        store.embedding_config_override = None
        records_by_query = {
            "query-alpha": [
                {"node_id": "node-a", "score": 0.6, "matched_keywords": ["stored-alpha", "stored-alias"]},
                {"node_id": "node-b", "score": 0.6, "matched_keywords": ["stored-beta"]},
            ],
            "query-beta": [
                {"node_id": "node-a", "score": 0.5, "matched_keywords": ["stored-gamma"]},
            ],
        }

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def execute_read(self, _callback, _run_id, _index_name, query_keyword, *_args):
                return list(records_by_query[str(query_keyword)])

        class FakeDriver:
            def session(self, database=None):
                return FakeSession()

        store.driver = FakeDriver()
        store.fetch_node = lambda *, run_id, node_id, detail_mode: {  # type: ignore[method-assign]
            "run_id": run_id,
            "node_id": node_id,
            "summary": f"summary {node_id}",
            "edges": [],
        }

        with patch("server.neo4j.graph_store.embed_keywords", return_value=[[1.0], [0.5]]):
            with patch("server.neo4j.graph_store.keyword_embedding_index_name", return_value="keyword_index"):
                result = store.recall_nodes_by_keywords(
                    run_id=run_id,
                    query_keywords=["query-alpha", "query-beta"],
                    top_k=3,
                    detail_mode="summary",
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["candidate_count"], 2)
        first = result["results"][0]
        self.assertEqual(first["node_id"], "node-a")
        self.assertEqual(first["matched_keywords"], ["stored-alpha", "stored-alias", "stored-gamma"])
        self.assertNotIn("query-alpha", first["matched_keywords"])
        self.assertAlmostEqual(float(first["score"]), 1.2)

    def test_keyword_recall_uses_embedding_override_for_query_side(self) -> None:
        run_id = f"graph_keyword_override_{token_hex(4)}"
        override = {
            "provider": "openai",
            "model": "override-embedding",
            "base_url": "https://override.example/v1",
            "api_key": "override-key",
            "dimensions": 256,
        }
        store = object.__new__(GraphStore)
        store.run_id = run_id
        store.database = None
        store.persist_keyword_embeddings = True
        store.embedding_config_override = override
        store.fetch_node = lambda *, run_id, node_id, detail_mode: None  # type: ignore[method-assign]

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def execute_read(self, *_args):
                return []

        class FakeDriver:
            def session(self, database=None):
                return FakeSession()

        store.driver = FakeDriver()

        with patch("server.neo4j.graph_store.embed_keywords", return_value=[[1.0]]) as embed_mock:
            with patch("server.neo4j.graph_store.keyword_embedding_index_name", return_value="override_index") as index_mock:
                result = store.recall_nodes_by_keywords(
                    run_id=run_id,
                    query_keywords=["override"],
                    top_k=3,
                    detail_mode="summary",
                )

        self.assertTrue(result["ok"])
        embed_mock.assert_called_once_with(["override"], salt="", config_override=override)
        index_mock.assert_called_once_with(config_override=override)

    def test_keyword_recall_rejects_misaligned_store_override(self) -> None:
        store = object.__new__(GraphStore)
        store.run_id = "misaligned-run"
        store.keyword_index_name = "chunk_keyword_embedding_index__openai__other__128"
        store.keyword_dimensions = 128

        with self.assertRaisesRegex(ValueError, "does not match graph_store keyword index"):
            build_keyword_recall_tool(
                config=KeywordRecallToolConfig(
                    neo4j_config_path=CONFIG_PATH,
                    run_id="misaligned-run",
                    embedding_provider="openai",
                    embedding_model="text-embedding-3-small",
                    embedding_base_url="https://example.invalid/v1",
                    embedding_api_key="sk-test",
                    embedding_dimensions=256,
                ),
                store=store,
            )

    def test_keyword_recall_declares_limit_and_returns_structured_error(self) -> None:
        class FakeStore:
            def recall_nodes_by_keywords(self, **_kwargs):
                raise AssertionError("over-limit calls must not query the store")

        tool = build_keyword_recall_tool(
            config=KeywordRecallToolConfig(
                neo4j_config_path=CONFIG_PATH,
                run_id="keyword-limit",
                default_top_k=2,
                top_k_limit=3,
            ),
            store=FakeStore(),
        )
        self.assertIn("top_k_limit=3", tool.description)
        result = tool.invoke(KeywordRecallInput(query_keywords=["limit"], top_k=4).model_dump())
        self.assertEqual(result["operation"], "keyword_recall")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["requested_top_k"], 4)
        self.assertEqual(result["top_k_limit"], 3)
        self.assertEqual(result["suggested_top_k"], 3)

    def test_legacy_keyword_recall_declares_limit_and_returns_structured_error(self) -> None:
        class FakeStore:
            def recall_nodes_by_keywords(self, **_kwargs):
                raise AssertionError("over-limit calls must not query the store")

        tool = build_recall_nodes_by_keywords_tool(
            config=RecallNodesByKeywordsToolConfig(
                neo4j_config_path=CONFIG_PATH,
                run_id="legacy-keyword-limit",
                default_top_k=2,
                top_k_limit=3,
            ),
            store=FakeStore(),
        )
        self.assertIn("top_k_limit=3", tool.description)
        result = tool.invoke({"query_keywords": ["limit"], "top_k": 4, "detail_mode": "summary"})
        self.assertEqual(result["operation"], "recall_nodes_by_keywords")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["requested_top_k"], 4)
        self.assertEqual(result["top_k_limit"], 3)

    def test_graph_distance_recall_declares_limit_and_returns_structured_error(self) -> None:
        class FakeStore:
            def distance_recall(self, **_kwargs):
                raise AssertionError("over-limit calls must not query the store")

        tool = build_graph_distance_recall_tool(
            config=GraphDistanceRecallToolConfig(
                neo4j_config_path=CONFIG_PATH,
                run_id="distance-limit",
                default_top_k=2,
                top_k_limit=3,
                default_max_distance=0.8,
            ),
            store=FakeStore(),
        )
        self.assertIn("top_k_limit=3", tool.description)
        self.assertIn("默认 max_distance=0.8", tool.description)
        result = tool.invoke(GraphDistanceRecallInput(anchor_node_id="node-a", top_k=4).model_dump())
        self.assertEqual(result["operation"], "graph_distance_recall")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["requested_top_k"], 4)
        self.assertEqual(result["top_k_limit"], 3)

    def test_recall_configs_clamp_default_top_k_to_limit(self) -> None:
        keyword_config = KeywordRecallToolConfig(neo4j_config_path=CONFIG_PATH, default_top_k=10, top_k_limit=3)
        legacy_config = RecallNodesByKeywordsToolConfig(neo4j_config_path=CONFIG_PATH, default_top_k=10, top_k_limit=3)
        distance_config = GraphDistanceRecallToolConfig(neo4j_config_path=CONFIG_PATH, default_top_k=10, top_k_limit=3)
        self.assertEqual(keyword_config.default_top_k, 3)
        self.assertEqual(legacy_config.default_top_k, 3)
        self.assertEqual(distance_config.default_top_k, 3)

    def test_manage_read_and_distance_recall_graph_nodes(self) -> None:
        run_id = f"graph_tool_{token_hex(4)}"
        first_id = f"graph_a_{token_hex(4)}"
        second_id = f"graph_b_{token_hex(4)}"
        store = GraphStore(config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=True)
        manage_tool = build_graph_manage_nodes_tool(
            config=GraphManageNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=True),
            store=store,
        )
        read_tool = build_read_nodes_tool(config=ReadNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id), store=store)
        distance_tool = build_graph_distance_recall_tool(
            config=GraphDistanceRecallToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id),
            store=store,
        )
        fake_embedding = [[1.0] + [0.0] * 1535]
        try:
            with patch("server.neo4j.graph_store.embed_keywords", side_effect=lambda keywords, salt="": fake_embedding * len(list(keywords))):
                with patch("server.embedding_keywords.embed_keywords", side_effect=lambda keywords, salt="": fake_embedding * len(list(keywords))):
                    create_result = manage_tool.invoke(
                        GraphManageNodesInput(
                            actions=[
                                GraphNodeAction(
                                    operation="create",
                                    ids=[first_id],
                                    summary="第一图节点",
                                    body="第一图节点正文。",
                                    keyword_ops=[GraphKeywordOp(op="+", keywords=["alpha"])],
                                ),
                                GraphNodeAction(
                                    operation="create",
                                    ids=[second_id],
                                    summary="第二图节点",
                                    body="第二图节点正文。",
                                    keyword_ops=[GraphKeywordOp(op="+", keywords=["beta"])],
                                    edge_ops=[GraphEdgeOp(op="+", targets=[first_id], dist=0.4)],
                                ),
                            ]
                        ).model_dump()
                    )
                    self.assertEqual(create_result["status"], "success")
                    self.assertEqual(create_result["action_count"], 2)
                    self.assertTrue(all("body" not in item for item in create_result["results"]))
                    self.assertTrue(all("keywords" not in item for item in create_result["results"]))
                    self.assertTrue(all("results" not in item for item in create_result["results"]))

                    read_result = read_tool.invoke(ReadNodesInput(ids=[first_id, second_id], detail_mode="detail").model_dump())
                    self.assertEqual(read_result["status"], "success")
                    self.assertEqual(read_result["results"][0]["keywords"], ["alpha"])
                    self.assertNotIn("node_label", read_result["results"][0])

                    distance_result = distance_tool.invoke(GraphDistanceRecallInput(anchor_node_id=second_id, max_distance=0.5).model_dump())
                    self.assertEqual(distance_result["status"], "success")
                    self.assertEqual(distance_result["candidate_count"], 1)
                    self.assertEqual(distance_result["results"][0]["node_id"], first_id)
                    self.assertEqual(float(distance_result["results"][0]["distance"]), 0.4)

                    unlink_result = manage_tool.invoke(
                        GraphManageNodesInput(
                            actions=[GraphNodeAction(operation="update", ids=[second_id], edge_ops=[GraphEdgeOp(op="-", targets="all")])]
                        ).model_dump()
                    )
                    self.assertEqual(unlink_result["status"], "success")
                    read_after_unlink = read_tool.invoke(ReadNodesInput(ids=[second_id], detail_mode="summary").model_dump())
                    self.assertEqual(read_after_unlink["results"][0]["edges"], [])

                    delete_result = manage_tool.invoke(GraphManageNodesInput(actions=[GraphNodeAction(operation="delete", ids=[first_id])]).model_dump())
                    self.assertEqual(delete_result["status"], "success")
                    read_after_delete = read_tool.invoke(ReadNodesInput(ids=[first_id], detail_mode="summary").model_dump())
                    self.assertEqual(read_after_delete["status"], "error")
        finally:
            store.close()

    def test_graph_tools_can_link_chunk_without_modifying_document_content(self) -> None:
        run_id = f"graph_chunk_{token_hex(4)}"
        document_name = f"graph_chunk_doc_{token_hex(4)}"
        graph_node_id = f"graph_node_{token_hex(4)}"
        document_store = DocumentStore(config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False)
        graph_store = GraphStore(config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False)
        create_document_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False),
            store=document_store,
        )
        manage_chunks_tool = build_manage_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False),
            store=document_store,
        )
        manage_graph_tool = build_graph_manage_nodes_tool(
            config=GraphManageNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False),
            store=graph_store,
        )
        read_tool = build_read_nodes_tool(config=ReadNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id), store=graph_store)
        try:
            create_doc = create_document_tool.invoke(
                CreateChunkDocumentInput(
                    document_name=document_name,
                    summary="第一块",
                    body="第一块正文。",
                    keywords=[],
                ).model_dump()
            )
            self.assertEqual(create_doc["status"], "success")
            insert_doc = manage_chunks_tool.invoke(
                ManageChunksInput(
                    document_name=document_name,
                    actions=[ChunkManageAction(op="insert", insert_after=0, summary="第二块", body="第二块正文。")],
                ).model_dump()
            )
            self.assertEqual(insert_doc["status"], "success")
            chunk_id = str(insert_doc["chunks"][0]["id"])

            graph_create = manage_graph_tool.invoke(
                GraphManageNodesInput(
                    actions=[
                        GraphNodeAction(
                            operation="create",
                            ids=[graph_node_id],
                            summary="普通图节点",
                            body="普通图节点正文。",
                            keyword_ops=[GraphKeywordOp(op="+", keywords=["graph"])],
                        )
                    ]
                ).model_dump()
            )
            self.assertEqual(graph_create["status"], "success")

            link_result = manage_graph_tool.invoke(
                GraphManageNodesInput(
                    actions=[
                        GraphNodeAction(
                            operation="update",
                            ids=[chunk_id],
                            edge_ops=[GraphEdgeOp(op="+", targets=[graph_node_id], dist=0.6)],
                        )
                    ]
                ).model_dump()
            )
            self.assertEqual(link_result["status"], "success")
            self.assertEqual(link_result["results"][0]["document_name"], document_name)
            self.assertEqual(link_result["results"][0]["chunk_index"], 0)
            read_chunk = read_tool.invoke(ReadNodesInput(ids=[chunk_id], detail_mode="summary").model_dump())
            edges = read_chunk["results"][0]["edges"]
            self.assertTrue(any(edge["neighbor_node_id"] == graph_node_id for edge in edges))
            self.assertTrue(any(float(edge["dist"]) == 0.6 for edge in edges))
            self.assertTrue(all("relation" not in edge for edge in edges))
            self.assertTrue(all("neighbor_label" not in edge for edge in edges))
            self.assertTrue(all("distance" not in edge for edge in edges))

            illegal_update = manage_graph_tool.invoke(
                GraphManageNodesInput(actions=[GraphNodeAction(operation="update", ids=[chunk_id], summary="不允许")]).model_dump()
            )
            self.assertEqual(illegal_update["status"], "error")
            illegal_delete = manage_graph_tool.invoke(
                GraphManageNodesInput(actions=[GraphNodeAction(operation="delete", ids=[chunk_id])]).model_dump()
            )
            self.assertEqual(illegal_delete["status"], "error")
        finally:
            document_store.close()
            graph_store.close()

    def test_graph_tools_can_link_two_chunk_targets_in_one_update(self) -> None:
        run_id = f"graph_multi_chunk_{token_hex(4)}"
        imported_document_name = f"graph_multi_imported_{token_hex(4)}"
        managed_document_name = f"graph_multi_managed_{token_hex(4)}"
        graph_node_id = f"graph_multi_node_{token_hex(4)}"
        document_store = DocumentStore(config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False)
        graph_store = GraphStore(config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False)
        create_document_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False),
            store=document_store,
        )
        manage_graph_tool = build_graph_manage_nodes_tool(
            config=GraphManageNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False),
            store=graph_store,
        )
        read_tool = build_read_nodes_tool(
            config=ReadNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id),
            store=graph_store,
        )
        try:
            imported_create = create_document_tool.invoke(
                CreateChunkDocumentInput(
                    document_name=imported_document_name,
                    summary="导入文档第一段",
                    body="导入文档第一段正文。",
                    keywords=[],
                ).model_dump()
            )
            managed_create = create_document_tool.invoke(
                CreateChunkDocumentInput(
                    document_name=managed_document_name,
                    summary="管理文档第一段",
                    body="管理文档第一段正文。",
                    keywords=[],
                ).model_dump()
            )
            imported_chunk_id = str(imported_create["chunks"][0]["id"])
            managed_chunk_id = str(managed_create["chunks"][0]["id"])

            graph_create = manage_graph_tool.invoke(
                GraphManageNodesInput(
                    actions=[
                        GraphNodeAction(
                            operation="create",
                            ids=[graph_node_id],
                            summary="多 chunk 连边图节点",
                            body="多 chunk 连边图节点正文。",
                            keyword_ops=[GraphKeywordOp(op="+", keywords=["graph", "multi-link"])],
                        )
                    ]
                ).model_dump()
            )
            self.assertEqual(graph_create["status"], "success")

            graph_update = manage_graph_tool.invoke(
                GraphManageNodesInput(
                    actions=[
                        GraphNodeAction(
                            operation="update",
                            ids=[graph_node_id],
                            edge_ops=[
                                GraphEdgeOp(op="+", targets=[imported_chunk_id], dist=0.3),
                                GraphEdgeOp(op="+", targets=[managed_chunk_id], dist=0.3),
                            ],
                        )
                    ]
                ).model_dump()
            )
            self.assertEqual(graph_update["status"], "success")

            read_graph = read_tool.invoke(ReadNodesInput(ids=[graph_node_id], detail_mode="summary").model_dump())
            self.assertEqual(read_graph["status"], "success")
            neighbor_ids = {edge["neighbor_node_id"] for edge in read_graph["results"][0]["edges"]}
            self.assertEqual(neighbor_ids, {imported_chunk_id, managed_chunk_id})
        finally:
            document_store.close()
            graph_store.close()

    def test_split_graph_write_tools_reuse_manage_nodes_logic(self) -> None:
        run_id = f"graph_split_{token_hex(4)}"
        node_id = f"graph_split_node_{token_hex(4)}"
        target_id = f"graph_split_target_{token_hex(4)}"
        store = GraphStore(config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False)
        create_tool = build_graph_create_nodes_tool(
            config=GraphManageNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False),
            store=store,
        )
        update_tool = build_graph_update_node_tool(
            config=GraphManageNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False),
            store=store,
        )
        delete_tool = build_graph_delete_nodes_tool(
            config=GraphManageNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False),
            store=store,
        )
        read_tool = build_read_nodes_tool(config=ReadNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id), store=store)
        try:
            create_result = create_tool.invoke(
                GraphCreateNodesInput(
                    items=[
                        GraphCreateNodeAction(
                            ids=[node_id],
                            summary="拆分接口主节点",
                            body="拆分接口主节点正文。",
                            keyword_ops=[GraphKeywordOp(op="+", keywords=["split", "main"])],
                        ),
                        GraphCreateNodeAction(
                            ids=[target_id],
                            summary="拆分接口目标节点",
                            body="拆分接口目标节点正文。",
                            keyword_ops=[GraphKeywordOp(op="+", keywords=["split", "target"])],
                        ),
                    ]
                ).model_dump()
            )
            self.assertEqual(create_result["operation"], "graph_create_nodes")
            self.assertEqual(create_result["status"], "success")
            self.assertEqual(create_result["action_count"], 2)

            update_result = update_tool.invoke(
                GraphUpdateNodeInput(
                    id=node_id,
                    body="拆分接口主节点更新正文。",
                    edge_ops=[GraphEdgeOp(op="+", targets=[target_id], dist=0.4)],
                ).model_dump()
            )
            self.assertEqual(update_result["operation"], "graph_update_node")
            self.assertEqual(update_result["status"], "success")

            read_result = read_tool.invoke(ReadNodesInput(ids=[node_id], detail_mode="summary").model_dump())
            self.assertEqual(read_result["status"], "success")
            self.assertTrue(any(edge["neighbor_node_id"] == target_id for edge in read_result["results"][0]["edges"]))

            delete_result = delete_tool.invoke(GraphDeleteNodesInput(ids=[target_id]).model_dump())
            self.assertEqual(delete_result["operation"], "graph_delete_nodes")
            self.assertEqual(delete_result["status"], "success")
            read_deleted = read_tool.invoke(ReadNodesInput(ids=[target_id], detail_mode="summary").model_dump())
            self.assertEqual(read_deleted["status"], "error")
        finally:
            store.close()

    def test_useful_and_blocked_buckets_follow_old_state_behavior(self) -> None:
        run_id = f"graph_state_{token_hex(4)}"
        first_id = f"state_a_{token_hex(4)}"
        second_id = f"state_b_{token_hex(4)}"
        store = GraphStore(config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False)
        manage_tool = build_graph_manage_nodes_tool(
            config=GraphManageNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=False),
            store=store,
        )
        useful_tool = build_graph_mark_useful_tool(
            config=GraphMarkUsefulToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id),
            store=store,
        )
        blocked_tool = build_graph_mark_blocked_tool(
            config=GraphMarkBlockedToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id),
            store=store,
        )
        clear_tool = build_graph_clear_blocked_tool(config=GraphClearBlockedToolConfig(run_id=run_id))
        distance_tool = build_graph_distance_recall_tool(
            config=GraphDistanceRecallToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id),
            store=store,
        )
        try:
            result = manage_tool.invoke(
                GraphManageNodesInput(
                    actions=[
                        GraphNodeAction(
                            operation="create",
                            ids=[second_id],
                            summary="状态节点二",
                            body="状态节点二正文。",
                            keyword_ops=[GraphKeywordOp(op="+", keywords=["state"])],
                        ),
                        GraphNodeAction(
                            operation="create",
                            ids=[first_id],
                            summary="状态节点一",
                            body="状态节点一正文。",
                            keyword_ops=[GraphKeywordOp(op="+", keywords=["state"])],
                            edge_ops=[GraphEdgeOp(op="+", targets=[second_id], dist=0.4)],
                        ),
                    ]
                ).model_dump()
            )
            self.assertEqual(result["status"], "success")

            useful_result = _normalize_result(useful_tool.invoke(GraphMarkUsefulInput(node_ids=[first_id], rationale="优先关注").model_dump()))
            self.assertEqual(useful_result["status"], "success")
            blocked_result = _normalize_result(blocked_tool.invoke(GraphMarkBlockedInput(node_ids=[second_id], rationale="暂时跳过").model_dump()))
            self.assertEqual(blocked_result["status"], "success")
            clear_result = _normalize_result(clear_tool.invoke(GraphClearBlockedInput(node_ids=[second_id]).model_dump()))
            self.assertEqual(clear_result["status"], "success")

            distance_result = distance_tool.invoke(GraphDistanceRecallInput(anchor_node_id=first_id, max_distance=0.5).model_dump())
            self.assertEqual(distance_result["status"], "success")
            self.assertEqual(distance_result["candidate_count"], 1)
        finally:
            store.close()

    def test_graph_body_update_reembeds_when_embedding_profile_changes(self) -> None:
        run_id = f"graph_profile_{token_hex(4)}"
        node_id = f"graph_profile_node_{token_hex(4)}"
        base_store = GraphStore(config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=True)
        create_tool = build_graph_manage_nodes_tool(
            config=GraphManageNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=True),
            store=base_store,
        )
        create_result = create_tool.invoke(
            GraphManageNodesInput(
                actions=[
                    GraphNodeAction(
                        operation="create",
                        ids=[node_id],
                        summary="旧 profile 图节点",
                        body="旧 profile 图节点正文。",
                        keyword_ops=[GraphKeywordOp(op="+", keywords=["alpha", "beta"])],
                    )
                ]
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")
        base_store.close()

        switched_store = GraphStore(
            config_path=CONFIG_PATH,
            run_id=run_id,
            persist_keyword_embeddings=True,
            embedding_config_override={
                "provider": "ollama",
                "model": "qwen3-embedding:4b",
                "base_url": "http://127.0.0.1:11434",
                "dimensions": 2560,
            },
        )
        manage_tool = build_graph_manage_nodes_tool(
            config=GraphManageNodesToolConfig(neo4j_config_path=CONFIG_PATH, run_id=run_id, persist_keyword_embeddings=True),
            store=switched_store,
        )
        try:
            with patch.object(switched_store, "_embed_keywords", return_value=[[0.1] * 2560, [0.2] * 2560]) as embed_mock:
                update_result = manage_tool.invoke(
                    GraphManageNodesInput(
                        actions=[GraphNodeAction(operation="update", ids=[node_id], body="只改正文，但 profile 已切换。")]
                    ).model_dump()
                )
            self.assertEqual(update_result["status"], "success")
            embed_mock.assert_called_once_with(["alpha", "beta"], salt=f"{run_id}:{node_id}")
        finally:
            switched_store.close()

    def test_graph_mark_useful_does_not_mutate_state_when_bucket_overflows(self) -> None:
        class FakeStore:
            def _resolve_run_id(self, run_id):
                return run_id or "graph-mark-useful"

            def fetch_node(self, *, run_id, node_id, detail_mode):
                return {"node_id": node_id, "body": "x" * 12, "edges": []}

        tool = build_graph_mark_useful_tool(
            config=GraphMarkUsefulToolConfig(
                neo4j_config_path=CONFIG_PATH,
                run_id="graph-mark-useful",
                max_items=1,
                max_total_chars=8,
            ),
            store=FakeStore(),
        )
        result = _normalize_result(
            tool.func(
                node_ids=["node-a"],
                rationale="overflow",
                runtime=SimpleNamespace(state={"useful_items": {}}, tool_call_id="manual"),
            )
        )
        self.assertEqual(result["status"], "error")
        self.assertNotIn("state_update", result)

    def test_graph_mark_blocked_does_not_mutate_state_when_bucket_overflows(self) -> None:
        class FakeStore:
            def _resolve_run_id(self, run_id):
                return run_id or "graph-mark-blocked"

            def fetch_node(self, *, run_id, node_id, detail_mode):
                return {"node_id": node_id, "body": "x" * 12, "edges": []}

        tool = build_graph_mark_blocked_tool(
            config=GraphMarkBlockedToolConfig(
                neo4j_config_path=CONFIG_PATH,
                run_id="graph-mark-blocked",
                max_items=1,
                max_total_chars=8,
            ),
            store=FakeStore(),
        )
        result = _normalize_result(
            tool.func(
                node_ids=["node-a"],
                rationale="overflow",
                runtime=SimpleNamespace(state={"blocked_items": {}}, tool_call_id="manual"),
            )
        )
        self.assertEqual(result["status"], "error")
        self.assertNotIn("state_update", result)


if __name__ == "__main__":
    unittest.main()
