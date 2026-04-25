import unittest
from pathlib import Path
from secrets import token_hex
from unittest.mock import patch

from neo4j import GraphDatabase

from server.neo4j import DocumentStore
from tools.create_chunk_document import CreateChunkDocumentInput, CreateChunkDocumentToolConfig, build_create_chunk_document_tool
from tools.manage_chunks import (
    DeleteChunkAction,
    DeleteChunksInput,
    EdgeOp,
    InsertChunkAction,
    InsertChunksInput,
    KeywordOp,
    ChunkManageAction,
    ManageChunksInput,
    ManageChunksToolConfig,
    UpdateChunkAction,
    UpdateChunksInput,
    build_delete_chunks_tool,
    build_insert_chunks_tool,
    build_manage_chunks_tool,
    build_update_chunks_tool,
)
from tools.query_chunk_positions import QueryChunkPositionsInput, QueryChunkPositionsToolConfig, build_query_chunk_positions_tool


CONFIG_PATH = Path("/Users/apexwave/Desktop/memory/workspace/config/database_config.json")


class DocumentWriteToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._stores: list[DocumentStore] = []

    def tearDown(self) -> None:
        for store in self._stores:
            store.close()
        self._stores.clear()

    def _new_store(self, *, persist_keyword_embeddings: bool) -> DocumentStore:
        store = DocumentStore(
            config_path=CONFIG_PATH,
            persist_keyword_embeddings=persist_keyword_embeddings,
        )
        self._stores.append(store)
        return store

    def test_create_insert_update_delete_and_reindex(self) -> None:
        document_name = f"write_doc_{token_hex(4)}"
        store = self._new_store(persist_keyword_embeddings=False)
        create_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        manage_tool = build_manage_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        query_tool = build_query_chunk_positions_tool(
            config=QueryChunkPositionsToolConfig(neo4j_config_path=CONFIG_PATH),
            store=store,
        )

        create_result = create_tool.invoke(
            CreateChunkDocumentInput(
                document_name=document_name,
                summary="初始 chunk",
                body="初始内容。",
                keywords=["初始"],
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")

        insert_result = manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[
                    ChunkManageAction(
                        op="insert",
                        insert_after=0,
                        summary="插入 chunk",
                        body="插入内容。",
                        keywords=["插入"],
                    )
                ],
            ).model_dump()
        )
        self.assertEqual(insert_result["status"], "success")
        self.assertEqual(insert_result["chunk_count_before"], 1)
        self.assertEqual(insert_result["chunk_count_after"], 2)
        self.assertEqual(insert_result["results"][0]["operation"], "insert")
        self.assertEqual([chunk["chunk_index"] for chunk in insert_result["chunks"]], [0, 1])
        self.assert_document_order_edges(document_name, expected_count=1, expected_dist=0.3)

        update_result = manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[
                    ChunkManageAction(
                        op="update",
                        chunk_index=1,
                        summary="更新后的 chunk",
                        body="更新后的内容。",
                        keywords=["更新"],
                    )
                ],
            ).model_dump()
        )
        self.assertEqual(update_result["status"], "success")
        self.assertEqual(update_result["chunks"][1]["body"], "更新后的内容。")
        self.assert_document_order_edges(document_name, expected_count=1, expected_dist=0.3)

        delete_result = manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[ChunkManageAction(op="delete", chunk_index=0)],
            ).model_dump()
        )
        self.assertEqual(delete_result["status"], "success")
        self.assertEqual([chunk["chunk_index"] for chunk in delete_result["chunks"]], [0])
        self.assertEqual(delete_result["chunks"][0]["summary"], "更新后的 chunk")
        self.assert_document_order_edges(document_name, expected_count=0, expected_dist=0.3)

        query_result = query_tool.invoke(QueryChunkPositionsInput(document_name=document_name, positions=[0]).model_dump())
        self.assertEqual(query_result["chunk_count"], 1)
        self.assertEqual(query_result["chunks"][0]["body"], "更新后的内容。")

    def test_split_document_write_tools_reuse_manage_chunks_logic(self) -> None:
        document_name = f"split_write_doc_{token_hex(4)}"
        store = self._new_store(persist_keyword_embeddings=False)
        create_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        insert_tool = build_insert_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        update_tool = build_update_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        delete_tool = build_delete_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )

        create_result = create_tool.invoke(
            CreateChunkDocumentInput(
                document_name=document_name,
                summary="首段",
                body="首段正文。",
                keywords=["首段"],
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")

        insert_result = insert_tool.invoke(
            InsertChunksInput(
                document_name=document_name,
                items=[
                    InsertChunkAction(
                        insert_after=0,
                        summary="第二段",
                        body="第二段正文。",
                        keywords=["第二段"],
                    )
                ],
            ).model_dump()
        )
        self.assertEqual(insert_result["operation"], "insert_chunks")
        self.assertEqual(insert_result["status"], "success")
        self.assertEqual(insert_result["results"][0]["operation"], "insert")
        self.assertEqual(insert_result["chunk_count_after"], 2)

        update_result = update_tool.invoke(
            UpdateChunksInput(
                document_name=document_name,
                items=[
                    UpdateChunkAction(
                        chunk_index=1,
                        summary="第二段更新",
                        keyword_ops=[KeywordOp(op="+", keywords=["补充关键词"])],
                    )
                ],
            ).model_dump()
        )
        self.assertEqual(update_result["operation"], "update_chunks")
        self.assertEqual(update_result["status"], "success")
        self.assertEqual(update_result["results"][0]["operation"], "update")

        delete_result = delete_tool.invoke(
            DeleteChunksInput(document_name=document_name, items=[DeleteChunkAction(chunk_index=0)]).model_dump()
        )
        self.assertEqual(delete_result["operation"], "delete_chunks")
        self.assertEqual(delete_result["status"], "success")
        self.assertEqual(delete_result["results"][0]["operation"], "delete")
        self.assertEqual([chunk["chunk_index"] for chunk in delete_result["chunks"]], [0])

    def test_document_write_does_not_create_document_tag_or_tags(self) -> None:
        document_name = f"schema_doc_{token_hex(4)}"
        store = self._new_store(persist_keyword_embeddings=False)
        create_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        create_result = create_tool.invoke(
            CreateChunkDocumentInput(
                document_name=document_name,
                summary="结构测试",
                body="结构测试内容。",
                keywords=["结构"],
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")
        with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "1575338771")) as driver:
            with driver.session() as session:
                record = session.run(
                    """
                    MATCH (chunk:Chunk {document_name: $document_name})
                    WITH collect(chunk) AS chunks
                    MATCH (node)
                    RETURN
                        size(chunks) AS chunk_count,
                        sum(CASE WHEN 'Document' IN labels(node) THEN 1 ELSE 0 END) AS document_count,
                        sum(CASE WHEN 'Tag' IN labels(node) THEN 1 ELSE 0 END) AS tag_count,
                        any(item IN chunks WHERE item.tags IS NOT NULL) AS has_tags
                    """,
                    document_name=document_name,
                ).single()
        assert record is not None
        self.assertGreater(int(record["chunk_count"]), 0)
        self.assertEqual(int(record["document_count"]), 0)
        self.assertEqual(int(record["tag_count"]), 0)
        self.assertFalse(bool(record["has_tags"]))

    def test_manage_chunks_rebuilds_document_edges_without_touching_graph_edges(self) -> None:
        document_name = f"graph_safe_doc_{token_hex(4)}"
        marker = f"normal_edge_{token_hex(4)}"
        store = self._new_store(persist_keyword_embeddings=False)
        create_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        manage_tool = build_manage_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )

        create_result = create_tool.invoke(
            CreateChunkDocumentInput(
                document_name=document_name,
                summary="第一块",
                body="第一块内容。",
                keywords=[],
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")
        insert_result = manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[
                    ChunkManageAction(op="insert", insert_after=0, summary="第二块", body="第二块内容。"),
                    ChunkManageAction(op="insert", insert_after=1, summary="第三块", body="第三块内容。"),
                ],
            ).model_dump()
        )
        self.assertEqual(insert_result["status"], "success")
        chunks = insert_result["chunks"]
        self.assertEqual([chunk["chunk_index"] for chunk in chunks], [0, 1, 2])
        first_id = str(chunks[0]["id"])
        third_id = str(chunks[2]["id"])

        with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "1575338771")) as driver:
            with driver.session() as session:
                session.run(
                    """
                    MATCH (source:Chunk {document_name: $document_name, chunk_id: $first_id})
                    MATCH (target:Chunk {document_name: $document_name, chunk_id: $third_id})
                    MERGE (source)-[edge:LINKS {marker: $marker}]->(target)
                    SET edge.edge_kind = 'normal_graph'
                    """,
                    document_name=document_name,
                    first_id=first_id,
                    third_id=third_id,
                    marker=marker,
                ).consume()

        delete_result = manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[ChunkManageAction(op="delete", chunk_index=1)],
            ).model_dump()
        )
        self.assertEqual(delete_result["status"], "success")
        self.assertEqual([chunk["chunk_index"] for chunk in delete_result["chunks"]], [0, 1])

        with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "1575338771")) as driver:
            with driver.session() as session:
                record = session.run(
                    """
                    MATCH (source:Chunk {document_name: $document_name, chunk_id: $first_id})
                    MATCH (target:Chunk {document_name: $document_name, chunk_id: $third_id})
                    OPTIONAL MATCH (source)-[normal:LINKS {marker: $marker}]->(target)
                    OPTIONAL MATCH (source)-[order:DOCUMENT_NEXT]->(target)
                    RETURN count(normal) AS normal_count, count(order) AS order_count
                    """,
                    document_name=document_name,
                    first_id=first_id,
                    third_id=third_id,
                    marker=marker,
                ).single()
        assert record is not None
        self.assertEqual(int(record["normal_count"]), 1)
        self.assertEqual(int(record["order_count"]), 1)

    def test_manage_chunks_supports_edge_ops_without_touching_document_order(self) -> None:
        document_name = f"edge_ops_doc_{token_hex(4)}"
        store = self._new_store(persist_keyword_embeddings=False)
        create_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        manage_tool = build_manage_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        query_tool = build_query_chunk_positions_tool(
            config=QueryChunkPositionsToolConfig(neo4j_config_path=CONFIG_PATH),
            store=store,
        )

        create_result = create_tool.invoke(
            CreateChunkDocumentInput(
                document_name=document_name,
                summary="第 0 段",
                body="第 0 段正文。",
                keywords=[],
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")
        manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[
                    ChunkManageAction(op="insert", insert_after=0, summary="第 1 段", body="第 1 段正文。"),
                    ChunkManageAction(op="insert", insert_after=1, summary="第 2 段", body="第 2 段正文。"),
                ],
            ).model_dump()
        )

        link_result = manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[ChunkManageAction(op="update", chunk_index=0, edge_ops=[EdgeOp(op="+", targets=[2], dist=0.7)])],
            ).model_dump()
        )
        self.assertEqual(link_result["status"], "success")
        query_result = query_tool.invoke(QueryChunkPositionsInput(document_name=document_name, positions=[0], mode="summary").model_dump())
        links = [edge for edge in query_result["chunks"][0]["edges"] if edge["relation"] == "LINKS"]
        order_edges = [edge for edge in query_result["chunks"][0]["edges"] if edge["relation"] == "DOCUMENT_NEXT"]
        self.assertEqual(len(links), 1)
        link_positions = {links[0]["source_position"], links[0]["target_position"]}
        self.assertEqual(link_positions, {0, 2})
        self.assertEqual(float(links[0]["dist"]), 0.7)
        self.assertEqual(len(order_edges), 1)

        unlink_result = manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[ChunkManageAction(op="update", chunk_index=0, edge_ops=[EdgeOp(op="-", targets="all")])],
            ).model_dump()
        )
        self.assertEqual(unlink_result["status"], "success")
        query_after = query_tool.invoke(QueryChunkPositionsInput(document_name=document_name, positions=[0], mode="summary").model_dump())
        self.assertFalse([edge for edge in query_after["chunks"][0]["edges"] if edge["relation"] == "LINKS"])
        self.assertTrue([edge for edge in query_after["chunks"][0]["edges"] if edge["relation"] == "DOCUMENT_NEXT"])

    def test_manage_chunks_supports_keyword_ops(self) -> None:
        document_name = f"keyword_doc_{token_hex(4)}"
        store = self._new_store(persist_keyword_embeddings=False)
        create_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        manage_tool = build_manage_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )

        create_result = create_tool.invoke(
            CreateChunkDocumentInput(
                document_name=document_name,
                summary="关键词测试",
                body="关键词测试内容。",
                keywords=["alpha", "beta"],
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")

        plus_result = manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[ChunkManageAction(op="update", chunk_index=0, keyword_ops=[KeywordOp(op="+", keywords=["gamma", "alpha"])])],
            ).model_dump()
        )
        self.assertEqual(plus_result["chunks"][0]["keywords"], ["alpha", "beta", "gamma"])

        minus_result = manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[ChunkManageAction(op="update", chunk_index=0, keyword_ops=[KeywordOp(op="-", keywords=["beta"])])],
            ).model_dump()
        )
        self.assertEqual(minus_result["chunks"][0]["keywords"], ["alpha", "gamma"])

        replace_result = manage_tool.invoke(
            ManageChunksInput(
                document_name=document_name,
                actions=[ChunkManageAction(op="update", chunk_index=0, keyword_ops=[KeywordOp(op="replace", keywords=["delta"])])],
            ).model_dump()
        )
        self.assertEqual(replace_result["chunks"][0]["keywords"], ["delta"])

    def test_body_only_update_keeps_existing_keyword_vectors(self) -> None:
        document_name = f"keyword_keep_doc_{token_hex(4)}"
        store = self._new_store(persist_keyword_embeddings=True)
        create_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=True),
            store=store,
        )
        manage_tool = build_manage_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=True),
            store=store,
        )

        create_result = create_tool.invoke(
            CreateChunkDocumentInput(
                document_name=document_name,
                summary="关键词保留测试",
                body="初始正文。",
                keywords=["alpha", "beta"],
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")
        metadata_before = self.keyword_metadata_snapshot(document_name)
        self.assertEqual(metadata_before["providers"], ["openai"])
        self.assertEqual(metadata_before["models"], ["text-embedding-3-small"])
        self.assertEqual(metadata_before["dimensions"], [1536])

        with patch.object(store, "_embed_keywords", side_effect=AssertionError("body-only update should not re-embed")):
            update_result = manage_tool.invoke(
                ManageChunksInput(
                    document_name=document_name,
                    actions=[ChunkManageAction(op="update", chunk_index=0, body="只改正文，不改关键词。")],
                ).model_dump()
            )

        self.assertEqual(update_result["status"], "success")
        metadata_after = self.keyword_metadata_snapshot(document_name)
        self.assertEqual(metadata_after, metadata_before)

    def test_keyword_plus_only_embeds_new_keywords(self) -> None:
        document_name = f"keyword_plus_doc_{token_hex(4)}"
        store = self._new_store(persist_keyword_embeddings=True)
        create_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=True),
            store=store,
        )
        manage_tool = build_manage_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=True),
            store=store,
        )

        create_result = create_tool.invoke(
            CreateChunkDocumentInput(
                document_name=document_name,
                summary="关键词增量测试",
                body="初始正文。",
                keywords=["alpha", "beta"],
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")

        with patch.object(store, "_embed_keywords", return_value=[[0.1, 0.2, 0.3]]) as embed_mock:
            plus_result = manage_tool.invoke(
                ManageChunksInput(
                    document_name=document_name,
                    actions=[ChunkManageAction(op="update", chunk_index=0, keyword_ops=[KeywordOp(op="+", keywords=["gamma", "alpha"])])],
                ).model_dump()
            )

        self.assertEqual(plus_result["status"], "success")
        embed_mock.assert_called_once_with(["gamma"], salt=document_name)
        self.assertEqual(plus_result["chunks"][0]["keywords"], ["alpha", "beta", "gamma"])

    def test_switching_embedding_profile_triggers_full_reembed(self) -> None:
        document_name = f"keyword_profile_doc_{token_hex(4)}"
        base_store = self._new_store(persist_keyword_embeddings=True)
        create_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=True),
            store=base_store,
        )
        create_result = create_tool.invoke(
            CreateChunkDocumentInput(
                document_name=document_name,
                summary="profile 测试",
                body="初始正文。",
                keywords=["alpha", "beta"],
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")

        switched_store = DocumentStore(
            config_path=CONFIG_PATH,
            persist_keyword_embeddings=True,
            embedding_config_override={
                "provider": "ollama",
                "model": "qwen3-embedding:4b",
                "base_url": "http://127.0.0.1:11434",
                "dimensions": 2560,
            },
        )
        self._stores.append(switched_store)
        manage_tool = build_manage_chunks_tool(
            config=ManageChunksToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=True),
            store=switched_store,
        )
        with patch.object(switched_store, "_embed_keywords", return_value=[[0.1] * 2560, [0.2] * 2560]) as embed_mock:
            update_result = manage_tool.invoke(
                ManageChunksInput(
                    document_name=document_name,
                    actions=[ChunkManageAction(op="update", chunk_index=0, body="只改正文，但 profile 已切换。")],
                ).model_dump()
            )
        self.assertEqual(update_result["status"], "success")
        embed_mock.assert_called_once_with(["alpha", "beta"], salt=document_name)
        metadata_after = self.keyword_metadata_snapshot(document_name)
        self.assertEqual(metadata_after["providers"], ["ollama"])
        self.assertEqual(metadata_after["models"], ["qwen3-embedding:4b"])
        self.assertEqual(metadata_after["dimensions"], [2560])

    def assert_document_order_edges(self, document_name: str, *, expected_count: int, expected_dist: float) -> None:
        with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "1575338771")) as driver:
            with driver.session() as session:
                record = session.run(
                    """
                    MATCH (:Chunk {document_name: $document_name})-[edge:DOCUMENT_NEXT]->
                          (:Chunk {document_name: $document_name})
                    RETURN count(edge) AS edge_count, collect(DISTINCT edge.dist) AS dists
                    """,
                    document_name=document_name,
                ).single()
        assert record is not None
        self.assertEqual(int(record["edge_count"]), expected_count)
        if expected_count:
            self.assertEqual([float(item) for item in record["dists"]], [expected_dist])

    def keyword_metadata_snapshot(self, document_name: str) -> dict[str, list[object]]:
        with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "1575338771")) as driver:
            with driver.session() as session:
                record = session.run(
                    """
                    MATCH (:Chunk {document_name: $document_name})-[:HAS_KEYWORD]->
                          (keyword:KeywordNode {document_name: $document_name})
                    RETURN
                        collect(DISTINCT keyword.embedding_provider) AS providers,
                        collect(DISTINCT keyword.embedding_model) AS models,
                        collect(DISTINCT keyword.embedding_dimensions) AS dimensions
                    """,
                    document_name=document_name,
                ).single()
        assert record is not None
        return {
            "providers": sorted(str(item) for item in list(record["providers"] or []) if item is not None),
            "models": sorted(str(item) for item in list(record["models"] or []) if item is not None),
            "dimensions": sorted(int(item) for item in list(record["dimensions"] or []) if item is not None),
        }


if __name__ == "__main__":
    unittest.main()
