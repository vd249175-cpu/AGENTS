import unittest
from pathlib import Path
from secrets import token_hex

from server.neo4j import DocumentStore
from tools.create_chunk_document import CreateChunkDocumentInput, CreateChunkDocumentToolConfig, build_create_chunk_document_tool
from tools.list_chunk_documents import ListChunkDocumentsInput, ListChunkDocumentsToolConfig, build_list_chunk_documents_tool
from tools.manage_chunks import ChunkManageAction, ManageChunksInput, ManageChunksToolConfig, build_manage_chunks_tool
from tools.query_chunk_positions import (
    QueryChunkPositionsInput,
    QueryChunkPositionsItem,
    QueryChunkPositionsToolConfig,
    build_query_chunk_positions_tool,
)


CONFIG_PATH = Path("/Users/apexwave/Desktop/memory/workspace/config/database_config.json")


class DocumentQueryToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._stores: list[DocumentStore] = []

    def tearDown(self) -> None:
        for store in self._stores:
            store.close()
        self._stores.clear()

    def _new_store(self) -> DocumentStore:
        store = DocumentStore(config_path=CONFIG_PATH, persist_keyword_embeddings=False)
        self._stores.append(store)
        return store

    def test_list_and_query_chunk_documents(self) -> None:
        document_name = f"query_doc_{token_hex(4)}"
        store = self._new_store()
        create_tool = build_create_chunk_document_tool(
            config=CreateChunkDocumentToolConfig(neo4j_config_path=CONFIG_PATH, persist_keyword_embeddings=False),
            store=store,
        )
        list_tool = build_list_chunk_documents_tool(
            config=ListChunkDocumentsToolConfig(neo4j_config_path=CONFIG_PATH),
            store=store,
        )
        query_tool = build_query_chunk_positions_tool(
            config=QueryChunkPositionsToolConfig(neo4j_config_path=CONFIG_PATH),
            store=store,
        )
        create_result = create_tool.invoke(
            CreateChunkDocumentInput(
                document_name=f"{document_name}.md",
                summary="查询测试文档",
                body="第一段内容。",
                keywords=["查询", "测试"],
            ).model_dump()
        )
        self.assertEqual(create_result["status"], "success")
        self.assertEqual(create_result["document_name"], document_name)

        list_result = list_tool.invoke(ListChunkDocumentsInput().model_dump())
        self.assertEqual(list_result["status"], "success")
        document_names = {str(item["document_name"]) for item in list_result["documents"]}
        self.assertIn(document_name, document_names)

        query_result = query_tool.invoke(QueryChunkPositionsInput(document_name=document_name, positions=[0]).model_dump())
        self.assertEqual(query_result["status"], "success")
        self.assertEqual(query_result["chunk_count"], 1)
        self.assertEqual(query_result["chunks"][0]["chunk_index"], 0)
        self.assertEqual(query_result["chunks"][0]["body"], "第一段内容。")

    def test_query_chunk_positions_supports_batch_and_ranges(self) -> None:
        first_document = f"range_doc_{token_hex(4)}"
        second_document = f"range_doc_{token_hex(4)}"
        store = self._new_store()
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

        for document_name in (first_document, second_document):
            create_tool.invoke(
                CreateChunkDocumentInput(
                    document_name=document_name,
                    summary="第 0 段",
                    body="第 0 段正文。",
                    keywords=["query"],
                ).model_dump()
            )
            manage_tool.invoke(
                ManageChunksInput(
                    document_name=document_name,
                    actions=[
                        ChunkManageAction(op="insert", insert_after=0, summary="第 1 段", body="第 1 段正文。"),
                        ChunkManageAction(op="insert", insert_after=1, summary="第 2 段", body="第 2 段正文。"),
                    ],
                ).model_dump()
            )

        batch_result = query_tool.invoke(
            QueryChunkPositionsInput(
                items=[
                    QueryChunkPositionsItem(document_name=first_document, positions=[0, [1, 2]], mode="detail"),
                    QueryChunkPositionsItem(document_name=second_document, positions=[[1, 2]], mode="summary"),
                ]
            ).model_dump()
        )
        self.assertEqual(batch_result["status"], "success")
        self.assertEqual(batch_result["item_count"], 2)
        self.assertEqual(batch_result["results"][0]["requested_positions"], [0, 1, 2])
        self.assertEqual(batch_result["results"][0]["found_positions"], [0, 1, 2])
        self.assertEqual(batch_result["results"][0]["chunks"][0]["body"], "第 0 段正文。")
        self.assertEqual(batch_result["results"][1]["requested_positions"], [1, 2])
        self.assertNotIn("body", batch_result["results"][1]["chunks"][0])
        self.assertIn("edges", batch_result["results"][1]["chunks"][0])
        self.assertTrue(batch_result["results"][1]["chunks"][0]["edges"])



if __name__ == "__main__":
    unittest.main()
