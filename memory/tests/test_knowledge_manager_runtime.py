import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from server.knowledge_manager_runtime import KnowledgeManagerAgentOverrides, KnowledgeManagerModelOverride, create_knowledge_manager_agent


class KnowledgeManagerRuntimeTests(unittest.TestCase):
    def test_create_knowledge_manager_agent_applies_model_and_middleware_overrides(self) -> None:
        captured: dict[str, object] = {"store_kwargs": []}

        class FakeStore:
            def __init__(self, **kwargs):
                captured["store_kwargs"].append(kwargs)

        def fake_init_chat_model(**kwargs):
            captured["chat_kwargs"] = kwargs
            return SimpleNamespace(name="chat-model")

        def fake_create_agent(**kwargs):
            captured.update(kwargs)
            return kwargs

        with (
            patch("server.knowledge_manager_runtime.get_chat_model_config", return_value={
                "provider": "openai",
                "model": "gpt-5.4-nano",
                "base_url": "https://default.invalid/v1",
                "api_key": "sk-default",
            }),
            patch("server.knowledge_manager_runtime.init_chat_model", side_effect=fake_init_chat_model),
            patch("middleware.document_query.DocumentStore", FakeStore),
            patch("middleware.document_write.DocumentStore", FakeStore),
            patch("middleware.graph_query.GraphStore", FakeStore),
            patch("middleware.graph_write.GraphStore", FakeStore),
            patch("server.knowledge_manager_runtime.create_agent", side_effect=fake_create_agent),
        ):
            result = create_knowledge_manager_agent(
                model=None,
                run_id="runtime-run",
                neo4j_config_path=Path("/tmp/runtime-database.json"),
                temperature=0.1,
                debug=False,
                overrides=KnowledgeManagerAgentOverrides(
                    model=KnowledgeManagerModelOverride(
                        model="gpt-5.4-mini",
                        model_provider="openai",
                        base_url="https://override.invalid/v1",
                        api_key="sk-override",
                        temperature=0.8,
                    ),
                    embedding={
                        "provider": "openai",
                        "model": "text-embedding-3-small",
                        "base_url": "https://embedding.invalid/v1",
                        "api_key": "sk-embedding",
                        "dimensions": 256,
                    },
                    system_prompt="override system prompt",
                    debug=True,
                    document_query={"trace_limit": 9},
                    graph_query={"capability_preset": {"keyword_top_k": 11}},
                ),
            )

        self.assertEqual(result["name"], "knowledge-manager-agent")
        self.assertEqual(captured["chat_kwargs"]["model"], "gpt-5.4-mini")
        self.assertEqual(captured["chat_kwargs"]["model_provider"], "openai")
        self.assertEqual(captured["chat_kwargs"]["base_url"], "https://override.invalid/v1")
        self.assertEqual(captured["chat_kwargs"]["api_key"], "sk-override")
        self.assertEqual(captured["chat_kwargs"]["temperature"], 0.8)
        self.assertEqual(captured["system_prompt"], "override system prompt")
        self.assertTrue(captured["debug"])

        middleware = captured["middleware"]
        self.assertEqual([type(item).__name__ for item in middleware], [
            "ManagementDiscoveryMiddleware",
            "DocumentQueryCapabilityMiddleware",
            "DocumentWriteCapabilityMiddleware",
            "GraphQueryCapabilityMiddleware",
            "GraphWriteCapabilityMiddleware",
        ])
        self.assertEqual(middleware[1].config.trace_limit, 9)
        self.assertEqual([tool.name for tool in middleware[2].tools], ["create_chunk_document", "insert_chunks", "update_chunks", "delete_chunks"])
        self.assertEqual(middleware[2].config.embedding_model, "text-embedding-3-small")
        self.assertEqual(middleware[2].config.embedding_dimensions, 256)
        self.assertEqual(middleware[3].config.capability_preset.keyword_top_k, 11)
        self.assertEqual(middleware[3].config.capability_preset.embedding_model, "text-embedding-3-small")
        self.assertEqual([tool.name for tool in middleware[4].tools], ["graph_create_nodes", "graph_update_node", "graph_delete_nodes", "read_nodes"])
        self.assertEqual(middleware[4].config.embedding_dimensions, 256)
        self.assertEqual(len(captured["store_kwargs"]), 4)
        embedding_store_kwargs = [kwargs for kwargs in captured["store_kwargs"] if kwargs.get("embedding_config_override")]
        self.assertTrue(embedding_store_kwargs)
        self.assertEqual(embedding_store_kwargs[0]["embedding_config_override"]["model"], "text-embedding-3-small")
        self.assertEqual(embedding_store_kwargs[0]["embedding_config_override"]["dimensions"], 256)


if __name__ == "__main__":
    unittest.main()
