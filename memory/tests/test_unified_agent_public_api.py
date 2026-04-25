import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.unified_agent import DEFAULT_UNIFIED_AGENT_SYSTEM_PROMPT, create_unified_agent


class UnifiedAgentPublicApiTests(unittest.TestCase):
    def test_create_unified_agent_lets_middlewares_bind_their_own_tools(self) -> None:
        run_id = "unified-agent-run"
        neo4j_config_path = Path("/tmp/unified-agent-database.json")
        captured: dict[str, object] = {"store_kwargs": []}

        class FakeStore:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                captured["store_kwargs"].append(kwargs)

        def fake_chat_model(**kwargs):
            captured["chat_kwargs"] = kwargs
            return SimpleNamespace(name="chat-model")

        def fake_create_agent(**kwargs):
            captured.update(kwargs)
            return kwargs

        with (
            patch("agents.unified_agent.get_chat_model_config", return_value={
                "provider": "openai",
                "model": "gpt-5.4-nano",
                "base_url": "https://example.invalid/v1",
                "api_key": "sk-test",
            }),
            patch("agents.unified_agent.init_chat_model", side_effect=fake_chat_model),
            patch("middleware.document_query.DocumentStore", FakeStore),
            patch("middleware.document_write.DocumentStore", FakeStore),
            patch("middleware.graph_query.GraphStore", FakeStore),
            patch("middleware.graph_write.GraphStore", FakeStore),
            patch("agents.unified_agent.create_agent", side_effect=fake_create_agent),
        ):
            result = create_unified_agent(
                model=None,
                run_id=run_id,
                neo4j_config_path=neo4j_config_path,
                include_memory=True,
            )

        self.assertEqual(result["name"], "unified-agent")
        self.assertIn("联合管理 agent", captured["system_prompt"])

        middleware = captured["middleware"]
        middleware_names = [type(item).__name__ for item in middleware]
        self.assertEqual(
            middleware_names,
            [
                "ToolRetryMiddleware",
                "MemoryCapabilityMiddleware",
                "DocumentQueryCapabilityMiddleware",
                "DocumentWriteCapabilityMiddleware",
                "GraphQueryCapabilityMiddleware",
                "GraphWriteCapabilityMiddleware",
            ],
        )

        memory_middleware = middleware[1]
        self.assertEqual([tool.name for tool in memory_middleware.tools], ["manage_memory"])

        document_query_middleware = middleware[2]
        self.assertEqual([tool.name for tool in document_query_middleware.tools], ["list_chunk_documents", "query_chunk_positions"])
        self.assertEqual(document_query_middleware.config.run_id, run_id)

        document_write_middleware = middleware[3]
        self.assertEqual(
            [tool.name for tool in document_write_middleware.tools],
            ["create_chunk_document", "insert_chunks", "update_chunks", "delete_chunks"],
        )
        self.assertEqual(document_write_middleware.config.run_id, run_id)

        graph_query_middleware = middleware[4]
        self.assertEqual(
            [tool.name for tool in graph_query_middleware.tools],
            [
                "keyword_recall",
                "graph_distance_recall",
                "graph_mark_useful",
                "graph_mark_blocked",
                "graph_clear_blocked",
            ],
        )
        self.assertEqual(graph_query_middleware.config.run_id, run_id)

        graph_write_middleware = middleware[5]
        self.assertEqual(
            [tool.name for tool in graph_write_middleware.tools],
            ["graph_create_nodes", "graph_update_node", "graph_delete_nodes", "read_nodes"],
        )
        self.assertEqual(graph_write_middleware.config.run_id, run_id)

        self.assertEqual(len(captured["store_kwargs"]), 4)
        self.assertEqual({kwargs["run_id"] for kwargs in captured["store_kwargs"]}, {run_id})
        self.assertIn(DEFAULT_UNIFIED_AGENT_SYSTEM_PROMPT.splitlines()[0], captured["system_prompt"])


if __name__ == "__main__":
    unittest.main()
