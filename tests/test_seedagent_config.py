import unittest

from Deepagents.SeedAgent.Agent.MainAgent import Config, build_middlewares, render_system_prompt
from Deepagents.SeedAgent.Agent.middlewares.knowledge_ingest import KnowledgeIngestMiddleware
from Deepagents.SeedAgent.Agent.server.memory_bridge import build_chunk_apply_config
from Deepagents.SeedAgent.Agent.server.memory_bridge import build_knowledge_manager_config_payload
from Deepagents.SeedAgent.AgentServer.service import load_service_config


class SeedAgentConfigTests(unittest.TestCase):
    def test_example_config_loads_as_complete_runtime_config(self) -> None:
        config = Config.load_config_seed_agent()

        self.assertEqual(config.agentName, "SeedAgent")
        self.assertEqual(config.chatModelProvider, "openai")
        self.assertEqual(config.neo4jUri, "neo4j://localhost:7687")
        self.assertTrue(config.enableKnowledgeIngestMiddleware)
        self.assertEqual(config.chunkHistoryLineCount, 4)
        self.assertEqual(config.graphQueryKeywordTopK, 6)
        self.assertIn("SeedAgent", render_system_prompt(config))

    def test_ingest_tool_is_owned_by_middleware(self) -> None:
        middleware = KnowledgeIngestMiddleware(runingConfig=KnowledgeIngestMiddleware.config(agentName="SeedAgent"))

        self.assertEqual(middleware.middleware.name, "knowledge_ingest")
        self.assertEqual([tool.name for tool in middleware.middleware.tools], ["ingest_knowledge_document"])

    def test_top_level_build_adds_no_free_tools_for_ingest_only(self) -> None:
        config = Config.load_config_seed_agent().model_copy(
            update={
                "enableMemoryMiddleware": False,
                "enableSkillsMiddleware": False,
                "enableKnowledgeManagerMiddleware": False,
                "enableReceiveMessagesMiddleware": False,
                "enableSendMessagesMiddleware": False,
                "enableAgentStepTraceMiddleware": False,
                "enableDebugTraceMiddleware": False,
            }
        )

        middlewares = build_middlewares(config=config, comm=None)
        self.assertEqual([middleware.name for middleware in middlewares], ["knowledge_ingest"])
        self.assertEqual([tool.name for tool in middlewares[0].tools], ["ingest_knowledge_document"])

    def test_service_config_keeps_port_with_agentserver(self) -> None:
        service_config = load_service_config()

        self.assertEqual(service_config.agent_name, "SeedAgent")
        self.assertEqual(service_config.host, "127.0.0.1")
        self.assertEqual(service_config.port, 8010)

    def test_chunk_apply_config_maps_readme_public_fields(self) -> None:
        config = Config.load_config_seed_agent().model_copy(
            update={
                "knowledgeRunId": "SeedAgent-knowledge",
                "chunkApplyDeriveDocumentRunId": False,
                "chunkHistoryLineCount": 7,
                "chunkWindowBackBytes": 333,
                "documentEdgeDistance": 0.9,
                "persistKeywordEmbeddings": False,
            }
        )

        chunk_config = build_chunk_apply_config(agent_name="SeedAgent", config=config)

        self.assertEqual(chunk_config.identity.base_run_id, "SeedAgent-knowledge")
        self.assertFalse(chunk_config.identity.derive_document_run_id)
        self.assertEqual(chunk_config.chunking.history_line_count, 7)
        self.assertEqual(chunk_config.chunking.window_back_bytes, 333)
        self.assertEqual(chunk_config.document_edge_distance, 0.9)
        self.assertFalse(chunk_config.persist_keyword_embeddings)

    def test_knowledge_manager_payload_maps_readme_public_fields(self) -> None:
        config = Config.load_config_seed_agent().model_copy(
            update={
                "knowledgeRunId": "SeedAgent-knowledge",
                "graphQueryKeywordTopK": 8,
                "graphQueryKeywordTopKLimit": 12,
                "graphQueryDistanceMaxDistance": 2.5,
                "managementDiscoveryMaxItems": 9,
                "documentWriteTraceLimit": 11,
            }
        )

        payload = build_knowledge_manager_config_payload(agent_name="SeedAgent", config=config)
        overrides = payload["tool"]["agent_overrides"]
        preset = overrides["graph_query"]["capability_preset"]

        self.assertEqual(payload["run_id"], "SeedAgent-knowledge")
        self.assertEqual(overrides["discovery"]["max_items"], 9)
        self.assertEqual(overrides["document_write"]["trace_limit"], 11)
        self.assertEqual(preset["keyword_top_k"], 8)
        self.assertEqual(preset["keyword_top_k_limit"], 12)
        self.assertEqual(preset["distance_max_distance"], 2.5)


if __name__ == "__main__":
    unittest.main()
