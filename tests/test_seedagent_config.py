import unittest

from Deepagents.SeedAgent.Agent.MainAgent import Config, EXAMPLE_CONFIG_PATH, build_middlewares, render_system_prompt
from Deepagents.SeedAgent.Agent.middlewares.knowledge_ingest import KnowledgeIngestMiddleware
from Deepagents.SeedAgent.Agent.middlewares.send_messages import Config as SendMessagesConfig
from Deepagents.SeedAgent.Agent.middlewares.send_messages import _format_peer_directory
from Deepagents.SeedAgent.Agent.middlewares.receive_messages import Config as ReceiveMessagesConfig
from Deepagents.SeedAgent.Agent.middlewares.receive_messages import Middleware as ReceiveMessagesMiddlewareImpl
from Deepagents.SeedAgent.Agent.tools.send_message_tool import ToolDescriptionSc, ToolInputSm, _task_payload
from Deepagents.SeedAgent.AgentServer.service import load_service_config
from Deepagents.KnowledgeSeedAgent.Agent.MainAgent import Config as KnowledgeSeedConfig
from Deepagents.KnowledgeSeedAgent.Agent.MainAgent import EXAMPLE_CONFIG_PATH as KNOWLEDGE_EXAMPLE_CONFIG_PATH
from Deepagents.KnowledgeSeedAgent.Agent.MainAgent import build_middlewares as build_knowledge_middlewares
from Deepagents.KnowledgeSeedAgent.Agent.MainAgent import render_system_prompt as render_knowledge_system_prompt
from Deepagents.KnowledgeSeedAgent.Agent.server.memory_bridge import build_chunk_apply_config
from Deepagents.KnowledgeSeedAgent.Agent.server.memory_bridge import build_knowledge_manager_config_payload
from types import SimpleNamespace


class SeedAgentConfigTests(unittest.TestCase):
    def test_seed_example_config_loads_as_knowledge_runtime_config(self) -> None:
        config = Config.load_config_seed_agent(EXAMPLE_CONFIG_PATH)

        self.assertNotIn("agentName", Config.model_fields)
        self.assertEqual(config.chatModelProvider, "openai")
        self.assertEqual(config.neo4jUri, "neo4j://localhost:7687")
        self.assertFalse(config.enableKnowledgeManagerMiddleware)
        self.assertFalse(config.enableKnowledgeIngestMiddleware)
        self.assertIsNone(config.checkpointPath)
        self.assertEqual(config.chunkHistoryLineCount, 4)
        self.assertEqual(config.graphQueryKeywordTopK, 6)
        self.assertIsNone(render_system_prompt(config))

    def test_knowledge_seed_keeps_memory_tools_enabled(self) -> None:
        config = KnowledgeSeedConfig.load_config_knowledge_seed_agent(KNOWLEDGE_EXAMPLE_CONFIG_PATH)

        self.assertNotIn("agentName", KnowledgeSeedConfig.model_fields)
        self.assertTrue(config.enableKnowledgeManagerMiddleware)
        self.assertTrue(config.enableKnowledgeIngestMiddleware)
        self.assertIsNone(config.checkpointPath)
        self.assertIsNone(render_knowledge_system_prompt(config))

    def test_knowledge_seed_top_prompt_only_uses_agents_md_memory_source(self) -> None:
        config = KnowledgeSeedConfig.load_config_knowledge_seed_agent(KNOWLEDGE_EXAMPLE_CONFIG_PATH).model_copy(
            update={
                "enableSkillsMiddleware": False,
                "enableKnowledgeManagerMiddleware": False,
                "enableKnowledgeIngestMiddleware": False,
                "enableReceiveMessagesMiddleware": False,
                "enableSendMessagesMiddleware": False,
                "enableAgentStepTraceMiddleware": False,
                "enableDebugTraceMiddleware": False,
            }
        )

        middlewares = build_knowledge_middlewares(config=config, comm=None)

        self.assertEqual([middleware.name for middleware in middlewares], ["MemoryMiddleware"])
        self.assertEqual(getattr(middlewares[0], "sources", None), ["/brain/AGENTS.md"])

    def test_ingest_tool_is_owned_by_middleware(self) -> None:
        middleware = KnowledgeIngestMiddleware(
            runingConfig=KnowledgeIngestMiddleware.config(runtimeAgentName="SeedAgent")
        )

        self.assertEqual(middleware.middleware.name, "knowledge_ingest")
        self.assertEqual([tool.name for tool in middleware.middleware.tools], ["ingest_knowledge_document"])

    def test_receive_messages_persists_inbox_as_state_message(self) -> None:
        class FakeComm:
            def recv(self):
                return [
                    {
                        "message_id": "mail-1",
                        "from": "KnowledgeSeedAgent",
                        "to": "SeedAgent",
                        "type": "message",
                        "content": "reply token beta-42",
                        "attachments": [{"link": "/workspace/mail/KnowledgeSeedAgent__abc/novel.txt"}],
                    }
                ]

            def peers(self):
                return ["KnowledgeSeedAgent"]

        events = []
        middleware = ReceiveMessagesMiddlewareImpl(
            comm=FakeComm(),
            runingConfig=ReceiveMessagesConfig(enabled=True, maxInboxItems=5),
        )

        update = middleware.before_model({"messages": []}, SimpleNamespace(stream_writer=events.append))

        self.assertIsNotNone(update)
        self.assertEqual(update["receiveMessagesLastCount"], 1)
        self.assertIn("reply token beta-42", update["receiveMessagesLastInbox"])
        self.assertIn("UTF-8", update["receiveMessagesLastInbox"])
        self.assertIn("read_file", update["receiveMessagesLastInbox"])
        self.assertIn("bytes ... head b", update["receiveMessagesLastInbox"])
        self.assertIn("/workspace/mail/KnowledgeSeedAgent__abc/novel.txt", update["receiveMessagesLastInbox"])
        self.assertEqual(update["messages"][0].name, "agent_inbox")
        self.assertIn("reply token beta-42", update["messages"][0].content)
        self.assertEqual(events[0]["event"], "inbox_persisted")

    def test_send_message_tool_attachment_guidance_is_explicit(self) -> None:
        tool_description = ToolDescriptionSc().toolDescription
        attachment_description = ToolInputSm.model_fields["attachments"].description or ""
        guidance = SendMessagesConfig.load_config_send_messages().guidancePrompt

        combined = "\n".join([tool_description, attachment_description, guidance])

        self.assertIn("attachments", combined)
        self.assertIn("/workspace/...", combined)
        self.assertIn("content", combined)
        self.assertIn("不会传输文件", guidance)
        self.assertIn("Mentioning a /workspace/... path in content is not enough", attachment_description)

    def test_send_message_peer_directory_includes_online_status_and_agent_card(self) -> None:
        prompt = _format_peer_directory(
            ["KnowledgeSeedAgent"],
            [
                {
                    "agent_name": "KnowledgeSeedAgent",
                    "online": True,
                    "status": "running",
                    "phase": "ready",
                    "card": {
                        "agent_name": "KnowledgeSeedAgent",
                        "capabilities": [
                            {
                                "title": "Memory Management",
                                "content": "Manage and query memory graph content.",
                            }
                        ],
                    },
                }
            ],
        )

        self.assertIn("KnowledgeSeedAgent", prompt)
        self.assertIn("online", prompt)
        self.assertIn("Memory Management", prompt)
        self.assertIn("send_message_to_agent", prompt)
        self.assertIn("dst", prompt)

    def test_task_payload_uses_attachments_as_only_file_transfer_field(self) -> None:
        payload = _task_payload(
            task_info={
                "title": "Review report",
                "goal": "Read the attached report.",
                "description": "Please inspect /workspace/notes/report.md.",
                "owner": "SeedAgent",
                "deliverables": [{"link": "/workspace/notes/report.md", "summary": "legacy field"}],
            },
            content=None,
            owner="SeedAgent",
        )

        self.assertNotIn("deliverables", payload)

        auto_payload = _task_payload(
            task_info=None,
            content="Read the attached report.",
            owner="SeedAgent",
        )

        self.assertNotIn("deliverables", auto_payload)

    def test_top_level_build_adds_no_free_tools_for_ingest_only(self) -> None:
        config = Config.load_config_seed_agent(EXAMPLE_CONFIG_PATH).model_copy(
            update={
                "enableMemoryMiddleware": False,
                "enableSkillsMiddleware": False,
                "enableKnowledgeManagerMiddleware": False,
                "enableKnowledgeIngestMiddleware": True,
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
        self.assertEqual(service_config.port, 8011)

    def test_chunk_apply_config_maps_readme_public_fields(self) -> None:
        config = KnowledgeSeedConfig.load_config_knowledge_seed_agent(KNOWLEDGE_EXAMPLE_CONFIG_PATH).model_copy(
            update={
                "knowledgeRunId": "KnowledgeSeedAgent-knowledge",
                "chunkApplyDeriveDocumentRunId": False,
                "chunkHistoryLineCount": 7,
                "chunkWindowBackBytes": 333,
                "documentEdgeDistance": 0.9,
                "persistKeywordEmbeddings": False,
            }
        )

        chunk_config = build_chunk_apply_config(agent_name="KnowledgeSeedAgent", config=config)

        self.assertEqual(chunk_config.identity.base_run_id, "KnowledgeSeedAgent-knowledge")
        self.assertFalse(chunk_config.identity.derive_document_run_id)
        self.assertEqual(chunk_config.chunking.history_line_count, 7)
        self.assertEqual(chunk_config.chunking.window_back_bytes, 333)
        self.assertEqual(chunk_config.document_edge_distance, 0.9)
        self.assertFalse(chunk_config.persist_keyword_embeddings)

    def test_knowledge_manager_payload_maps_readme_public_fields(self) -> None:
        config = KnowledgeSeedConfig.load_config_knowledge_seed_agent(KNOWLEDGE_EXAMPLE_CONFIG_PATH).model_copy(
            update={
                "knowledgeRunId": "KnowledgeSeedAgent-knowledge",
                "graphQueryKeywordTopK": 8,
                "graphQueryKeywordTopKLimit": 12,
                "graphQueryDistanceMaxDistance": 2.5,
                "managementDiscoveryMaxItems": 9,
                "documentWriteTraceLimit": 11,
            }
        )

        payload = build_knowledge_manager_config_payload(agent_name="KnowledgeSeedAgent", config=config)
        overrides = payload["tool"]["agent_overrides"]
        preset = overrides["graph_query"]["capability_preset"]

        self.assertEqual(payload["run_id"], "KnowledgeSeedAgent-knowledge")
        self.assertEqual(overrides["discovery"]["max_items"], 9)
        self.assertEqual(overrides["document_write"]["trace_limit"], 11)
        self.assertEqual(preset["keyword_top_k"], 8)
        self.assertEqual(preset["keyword_top_k_limit"], 12)
        self.assertEqual(preset["distance_max_distance"], 2.5)


if __name__ == "__main__":
    unittest.main()
