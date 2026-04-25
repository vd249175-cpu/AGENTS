import unittest
from pathlib import Path
from unittest.mock import patch

from middleware.knowledge_manager import (
    KnowledgeManagerCapabilityMiddleware,
    KnowledgeManagerCapabilityOverrides,
    KnowledgeManagerMiddlewareConfig,
    KnowledgeManagerMiddlewareOverride,
)
from server.knowledge_manager_runtime import KnowledgeManagerAgentOverrides, KnowledgeManagerModelOverride
from tools.manage_knowledge import ManageKnowledgeToolOverride


class KnowledgeManagerMiddlewarePublicApiTests(unittest.TestCase):
    def test_knowledge_manager_middleware_mounts_manage_knowledge_tool(self) -> None:
        run_id = "knowledge-manager-run"
        neo4j_config_path = Path("/tmp/knowledge-manager-database.json")
        captured: dict[str, object] = {}

        def fake_build_manage_knowledge_tool(*, config):
            captured["tool_config"] = config

            class _FakeTool:
                name = "manage_knowledge"

            return _FakeTool()

        with patch("middleware.knowledge_manager.build_manage_knowledge_tool", side_effect=fake_build_manage_knowledge_tool):
            middleware = KnowledgeManagerCapabilityMiddleware(
                config=KnowledgeManagerMiddlewareConfig(
                    neo4j_config_path=neo4j_config_path,
                    run_id=run_id,
                    trace_limit=16,
                ),
                overrides=KnowledgeManagerCapabilityOverrides(
                    middleware=KnowledgeManagerMiddlewareOverride(trace_limit=21),
                    tool=ManageKnowledgeToolOverride(
                        temperature=0.5,
                        debug=True,
                        inner_recursion_limit=33,
                        agent_overrides=KnowledgeManagerAgentOverrides(
                            model=KnowledgeManagerModelOverride(model="gpt-5.4-mini", temperature=0.2),
                            graph_query={"capability_preset": {"keyword_top_k": 8}},
                        ),
                    ),
                ),
            )

        self.assertEqual([tool.name for tool in middleware.tools], ["manage_knowledge"])
        self.assertIs(middleware.middleware, middleware)
        self.assertEqual(middleware.middlewareschema.name, "knowledge_manager")
        self.assertEqual(middleware.middlewareschema.affectedPrompts.Prompts[0].name, "knowledge_manager.guidance")
        self.assertEqual(middleware.config.run_id, run_id)
        self.assertEqual(middleware.config.trace_limit, 21)
        self.assertEqual(captured["tool_config"].run_id, run_id)
        self.assertEqual(captured["tool_config"].neo4j_config_path, neo4j_config_path)
        self.assertEqual(captured["tool_config"].temperature, 0.5)
        self.assertTrue(captured["tool_config"].debug)
        self.assertEqual(captured["tool_config"].inner_recursion_limit, 33)
        self.assertEqual(captured["tool_config"].agent_overrides.model.model, "gpt-5.4-mini")
        self.assertEqual(captured["tool_config"].agent_overrides.graph_query.capability_preset.keyword_top_k, 8)


if __name__ == "__main__":
    unittest.main()
