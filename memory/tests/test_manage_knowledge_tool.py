import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from server.knowledge_manager_runtime import KnowledgeManagerAgentOverrides, KnowledgeManagerModelOverride
from tools.manage_knowledge import ManageKnowledgeToolConfig, ManageKnowledgeToolOverride, build_manage_knowledge_tool


class ManageKnowledgeToolTests(unittest.TestCase):
    def test_manage_knowledge_passes_parent_dialogue_and_returns_full_useful_items(self) -> None:
        captured: dict[str, object] = {}

        class FakeAgent:
            def invoke(self, state, config):
                captured["state"] = state
                captured["config"] = config
                return {
                    "messages": [
                        AIMessage(content="管理完成，建议主 agent 继续跟进节点关系。"),
                        ToolMessage(
                            content='{"operation":"graph_mark_useful","status":"success","message":"ok"}',
                            tool_call_id="useful-1",
                        ),
                    ],
                    "useful_items": {
                        "node-a": {
                            "node_id": "node-a",
                            "summary": "节点A",
                            "body": "完整正文A",
                            "edges": [{"neighbor_node_id": "node-b", "dist": 0.3}],
                        }
                    },
                    "blocked_items": {},
                    "management_discoveries": [
                        {"id": "disc-1", "source": "graph_mark_useful", "summary": "记录了 useful 节点。", "refs": ["node:node-a"]}
                    ],
                }

        with patch("tools.manage_knowledge.create_knowledge_manager_agent", return_value=FakeAgent()) as create_agent_mock:
            tool = build_manage_knowledge_tool(
                config=ManageKnowledgeToolConfig.load().model_copy(
                    update={
                        "temperature": 0.2,
                        "inner_recursion_limit": 19,
                        "agent_overrides": KnowledgeManagerAgentOverrides(
                            model=KnowledgeManagerModelOverride(model="gpt-5.4", temperature=0.1),
                            debug=True,
                        ),
                    }
                )
            )
            result = tool.func(
                target="检查并整理知识库中的关键节点",
                runtime=SimpleNamespace(
                    state={
                        "messages": [
                            SystemMessage(content="父级 system prompt"),
                            HumanMessage(content="请帮我处理知识库"),
                            AIMessage(content="我会调用工具"),
                            ToolMessage(content='{"operation":"query_chunk_positions"}', tool_call_id="outer-tool"),
                            AIMessage(content="", tool_calls=[{"id": "call-1", "name": "manage_chunks", "args": {}}]),
                        ]
                    },
                    tool_call_id="manual",
                ),
            )

        create_agent_mock.assert_called_once()
        self.assertEqual(create_agent_mock.call_args.kwargs["temperature"], 0.2)
        self.assertTrue(create_agent_mock.call_args.kwargs["overrides"].debug)
        self.assertEqual(create_agent_mock.call_args.kwargs["overrides"].model.model, "gpt-5.4")
        self.assertEqual(result["operation"]["mark_useful_count"], 1)
        self.assertEqual(result["operation"]["create_count"], 0)
        self.assertIn("useful 标记 1", result["operation"]["summary"])
        self.assertIn("知识管理已完成", result["message"])
        self.assertIn("管理完成，建议主 agent 继续跟进节点关系。", result["message"])
        self.assertNotIn("useful_count", result["message"])
        self.assertNotIn("useful_count", result)
        self.assertEqual(set(result["useful_items"][0].keys()), {"node_id", "body", "edges"})
        self.assertEqual(result["useful_items"][0]["body"], "完整正文A")
        self.assertEqual(result["useful_items"][0]["edges"][0]["neighbor_node_id"], "node-b")
        self.assertEqual(captured["config"]["recursion_limit"], 19)
        delegated_messages = captured["state"]["messages"]
        self.assertEqual(delegated_messages[0].content, "请帮我处理知识库")
        self.assertEqual(len(delegated_messages), 3)
        self.assertIn("目标：检查并整理知识库中的关键节点", delegated_messages[-1].content)

    def test_manage_knowledge_tool_override_merges_all_construction_layers(self) -> None:
        streamed_events: list[object] = []

        class FakeAgent:
            def stream(self, state, config, stream_mode):
                yield {"state": state, "config": config, "stream_mode": stream_mode}

            def get_state(self, config):
                return SimpleNamespace(values={"messages": [AIMessage(content="已完成")]})

        with patch("tools.manage_knowledge.create_knowledge_manager_agent", return_value=FakeAgent()) as create_agent_mock:
            tool = build_manage_knowledge_tool(
                config=ManageKnowledgeToolConfig.load(),
                overrides=ManageKnowledgeToolOverride(
                    temperature=0.7,
                    debug=True,
                    stream_inner_agent=True,
                    inner_recursion_limit=23,
                    agent_overrides=KnowledgeManagerAgentOverrides(
                        model=KnowledgeManagerModelOverride(
                            model="gpt-5.4-mini",
                            model_provider="openai",
                            base_url="https://example.invalid/v1",
                            api_key="sk-test",
                            temperature=0.6,
                        ),
                        document_query={"trace_limit": 9},
                        graph_query={"capability_preset": {"keyword_top_k": 7}},
                    ),
                ),
            )
            result = tool.func(
                target="只做 override 校验",
                runtime=SimpleNamespace(
                    state={"messages": [HumanMessage(content="继续")]},
                    tool_call_id="manual",
                    stream_writer=streamed_events.append,
                ),
            )

        self.assertIn("知识管理已完成", result["message"])
        self.assertIn("summary", result["operation"])
        create_agent_mock.assert_called_once()
        self.assertEqual(create_agent_mock.call_args.kwargs["temperature"], 0.7)
        self.assertTrue(create_agent_mock.call_args.kwargs["debug"])
        overrides = create_agent_mock.call_args.kwargs["overrides"]
        self.assertEqual(overrides.model.model, "gpt-5.4-mini")
        self.assertEqual(overrides.model.base_url, "https://example.invalid/v1")
        self.assertEqual(overrides.document_query.trace_limit, 9)
        self.assertEqual(overrides.graph_query.capability_preset.keyword_top_k, 7)
        self.assertEqual(len(streamed_events), 1)
        self.assertEqual(streamed_events[0]["type"], "inner_agent_update")


if __name__ == "__main__":
    unittest.main()
