import json
import unittest
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, ToolMessage

from middleware.management_discovery import (
    ManagementDiscoveryMiddleware,
    ManagementDiscoveryMiddlewareConfig,
)


class _FakeRequest:
    def __init__(self, *, messages, state):
        self.messages = messages
        self.state = state

    def override(self, *, messages):
        return _FakeRequest(messages=messages, state=self.state)


class ManagementDiscoveryMiddlewareTests(unittest.TestCase):
    def test_before_model_extracts_discoveries_from_tool_messages(self) -> None:
        middleware = ManagementDiscoveryMiddleware(
            config=ManagementDiscoveryMiddlewareConfig(
                max_items=8,
                max_total_chars=1024,
                max_summary_chars=120,
                scan_message_limit=16,
            )
        )
        state = {
            "messages": [
                HumanMessage(content="请整理知识库"),
                ToolMessage(
                    content=json.dumps(
                        {
                            "operation": "create_chunk_document",
                            "status": "success",
                            "document_name": "alpha",
                            "chunk_count_after": 1,
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id="create-1",
                ),
                ToolMessage(
                    content=json.dumps(
                        {
                            "operation": "recall_nodes_by_keywords",
                            "status": "success",
                            "result_count": 2,
                            "results": [{"node_id": "graph-a"}, {"document_name": "alpha", "chunk_index": 0}],
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id="recall-1",
                ),
            ]
        }

        update = middleware.before_model(state, runtime=SimpleNamespace())

        self.assertIsNotNone(update)
        discoveries = update["management_discoveries"]
        self.assertEqual(len(discoveries), 2)
        self.assertEqual(discoveries[0]["source"], "create_chunk_document")
        self.assertIn("document:alpha", discoveries[0]["refs"])
        self.assertEqual(discoveries[1]["source"], "recall_nodes_by_keywords")
        self.assertIn("node:graph-a", discoveries[1]["refs"])

    def test_wrap_model_call_injects_discovery_prompts(self) -> None:
        middleware = ManagementDiscoveryMiddleware()
        request = _FakeRequest(
            messages=[
                HumanMessage(content="请检查图关系"),
                ToolMessage(
                    content=json.dumps(
                        {
                            "operation": "graph_manage_nodes",
                            "status": "success",
                            "action_count": 2,
                            "success_count": 2,
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id="graph-1",
                ),
            ],
            state={"management_discoveries": []},
        )
        captured = {}

        def handler(next_request):
            captured["messages"] = next_request.messages
            return SimpleNamespace(tool_calls=[])

        middleware.wrap_model_call(request, handler)

        system_messages = [message for message in captured["messages"] if getattr(message, "name", None)]
        names = [message.name for message in system_messages]
        self.assertIn("management_discovery.guidance", names)
        self.assertIn("management_discovery.state", names)
        state_prompt = next(message.content for message in system_messages if message.name == "management_discovery.state")
        self.assertIn("graph_manage_nodes", state_prompt)


if __name__ == "__main__":
    unittest.main()
