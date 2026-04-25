import unittest
import json
from types import SimpleNamespace

from middleware import MemoryCapabilityMiddleware
from server.memory_service import MemoryService
from tools.manage_memory import MemoryAction, MemoryBatchInput, build_manage_memory_tool


class MemoryToolTests(unittest.TestCase):
    def test_manage_memory_batch_create_update_delete(self) -> None:
        tool = build_manage_memory_tool()
        self.assertEqual(tool.name, "manage_memory")
        create_command = tool.invoke(
            MemoryBatchInput(
                items=[
                    MemoryAction(
                        operation="create",
                        label="偏好",
                        content="优先使用简洁输出。",
                    )
                ]
            ).model_dump()
        )
        create_payload = json.loads(create_command.update["messages"][0].content)
        self.assertEqual(create_payload["status"], "success")
        self.assertEqual(len(create_payload["memory"]), 1)
        memory_id = create_payload["memory"][0]["id"]

        update_command = tool.func(
            runtime=SimpleNamespace(state={"memory": create_payload["memory"]}, tool_call_id="manual"),
            items=[
                MemoryAction(
                    operation="update",
                    id=memory_id,
                    content="优先使用简洁、直接的输出。",
                )
            ],
        )
        update_payload = json.loads(update_command.update["messages"][0].content)
        self.assertEqual(update_payload["status"], "success")
        self.assertEqual(update_payload["memory"][0]["content"], "优先使用简洁、直接的输出。")

        delete_command = tool.func(
            runtime=SimpleNamespace(state={"memory": update_payload["memory"]}, tool_call_id="manual"),
            items=[MemoryAction(operation="delete", id=memory_id)],
        )
        delete_payload = json.loads(delete_command.update["messages"][0].content)
        self.assertEqual(delete_payload["status"], "success")
        self.assertEqual(delete_payload["memory"], [])

    def test_memory_action_validation_strips_and_rejects_bad_ids(self) -> None:
        action = MemoryAction(operation="create", label="  标签  ", content="  内容  ")
        self.assertEqual(action.label, "标签")
        self.assertEqual(action.content, "内容")
        with self.assertRaisesRegex(ValueError, "4-digit numeric string"):
            MemoryAction(operation="update", id="12a4", content="内容")

    def test_memory_service_and_middleware_render_state(self) -> None:
        middleware = MemoryCapabilityMiddleware()
        service = MemoryService()
        state = service.normalize_state(
            {
                "memory": [{"id": "1001", "label": "偏好", "content": "保持简洁"}],
                "monitor_trace": ["trace-a"],
            }
        )
        self.assertEqual(state["memory"][0]["id"], "1001")
        self.assertEqual(state["monitor_trace"], ["trace-a"])
        rendered = middleware.render_state_prompt(state)
        self.assertIn("<Memory>", rendered)
        self.assertIn("保持简洁", rendered)


if __name__ == "__main__":
    unittest.main()
