from __future__ import annotations

import unittest

from agents import build_chunking_agent
from middleware import ChunkingCapabilityMiddleware
from tools.split_chunk import SplitChunkAction, SplitChunkBatchInput, build_window_view


class ChunkingAgentTests(unittest.TestCase):
    def test_window_lines_are_wrapped_and_numbered(self) -> None:
        middleware = ChunkingCapabilityMiddleware()
        body = (
            "这是一个非常长非常长非常长非常长非常长的第一行，用来测试自动换行是否生效。\n"
            "第二行继续提供一些内容。\n"
            "第三行继续提供一些内容。\n"
            "第四行继续提供一些内容。\n"
        )
        state = middleware.build_state(document_body=body, document_name="demo")
        view = build_window_view(
            document_body=body,
            cursor=0,
            history_line_count=middleware.runing_config.history_line_count,
            active_line_count=middleware.runing_config.active_line_count,
            preview_line_count=middleware.runing_config.preview_line_count,
            line_wrap_width=middleware.runing_config.line_wrap_width,
        )
        self.assertTrue(view.active_lines)
        self.assertEqual(view.active_lines[0].line_number, 0)
        self.assertTrue(all(len(line.text) <= 30 for line in view.active_lines))
        rendered = middleware.build_window(state)
        self.assertIn("0 |", rendered)
        self.assertIn("<Window>", rendered)

    def test_split_tool_uses_line_end(self) -> None:
        middleware = ChunkingCapabilityMiddleware()
        body = "第一段第一行内容。\n第一段第二行内容。\n第二段第一行内容。\n"
        state = middleware.build_state(document_body=body, document_name="demo")
        next_state, payload = middleware.split_tool.run(
            batch_input=SplitChunkBatchInput(
                items=[
                    SplitChunkAction(
                        summary="第一段",
                        keywords=["第一段", "内容"],
                        line_end=1,
                    )
                ]
            ),
            state=state,
            history_line_count=middleware.runing_config.history_line_count,
            active_line_count=middleware.runing_config.active_line_count,
            preview_line_count=middleware.runing_config.preview_line_count,
        )
        self.assertEqual(payload["status"], "success")
        self.assertEqual(len(next_state["chunks"]), 1)
        self.assertIn("第一段第二行内容", next_state["chunks"][0]["text"])
        self.assertGreater(next_state["cursor"], 0)

    def test_window_byte_limits_trim_passed_and_forward_text(self) -> None:
        middleware = ChunkingCapabilityMiddleware()
        body = "alpha\nbeta\ngamma\ndelta\n"
        cursor = body.index("gamma")
        state = middleware.build_state(document_body=body, document_name="demo")
        state["cursor"] = cursor
        state["window_back_bytes"] = 5
        state["window_forward_bytes"] = 6
        rendered = middleware.build_window(state)
        self.assertIn("beta", rendered)
        self.assertNotIn("alpha", rendered)
        self.assertIn("0 | gamma", rendered)
        self.assertNotIn("delta", rendered)


if __name__ == "__main__":
    unittest.main()
