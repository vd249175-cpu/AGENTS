"""Manual live stream check for the migrated chunk workflow.

Run with:
    uv run python tests/test_live_chunk_stream.py
"""

from __future__ import annotations

import json
from pathlib import Path
from secrets import token_hex
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.chunk_apply import ChunkApplyInput, ChunkApplyTool


def _print_event(event: dict[str, object]) -> None:
    print(json.dumps(event, ensure_ascii=False))


def main() -> None:
    sample_path = PROJECT_ROOT / "workspace" / "knowledge" / f"live_sample_{token_hex(4)}.txt"
    sample_path.write_text(
        "如何做结构化迁移。\n"
        "第一步是先确认目录边界。\n"
        "第二步是把配置收回工具和中间件。\n"
        "第三步是验证流式输出。\n"
        "\n"
        "最后再补回文档和项目结构说明。\n",
        encoding="utf-8",
    )

    tool = ChunkApplyTool()
    result = tool.run(
        tool_input=ChunkApplyInput(
            path=str(sample_path),
            resume=False,
            chunking_requirement="按较粗颗粒度切分。",
        ),
        stream_writer=_print_event,
        progress_callback=_print_event,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    tool.close()


if __name__ == "__main__":
    main()
