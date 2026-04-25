"""Manual live stream check for document-side tools.

Run with:
    uv run python tests/test_live_document_stream.py
"""

from __future__ import annotations

import json
from pathlib import Path
from secrets import token_hex
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.create_chunk_document import CreateChunkDocumentInput, build_create_chunk_document_tool
from tools.list_chunk_documents import ListChunkDocumentsInput, build_list_chunk_documents_tool
from tools.manage_chunks import ChunkManageAction, ManageChunksInput, build_manage_chunks_tool
from tools.query_chunk_positions import QueryChunkPositionsInput, build_query_chunk_positions_tool


def _print_event(event: dict[str, object]) -> None:
    print(json.dumps(event, ensure_ascii=False))


def _call_tool(tool, *, name: str, payload: dict[str, object], state: dict[str, object] | None = None):
    _print_event({"type": "tool", "tool": name, "event": "start"})
    kwargs = dict(payload)
    if state is not None:
        kwargs["runtime"] = SimpleNamespace(state=state, tool_call_id=f"{name}-manual")
    result = tool.func(**kwargs)
    _print_event({"type": "tool", "tool": name, "event": "success"})
    return result


def main() -> None:
    document_name = f"live_document_{token_hex(4)}"
    create_tool = build_create_chunk_document_tool()
    manage_tool = build_manage_chunks_tool()
    list_tool = build_list_chunk_documents_tool()
    query_tool = build_query_chunk_positions_tool()

    create_result = _call_tool(
        create_tool,
        name="create_chunk_document",
        payload=CreateChunkDocumentInput(
            document_name=document_name,
            summary="live 文档",
            body="第一段 live 内容。",
            keywords=["live", "document"],
        ).model_dump(),
    )
    print(json.dumps(create_result, ensure_ascii=False, indent=2))

    manage_result = _call_tool(
        manage_tool,
        name="manage_chunks",
        payload=ManageChunksInput(
            document_name=document_name,
            actions=[
                ChunkManageAction(
                    op="insert",
                    insert_after=0,
                    summary="第二段 live 文档",
                    body="第二段 live 内容。",
                    keywords=["live", "insert"],
                )
            ],
        ).model_dump(),
    )
    print(json.dumps(manage_result, ensure_ascii=False, indent=2))

    query_result = _call_tool(
        query_tool,
        name="query_chunk_positions",
        payload=QueryChunkPositionsInput(document_name=document_name, positions=[0, 1]).model_dump(),
    )
    print(json.dumps(query_result, ensure_ascii=False, indent=2))

    list_result = _call_tool(
        list_tool,
        name="list_chunk_documents",
        payload=ListChunkDocumentsInput(limit=5).model_dump(),
    )
    print(json.dumps(list_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
