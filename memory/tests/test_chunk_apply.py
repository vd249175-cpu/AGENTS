from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from secrets import token_hex
from unittest.mock import patch

from agents.chunk_apply_agent import build_chunk_apply_agent
from server.neo4j import ChunkStore
from tools.chunk_apply import ChunkApplyInput, ChunkApplyOverrides, ChunkApplyTool, ChunkApplyToolConfig, _coerce_bool


class ChunkApplyToolTests(unittest.TestCase):
    def build_tool(self, root: Path) -> ChunkApplyTool:
        config = ChunkApplyToolConfig(
            resume=True,
            cache_path=root / "store" / "cache" / "chunk_cache.sqlite3",
            staging_path=root / "store" / "staging" / "chunk_staging.sqlite3",
            checkpoint_path=root / "store" / "checkpoint" / "chunk_checkpoint.sqlite3",
            public={
                "neo4j": {
                    "uri": "neo4j://localhost:7687",
                    "username": "neo4j",
                    "password": "1575338771",
                    "database": None,
                }
            },
            persist_keyword_embeddings=False,
        )
        return ChunkApplyTool(config=config)

    def test_chunk_apply_processes_text_file_and_persists_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / f"knowledge_{token_hex(4)}.txt"
            source.write_text(
                "第一段第一行。\n第一段第二行。\n第一段第三行。\n\n第二段第一行。\n第二段第二行。\n",
                encoding="utf-8",
            )
            tool = self.build_tool(root)
            result = tool.invoke(
                {
                    "path": str(source),
                    "resume": False,
                    "chunking_requirement": "尽量粗切，优先形成较大的 chunk。",
                }
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["success_count"], 1)
            self.assertEqual(result["document_count"], 1)
            document_result = result["results"][0]
            self.assertTrue(document_result["chunks"])
            self.assertEqual(document_result["document_name"], source.stem)
            store = ChunkStore(config_path=tool.config.neo4j_config_path)
            with store.driver.session(database=store.database) as session:
                run_record = session.run(
                    """
                    MATCH (chunk:Chunk {document_name: $document_name})
                    RETURN chunk.run_id AS run_id
                    LIMIT 1
                    """,
                    document_name=document_result["document_name"],
                ).single()
            assert run_record is not None
            run_id = str(run_record["run_id"])
            persisted_chunks = store.list_chunks(
                run_id=run_id,
                document_name=document_result["document_name"],
            )
            self.assertEqual(len(persisted_chunks), len(document_result["chunks"]))
            self.assertTrue(all(item["document_name"] == source.stem for item in persisted_chunks))
            with store.driver.session(database=store.database) as session:
                record = session.run(
                    """
                    MATCH (:Chunk {document_name: $document_name})-[edge:DOCUMENT_NEXT]->
                          (:Chunk {document_name: $document_name})
                    RETURN count(edge) AS edge_count, collect(DISTINCT edge.dist) AS dists
                    """,
                    run_id=run_id,
                    document_name=document_result["document_name"],
                ).single()
            assert record is not None
            self.assertEqual(int(record["edge_count"]), max(0, len(persisted_chunks) - 1))
            if len(persisted_chunks) > 1:
                self.assertEqual([float(item) for item in record["dists"]], [0.3])
            store.close()
            tool.close()

    def test_chunk_apply_rejects_duplicate_document_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            duplicate_name = f"knowledge_{token_hex(4)}.txt"
            first = root / duplicate_name
            first.write_text(
                "第一份文档第一行。\n第一份文档第二行。\n第一份文档第三行。\n",
                encoding="utf-8",
            )
            second_dir = root / "nested"
            second_dir.mkdir()
            second = second_dir / duplicate_name
            second.write_text(
                "另一份同名文档第一行。\n另一份同名文档第二行。\n",
                encoding="utf-8",
            )
            tool = self.build_tool(root)
            first_result = tool.invoke(
                {
                    "path": str(first),
                    "resume": False,
                    "chunking_requirement": "尽量粗切，优先形成较大的 chunk。",
                }
            )
            self.assertEqual(first_result["status"], "success")

            duplicate_result = tool.invoke(
                {
                    "path": str(second),
                    "resume": False,
                    "chunking_requirement": "尽量粗切，优先形成较大的 chunk。",
                }
            )
            self.assertEqual(duplicate_result["status"], "error")
            self.assertEqual(duplicate_result["message"], "当前文档已经在memory中")
            self.assertEqual(duplicate_result["failure_count"], 1)
            self.assertEqual(duplicate_result["results"][0]["message"], "当前文档已经在memory中")
            self.assertEqual(duplicate_result["results"][0]["document_name"], Path(duplicate_name).stem)
            tool.close()

    def test_chunk_apply_rejects_directory_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tool = self.build_tool(root)
            with self.assertRaisesRegex(ValueError, "single file path"):
                tool.invoke({"path": str(root), "resume": False})
            tool.close()

    def test_chunk_apply_rejects_same_stem_with_different_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / f"shared_{token_hex(4)}.txt"
            second = root / first.name.replace(".txt", ".md")
            first.write_text("第一份内容。\n第二行。\n", encoding="utf-8")
            second.write_text("# 第二份内容\n\n第二行。\n", encoding="utf-8")
            tool = self.build_tool(root)
            first_result = tool.invoke({"path": str(first), "resume": False})
            second_result = tool.invoke({"path": str(second), "resume": False})
            self.assertEqual(first_result["status"], "success")
            self.assertEqual(second_result["status"], "error")
            self.assertEqual(second_result["message"], "当前文档已经在memory中")
            self.assertEqual(first_result["results"][0]["document_name"], first.stem)
            self.assertEqual(second_result["results"][0]["document_name"], second.stem)
            tool.close()

    def test_chunk_apply_accepts_non_text_file_via_source_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / f"slides_{token_hex(4)}.pdf"
            source.write_bytes(b"%PDF-1.4\n%fake\n")
            tool = self.build_tool(root)
            with patch(
                "agents.chunk_apply_agent.load_single_source_document",
                return_value={
                    "source_path": str(source),
                    "document_name": source.stem,
                    "text": "第一页内容。\n第二页内容。\n",
                    "metadata": {"file_path": str(source)},
                },
            ):
                result = tool.invoke(
                    {
                        "path": str(source),
                        "resume": False,
                        "chunking_requirement": "按较粗粒度切分。",
                    }
                )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["results"][0]["document_name"], source.stem)
            tool.close()

    def test_chunk_apply_resume_is_not_blocked_by_existing_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / f"resume_{token_hex(4)}.txt"
            source.write_text("第一行。\n第二行。\n第三行。\n", encoding="utf-8")
            tool = self.build_tool(root)
            first_result = tool.invoke({"path": str(source), "resume": False})
            self.assertEqual(first_result["status"], "success")

            resumed_result = tool.invoke({"path": str(source), "resume": True})
            self.assertEqual(resumed_result["status"], "success")
            self.assertTrue(resumed_result["results"][0]["resumed"])
            tool.close()

    def test_chunk_apply_resume_rejects_changed_source_with_existing_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / f"resume_changed_{token_hex(4)}.txt"
            source.write_text("第一行。\n第二行。\n第三行。\n", encoding="utf-8")
            tool = self.build_tool(root)
            try:
                first_result = tool.invoke({"path": str(source), "resume": False})
                self.assertEqual(first_result["status"], "success")

                source.write_text("第一行已经改变。\n第二行。\n第三行。\n第四行。\n", encoding="utf-8")
                changed_result = tool.invoke({"path": str(source), "resume": True})
                self.assertEqual(changed_result["status"], "error")
                self.assertEqual(changed_result["message"], "当前文档已经在memory中")
                self.assertEqual(changed_result["failure_count"], 1)
            finally:
                tool.close()

    def test_chunk_apply_bool_strings_are_parsed_explicitly(self) -> None:
        self.assertFalse(_coerce_bool("false"))
        self.assertFalse(_coerce_bool("0"))
        self.assertFalse(_coerce_bool("off"))
        self.assertTrue(_coerce_bool("true"))
        self.assertTrue(_coerce_bool("1"))
        self.assertTrue(_coerce_bool("on"))

    def test_chunk_apply_input_can_override_sharding_runtime(self) -> None:
        captured: dict[str, object] = {}

        class FakeChunkStore:
            def __init__(self, **_):
                return None

            def document_exists(self, **_):
                return False

            def close(self):
                return None

        class FakeCheckpoint:
            def __init__(self, path):
                self.path = path

            def close(self):
                return None

        class FakeCache:
            def __init__(self, path):
                self.path = path

            def load(self, **_):
                return None

            def close(self):
                return None

        class FakeStagingStore:
            def __init__(self, path):
                self.path = path

            def close(self):
                return None

        class FakeAgent:
            def process_document(self, **kwargs):
                captured.update(kwargs)
                return {
                    "ok": True,
                    "resumed": False,
                    "source_path": str(kwargs["source_path"]),
                    "document_name": kwargs["source_path"].stem,
                    "chunks": [],
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "large_doc.txt"
            source.write_text("第一行。\n第二行。\n第三行。\n", encoding="utf-8")
            config = ChunkApplyToolConfig(
                public={
                    "neo4j": {
                        "uri": "neo4j://example.invalid:7687",
                        "username": "neo4j",
                        "password": "secret",
                        "database": None,
                    }
                },
                runtime={
                    "cache_path": root / "cache.sqlite3",
                    "staging_path": root / "staging.sqlite3",
                    "reference_bytes": 4096,
                    "max_workers": 1,
                    "shard_count": 1,
                },
                checkpoint_path=root / "checkpoint.sqlite3",
            )
            with (
                patch("tools.chunk_apply.build_chunk_apply_agent", return_value=FakeAgent()),
                patch("tools.chunk_apply.ChunkStore", FakeChunkStore),
                patch("tools.chunk_apply.SQLiteChunkCheckpoint", FakeCheckpoint),
                patch("tools.chunk_apply.SQLiteChunkCache", FakeCache),
                patch("tools.chunk_apply.SQLiteChunkStagingStore", FakeStagingStore),
            ):
                tool = ChunkApplyTool(config=config)
                result = tool.invoke(
                    {
                        "path": str(source),
                        "resume": "false",
                        "shard_count": "3",
                        "max_workers": "2",
                        "reference_bytes": "128",
                    }
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["shard_count"], 3)
        self.assertEqual(result["max_workers"], 2)
        self.assertEqual(result["reference_bytes"], 128)
        self.assertEqual(captured["shard_count"], 3)
        self.assertEqual(captured["max_workers"], 2)
        self.assertEqual(captured["reference_bytes"], 128)
        self.assertFalse(captured["resume"])

    def test_chunk_apply_agent_parallel_shards_merge_by_source_order(self) -> None:
        class FakeCache:
            def load(self, **_):
                return None

            def save(self, **_):
                return None

        class FakeStore:
            def save(self, **_):
                return None

        class FakeCheckpoint:
            def save(self, **_):
                return None

        class FakeChunkStore:
            def __init__(self):
                self.chunks: list[dict[str, object]] = []

            def clear_document(self, **_):
                self.chunks.clear()

            def upsert_chunk(self, **kwargs):
                self.chunks.append(kwargs)

            def rebuild_document_edges(self, **_):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sharded_source.txt"
            source.write_text("\n".join(f"第{i}行内容用于测试分片切分。" for i in range(30)), encoding="utf-8")
            chunk_store = FakeChunkStore()
            result = build_chunk_apply_agent().process_document(
                source_path=source,
                run_id="shard-run",
                thread_id="shard-thread",
                cache=FakeCache(),
                staging_store=FakeStore(),
                checkpoint=FakeCheckpoint(),
                chunk_store=chunk_store,
                resume=False,
                chunking_requirement="粗切",
                shard_count=4,
                reference_bytes=80,
                max_workers=2,
            )

        self.assertTrue(result["ok"])
        self.assertGreater(len(result["chunks"]), 1)
        starts = [int(chunk["char_start"]) for chunk in result["chunks"]]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(starts, [int(chunk["char_start"]) for chunk in chunk_store.chunks])
        self.assertEqual(int(result["chunks"][0]["char_start"]), 0)

    def test_chunk_apply_allows_same_document_name_in_different_base_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / f"kb_scope_{token_hex(4)}.txt"
            source.write_text("第一行。\n第二行。\n第三行。\n", encoding="utf-8")
            first_tool = ChunkApplyTool(
                config=ChunkApplyToolConfig(
                    identity={"base_run_id": "kb-a", "base_thread_id": "kb-a-thread"},
                    resume=True,
                    cache_path=root / "kb_a_cache.sqlite3",
                    staging_path=root / "kb_a_staging.sqlite3",
                    checkpoint_path=root / "kb_a_checkpoint.sqlite3",
                    public={
                        "neo4j": {
                            "uri": "neo4j://localhost:7687",
                            "username": "neo4j",
                            "password": "1575338771",
                            "database": None,
                        }
                    },
                    persist_keyword_embeddings=False,
                )
            )
            second_tool = ChunkApplyTool(
                config=ChunkApplyToolConfig(
                    identity={"base_run_id": "kb-b", "base_thread_id": "kb-b-thread"},
                    resume=True,
                    cache_path=root / "kb_b_cache.sqlite3",
                    staging_path=root / "kb_b_staging.sqlite3",
                    checkpoint_path=root / "kb_b_checkpoint.sqlite3",
                    public={
                        "neo4j": {
                            "uri": "neo4j://localhost:7687",
                            "username": "neo4j",
                            "password": "1575338771",
                            "database": None,
                        }
                    },
                    persist_keyword_embeddings=False,
                )
            )
            try:
                first_result = first_tool.invoke({"path": str(source), "resume": False})
                second_result = second_tool.invoke({"path": str(source), "resume": False})
                self.assertEqual(first_result["status"], "success")
                self.assertEqual(second_result["status"], "success")
            finally:
                first_tool.close()
                second_tool.close()

    def test_chunk_apply_overrides_merge_runtime_and_chunking_layers(self) -> None:
        captured: dict[str, object] = {}

        class FakeChunkStore:
            def __init__(self, *, uri, username, password, database, document_edge_distance, persist_keyword_embeddings, embedding_config_override=None):
                captured["chunk_store"] = {
                    "uri": uri,
                    "username": username,
                    "password": password,
                    "database": database,
                    "document_edge_distance": document_edge_distance,
                    "persist_keyword_embeddings": persist_keyword_embeddings,
                    "embedding_config_override": embedding_config_override,
                }

            def close(self):
                return None

        class FakeCheckpoint:
            def __init__(self, path):
                captured["checkpoint_path"] = path

        class FakeCache:
            def __init__(self, path):
                captured["cache_path"] = path

        class FakeStagingStore:
            def __init__(self, path):
                captured["staging_path"] = path

        def fake_build_chunk_apply_agent(*, middleware):
            captured["middleware"] = middleware
            return object()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_config = ChunkApplyToolConfig(
                identity={"base_run_id": "base-run", "base_thread_id": "base-thread"},
                public={
                    "checkpoint_path": root / "base-checkpoint.sqlite3",
                    "neo4j": {
                        "uri": "neo4j://example.invalid:7687",
                        "username": "neo4j",
                        "password": "secret",
                        "database": None,
                    },
                },
                runtime={
                    "resume": True,
                    "cache_path": root / "base-cache.sqlite3",
                    "staging_path": root / "base-staging.sqlite3",
                    "max_retries": 3,
                },
                document_edge_distance=0.3,
                persist_keyword_embeddings=True,
            )

            with (
                patch("tools.chunk_apply.build_chunk_apply_agent", side_effect=fake_build_chunk_apply_agent),
                patch("tools.chunk_apply.ChunkStore", FakeChunkStore),
                patch("tools.chunk_apply.SQLiteChunkCheckpoint", FakeCheckpoint),
                patch("tools.chunk_apply.SQLiteChunkCache", FakeCache),
                patch("tools.chunk_apply.SQLiteChunkStagingStore", FakeStagingStore),
            ):
                tool = ChunkApplyTool(
                    config=base_config,
                    overrides=ChunkApplyOverrides(
                        identity={"base_run_id": "override-run"},
                        public={
                            "checkpoint_path": root / "override-checkpoint.sqlite3",
                            "embedding_model": "text-embedding-3-small",
                            "embedding_dimensions": 256,
                        },
                        runtime={
                            "cache_path": root / "override-cache.sqlite3",
                            "staging_path": root / "override-staging.sqlite3",
                            "max_retries": 5,
                        },
                        chunking={
                            "active_line_count": 12,
                            "line_wrap_width": 18,
                            "window_back_bytes": 200,
                        },
                        document_edge_distance=0.9,
                        persist_keyword_embeddings=False,
                    ),
                )

        self.assertEqual(tool.config.identity.base_run_id, "override-run")
        self.assertEqual(tool.tool.name, "chunk_apply")
        self.assertIs(tool.toolschema.args_schema, ChunkApplyInput)
        self.assertEqual(tool.substate.__name__, "ChunkApplySubState")
        self.assertEqual(tool.config.identity.base_thread_id, "base-thread")
        self.assertEqual(tool.config.runtime.max_retries, 5)
        self.assertEqual(tool.config.resolved_cache_path, captured["cache_path"])
        self.assertEqual(captured["checkpoint_path"], (root / "override-checkpoint.sqlite3").resolve())
        self.assertEqual(captured["staging_path"], (root / "override-staging.sqlite3").resolve())
        self.assertEqual(captured["chunk_store"]["uri"], "neo4j://example.invalid:7687")
        self.assertEqual(captured["chunk_store"]["document_edge_distance"], 0.9)
        self.assertFalse(captured["chunk_store"]["persist_keyword_embeddings"])
        self.assertEqual(captured["chunk_store"]["embedding_config_override"]["model"], "text-embedding-3-small")
        self.assertEqual(captured["chunk_store"]["embedding_config_override"]["dimensions"], 256)
        self.assertEqual(captured["middleware"].runing_config.active_line_count, 12)
        self.assertEqual(captured["middleware"].runing_config.line_wrap_width, 18)
        self.assertEqual(captured["middleware"].runing_config.window_back_bytes, 200)
        self.assertEqual(captured["middleware"].runing_config.max_retries, 5)


if __name__ == "__main__":
    unittest.main()
