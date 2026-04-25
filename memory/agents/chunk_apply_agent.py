"""Chunk apply agent."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from middleware import ChunkingCapabilityMiddleware
from server.neo4j import ChunkStore
from server.source_ingest import load_single_source_document
from server.sqlite import SQLiteChunkCache, SQLiteChunkCheckpoint, SQLiteChunkStagingStore
from tools.split_chunk import SplitChunkStateTydict, render_chunked_document
from agents.chunking_agent import ChunkingAgent, build_chunking_agent


class ChunkApplyAgentSchema:
    name = "chunk_apply_agent"
    systemPrompt = (
        "你是一个负责读取源文件、分片并调用 chunking agent 处理的内部 workflow agent。"
        "只负责调度，不负责改写业务语义。"
    )
    middlewares = {
        ChunkingCapabilityMiddleware.name: ChunkingCapabilityMiddleware,
    }


@dataclass
class ChunkApplyAgent:
    chunking_agent: ChunkingAgent
    name: str = ChunkApplyAgentSchema.name
    agentschema: type = ChunkApplyAgentSchema

    def __post_init__(self) -> None:
        self.agent = self

    def process_document(
        self,
        *,
        source_path: Path,
        run_id: str,
        thread_id: str,
        cache: SQLiteChunkCache,
        staging_store: SQLiteChunkStagingStore,
        checkpoint: SQLiteChunkCheckpoint,
        chunk_store: ChunkStore,
        resume: bool,
        chunking_requirement: str | None,
        shard_count: int = 1,
        reference_bytes: int = 4000,
        max_workers: int | None = 1,
        stream_writer: Callable[[dict[str, Any]], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        document_name = source_path.stem.strip() or source_path.stem or source_path.name or "doc"
        source_fingerprint = build_source_fingerprint(source_path)
        cached_record = cache.load(document_name=document_name, run_id=run_id, thread_id=thread_id) if resume else None
        if cached_record is not None:
            cached_fingerprint = (cached_record.get("state") or {}).get("_source_fingerprint")
            if cached_fingerprint != source_fingerprint:
                cached_record = None
        if cached_record is not None and cached_record["completed"]:
            if chunk_store.document_exists(run_id=run_id, document_name=document_name):
                return {
                    "ok": True,
                    "resumed": True,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "source_path": str(source_path),
                    "document_name": document_name,
                    "chunks": chunk_store.list_chunks(run_id=run_id, document_name=document_name),
                }
            final_state = dict(cached_record["state"])
            chunk_store.clear_document(run_id=run_id, document_name=document_name)
            for chunk_index, chunk in enumerate(final_state.get("chunks", [])):
                chunk_store.upsert_chunk(
                    run_id=run_id,
                    document_name=document_name,
                    chunk_id=str(chunk["id"]),
                    chunk_index=chunk_index,
                    summary=str(chunk["summary"]),
                    keywords=list(chunk.get("keywords", [])),
                    text=str(chunk["text"]),
                    char_start=int(chunk["char_start"]),
                    char_end=int(chunk["char_end"]),
                )
            chunk_store.rebuild_document_edges(run_id=run_id, document_name=document_name)
            return {
                "ok": True,
                "resumed": True,
                "run_id": run_id,
                "thread_id": thread_id,
                "source_path": str(source_path),
                "document_name": document_name,
                "text": render_chunked_document(str(final_state.get("document_body", "")), list(final_state.get("chunks", []))),
                "chunks": list(final_state.get("chunks", [])),
            }
        if resume and cached_record is None and chunk_store.document_exists(run_id=run_id, document_name=document_name):
            return {
                "ok": False,
                "resumed": False,
                "run_id": run_id,
                "thread_id": thread_id,
                "source_path": str(source_path),
                "document_name": document_name,
                "message": "当前文档已经在memory中",
                "chunks": [],
            }

        middleware = self.chunking_agent.middleware
        if cached_record is not None:
            initial_state = dict(cached_record["state"])
            resumed = True
        else:
            source_document = load_single_source_document(source_path)
            text = str(source_document.get("text") or "")
            initial_state = middleware.build_state(document_body=text, document_name=document_name)
            initial_state["_source_fingerprint"] = source_fingerprint
            resumed = False

        cache.save(
            source_path=str(source_path),
            run_id=run_id,
            document_name=document_name,
            thread_id=thread_id,
            state=initial_state,
            completed=False,
        )
        staging_store.save(
            source_path=str(source_path),
            run_id=run_id,
            document_name=document_name,
            thread_id=thread_id,
            state=initial_state,
            status="draft",
        )
        checkpoint.save(
            run_id=run_id,
            thread_id=thread_id,
            payload={"document_name": document_name, "cursor": int(initial_state.get("cursor", 0))},
        )

        def wrapped_writer(event: dict[str, Any]) -> None:
            if stream_writer is not None:
                stream_writer({"document_name": document_name, **event})

        final_state = self.chunking_agent.run(
            initial_state=initial_state,
            requirement=chunking_requirement,
            stream_writer=wrapped_writer,
        ) if int(shard_count) <= 1 else self._run_shards(
            initial_state=initial_state,
            requirement=chunking_requirement,
            shard_count=shard_count,
            reference_bytes=reference_bytes,
            max_workers=max_workers,
            stream_writer=wrapped_writer,
        )
        cache.save(
            source_path=str(source_path),
            run_id=run_id,
            document_name=document_name,
            thread_id=thread_id,
            state=final_state,
            completed=True,
        )
        staging_store.save(
            source_path=str(source_path),
            run_id=run_id,
            document_name=document_name,
            thread_id=thread_id,
            state=final_state,
            status="ready",
        )
        checkpoint.save(
            run_id=run_id,
            thread_id=thread_id,
            payload={
                "document_name": document_name,
                "cursor": int(final_state.get("cursor", 0)),
                "completed": True,
            },
        )
        chunk_store.clear_document(run_id=run_id, document_name=document_name)
        for chunk_index, chunk in enumerate(final_state.get("chunks", [])):
            chunk_store.upsert_chunk(
                run_id=run_id,
                document_name=document_name,
                chunk_id=str(chunk["id"]),
                chunk_index=chunk_index,
                summary=str(chunk["summary"]),
                keywords=list(chunk.get("keywords", [])),
                text=str(chunk["text"]),
                char_start=int(chunk["char_start"]),
                char_end=int(chunk["char_end"]),
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "source_path": str(source_path),
                        "document_name": document_name,
                        "chunk_count": chunk_index + 1,
                        "cursor": int(final_state.get("cursor", 0)),
                    }
                )
        chunk_store.rebuild_document_edges(run_id=run_id, document_name=document_name)
        return {
            "ok": True,
            "resumed": resumed,
            "run_id": run_id,
            "thread_id": thread_id,
            "source_path": str(source_path),
            "document_name": document_name,
            "text": render_chunked_document(str(final_state.get("document_body", "")), list(final_state.get("chunks", []))),
            "chunks": list(final_state.get("chunks", [])),
        }

    def _run_shards(
        self,
        *,
        initial_state: SplitChunkStateTydict,
        requirement: str | None,
        shard_count: int,
        reference_bytes: int,
        max_workers: int | None,
        stream_writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> SplitChunkStateTydict:
        document_body = str(initial_state.get("document_body", ""))
        shards = _split_document_shards(
            document_body,
            shard_count=max(1, int(shard_count)),
            reference_bytes=max(1, int(reference_bytes)),
        )
        if len(shards) <= 1:
            return self.chunking_agent.run(
                initial_state=initial_state,
                requirement=requirement,
                stream_writer=stream_writer,
            )
        worker_count = max(1, min(len(shards), int(max_workers or 1)))

        def run_one(index_and_range: tuple[int, tuple[int, int]]) -> tuple[int, list[dict[str, Any]]]:
            shard_index, (start, end) = index_and_range
            shard_state = dict(initial_state)
            shard_state["document_body"] = document_body[start:end]
            shard_state["cursor"] = 0
            shard_state["chunks"] = []
            shard_state["messages"] = []
            shard_state["process_trace"] = []

            def shard_writer(event: dict[str, Any]) -> None:
                if stream_writer is not None:
                    stream_writer({"shard_index": shard_index, "shard_start": start, **event})

            final_shard_state = self.chunking_agent.run(
                initial_state=shard_state,
                requirement=requirement,
                stream_writer=shard_writer,
            )
            adjusted_chunks = []
            for chunk in final_shard_state.get("chunks", []):
                adjusted = dict(chunk)
                adjusted["id"] = f"s{shard_index}-{adjusted['id']}"
                adjusted["char_start"] = int(adjusted["char_start"]) + start
                adjusted["char_end"] = int(adjusted["char_end"]) + start
                adjusted_chunks.append(adjusted)
            return shard_index, adjusted_chunks

        if stream_writer is not None:
            stream_writer(
                {
                    "type": "workflow",
                    "stage": "shard_start",
                    "shard_count": len(shards),
                    "max_workers": worker_count,
                }
            )
        chunks_by_shard: dict[int, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for shard_index, chunks in executor.map(run_one, enumerate(shards)):
                chunks_by_shard[shard_index] = chunks
        final_state = dict(initial_state)
        final_chunks: list[dict[str, Any]] = []
        for shard_index in range(len(shards)):
            final_chunks.extend(chunks_by_shard.get(shard_index, []))
        final_chunks.sort(key=lambda item: int(item["char_start"]))
        final_state["chunks"] = final_chunks
        final_state["cursor"] = len(document_body)
        final_state["retry_count"] = 0
        if stream_writer is not None:
            stream_writer(
                {
                    "type": "workflow",
                    "stage": "shard_done",
                    "shard_count": len(shards),
                    "chunk_count": len(final_chunks),
                }
            )
        return final_state


def build_source_fingerprint(source_path: Path) -> dict[str, Any]:
    stat = source_path.stat()
    digest = sha256()
    with source_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _split_document_shards(document_body: str, *, shard_count: int, reference_bytes: int) -> list[tuple[int, int]]:
    if shard_count <= 1 or not document_body:
        return [(0, len(document_body))]
    encoded_size = len(document_body.encode("utf-8"))
    if encoded_size <= reference_bytes:
        return [(0, len(document_body))]
    target_count = min(shard_count, max(1, (encoded_size + reference_bytes - 1) // reference_bytes))
    approximate_chars = max(1, len(document_body) // target_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    for shard_index in range(target_count - 1):
        target = min(len(document_body), start + approximate_chars)
        newline = document_body.find("\n", target)
        if newline == -1:
            newline = document_body.rfind("\n", start, target)
        end = (newline + 1) if newline != -1 and newline + 1 > start else target
        if end <= start:
            end = target
        ranges.append((start, end))
        start = end
    if start < len(document_body):
        ranges.append((start, len(document_body)))
    return [(start, end) for start, end in ranges if end > start]


def build_chunk_apply_agent(
    *,
    chunking_agent: ChunkingAgent | None = None,
    middleware: ChunkingCapabilityMiddleware | None = None,
) -> ChunkApplyAgent:
    active_agent = chunking_agent or build_chunking_agent(middleware=middleware)
    return ChunkApplyAgent(chunking_agent=active_agent)
