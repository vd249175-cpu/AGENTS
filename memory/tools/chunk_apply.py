"""Chunk apply workflow tool."""

import json
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable

from agents import build_chunk_apply_agent
from agents.chunk_apply_agent import build_source_fingerprint
from langchain.agents.middleware import AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from middleware.chunking import ChunkingCapabilityMiddleware, MiddlewareRuningConfig
from server.component_config import config_from_external, emit
from server.config_overrides import merge_model
from server.neo4j import ChunkStore, Neo4jConnectionConfig, resolve_neo4j_connection
from server.sqlite import SQLiteChunkCache, SQLiteChunkCheckpoint, SQLiteChunkStagingStore
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ._output import limit_items, strip_internal_run_context


TOOL_CONFIG_PATH = Path(__file__).with_name("chunk_apply.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_RETURN_CHUNKS = 20


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_project_path(value: str | Path | None, *, default: str) -> Path:
    raw_value = value if value is not None else default
    return (PROJECT_ROOT / str(raw_value)).resolve()


class ChunkApplyIdentityConfig(BaseModel):
    base_run_id: str = Field(
        default="chunk-run",
        description="知识库级根 run_id 前缀。chunk_apply 会基于它和 document_name 派生稳定执行身份。",
    )
    base_thread_id: str = Field(
        default="chunk-thread",
        description="workflow 会话级根 thread_id 前缀。主要用于内部 checkpoint 和 shard 状态派生。",
    )
    derive_document_run_id: bool = Field(
        default=True,
        description="是否基于 document_name 派生文档级 run_id；关闭后所有文档共享 base_run_id。",
    )


class ChunkApplyPublicConfig(BaseModel):
    checkpoint_path: Path | None = Field(
        default=None,
        description="共享 checkpoint SQLite 路径，用于失败恢复和 resume。",
    )
    neo4j: Neo4jConnectionConfig | None = Field(
        default=None,
        description="显式的 Neo4j 连接参数。对外推荐直接传这组字段，不再依赖 database_config.json 中转。",
    )
    neo4j_config_path: Path | None = Field(
        default=None,
        description="兼容旧链路的 Neo4j 配置文件路径。仅保留给内部或历史调用，不作为公开推荐方式。",
    )
    embedding_provider: str | None = Field(
        default=None,
        description="关键词 embedding 提供商 override，例如 openai、ollama。",
    )
    embedding_model: str | None = Field(
        default=None,
        description="关键词 embedding 模型名 override，例如 text-embedding-3-small。",
    )
    embedding_base_url: str | None = Field(
        default=None,
        description="embedding 服务地址 override，接代理或本地服务时使用。",
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="embedding 鉴权密钥 override；本地无鉴权服务可留空。",
    )
    embedding_dimensions: int | None = Field(
        default=None,
        description="embedding 维度 override，会参与向量配置一致性判断。",
    )


class ChunkApplyRuntimeConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    resume: bool = Field(default=True, description="是否默认允许从 cache/checkpoint 恢复同一文档导入。")
    cache_path: Path | None = Field(default=None, description="chunk cache SQLite 文件路径。")
    staging_path: Path | None = Field(default=None, description="chunk staging SQLite 文件路径。")
    recursion_limit: int | None = Field(default=None, description="预留的工作流递归上限配置。")
    shard_count: int = Field(default=1, ge=1, description="长文档切分预留的分片数量。")
    reference_bytes: int = Field(default=4000, ge=1, description="长文档分片时使用的参考字节窗口大小。")
    max_retries: int = Field(default=3, ge=1, description="单次切分动作的默认最大重试次数。")
    max_workers: int | None = Field(default=1, description="预留的并发 worker 上限；默认建议保持 1。")
    cache: SQLiteChunkCache | None = Field(default=None, description="可直接注入的 chunk cache 实例。")
    staging_store: SQLiteChunkStagingStore | None = Field(default=None, description="可直接注入的 staging store 实例。")
    stream_callback: Callable[[dict[str, Any]], None] | None = Field(default=None, description="workflow 流式事件回调。")
    progress_callback: Callable[[dict[str, Any]], None] | None = Field(default=None, description="workflow 进度事件回调。")


class ChunkApplyToolConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    identity: ChunkApplyIdentityConfig = Field(default_factory=ChunkApplyIdentityConfig, description="chunk_apply 身份派生配置。")
    public: ChunkApplyPublicConfig = Field(default_factory=ChunkApplyPublicConfig, description="chunk_apply 公共环境配置。")
    runtime: ChunkApplyRuntimeConfig = Field(default_factory=ChunkApplyRuntimeConfig, description="chunk_apply 运行时配置。")
    chunking: MiddlewareRuningConfig = Field(default_factory=MiddlewareRuningConfig, description="chunk_apply 内部 chunking 窗口配置。")
    resume: bool | None = Field(default=None, description="兼容旧扁平配置的 resume 字段。")
    cache_path: Path | None = Field(default=None, description="兼容旧扁平配置的 cache_path 字段。")
    staging_path: Path | None = Field(default=None, description="兼容旧扁平配置的 staging_path 字段。")
    checkpoint_path: Path | None = Field(default=None, description="兼容旧扁平配置的 checkpoint_path 字段。")
    neo4j_config_path: Path | None = Field(default=None, description="兼容旧扁平配置的 neo4j_config_path 字段。")
    document_edge_distance: float = Field(default=0.3, ge=0.0, description="文档内部 DOCUMENT_NEXT 顺序边写入的默认距离值。")
    persist_keyword_embeddings: bool = Field(default=True, description="是否把关键词向量持久化到 Neo4j。")

    @model_validator(mode="after")
    def _synchronize_layers(self) -> "ChunkApplyToolConfig":
        if self.resume is not None:
            self.runtime.resume = bool(self.resume)
        else:
            self.resume = self.runtime.resume
        if self.cache_path is not None:
            self.runtime.cache_path = Path(self.cache_path)
        elif self.runtime.cache_path is not None:
            self.cache_path = Path(self.runtime.cache_path)
        if self.staging_path is not None:
            self.runtime.staging_path = Path(self.staging_path)
        elif self.runtime.staging_path is not None:
            self.staging_path = Path(self.runtime.staging_path)
        if self.checkpoint_path is not None:
            self.public.checkpoint_path = Path(self.checkpoint_path)
        elif self.public.checkpoint_path is not None:
            self.checkpoint_path = Path(self.public.checkpoint_path)
        if self.neo4j_config_path is not None:
            self.public.neo4j_config_path = Path(self.neo4j_config_path)
        elif self.public.neo4j_config_path is not None:
            self.neo4j_config_path = Path(self.public.neo4j_config_path)
        return self

    @property
    def resolved_resume(self) -> bool:
        return bool(self.resume if self.resume is not None else True)

    @property
    def resolved_cache_path(self) -> Path:
        return Path(self.cache_path or (PROJECT_ROOT / "store/cache/chunk_cache.sqlite3")).resolve()

    @property
    def resolved_staging_path(self) -> Path:
        return Path(self.staging_path or (PROJECT_ROOT / "store/staging/chunk_staging.sqlite3")).resolve()

    @property
    def resolved_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_path or (PROJECT_ROOT / "store/checkpoint/chunk_checkpoint.sqlite3")).resolve()

    @property
    def resolved_neo4j_config_path(self) -> Path:
        return Path(self.neo4j_config_path or (PROJECT_ROOT / "workspace/config/database_config.json")).resolve()

    @property
    def resolved_neo4j(self) -> Neo4jConnectionConfig:
        return resolve_neo4j_connection(connection=self.public.neo4j, path=self.resolved_neo4j_config_path)

    @classmethod
    def load(cls, path: Path = TOOL_CONFIG_PATH) -> "ChunkApplyToolConfig":
        payload = _load_json(path)
        identity_payload = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
        public_payload = payload.get("public") if isinstance(payload.get("public"), dict) else {}
        runtime_payload = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
        chunking_payload = payload.get("chunking") if isinstance(payload.get("chunking"), dict) else {}
        checkpoint_path = _resolve_project_path(
            public_payload.get("checkpoint_path") or payload.get("checkpoint_path"),
            default="store/checkpoint/chunk_checkpoint.sqlite3",
        )
        neo4j_config_path = _resolve_project_path(
            public_payload.get("neo4j_config_path") or payload.get("neo4j_config_path"),
            default="workspace/config/database_config.json",
        )
        cache_path = _resolve_project_path(
            runtime_payload.get("cache_path") or payload.get("cache_path"),
            default="store/cache/chunk_cache.sqlite3",
        )
        staging_path = _resolve_project_path(
            runtime_payload.get("staging_path") or payload.get("staging_path"),
            default="store/staging/chunk_staging.sqlite3",
        )
        runtime_resume = bool(runtime_payload.get("resume", payload.get("resume", True)))
        active_chunking_payload = {
            **chunking_payload,
            "max_retries": chunking_payload.get("max_retries", runtime_payload.get("max_retries", payload.get("max_retries", 3))),
        }
        return cls(
            identity=ChunkApplyIdentityConfig(
                base_run_id=str(identity_payload.get("base_run_id", payload.get("base_run_id", "chunk-run"))),
                base_thread_id=str(identity_payload.get("base_thread_id", payload.get("base_thread_id", "chunk-thread"))),
                derive_document_run_id=bool(identity_payload.get("derive_document_run_id", payload.get("derive_document_run_id", True))),
            ),
            public=ChunkApplyPublicConfig(
                checkpoint_path=checkpoint_path,
                neo4j=Neo4jConnectionConfig.model_validate(public_payload["neo4j"]) if isinstance(public_payload.get("neo4j"), dict) else None,
                neo4j_config_path=neo4j_config_path,
                embedding_provider=public_payload.get("embedding_provider"),
                embedding_model=public_payload.get("embedding_model"),
                embedding_base_url=public_payload.get("embedding_base_url"),
                embedding_api_key=public_payload.get("embedding_api_key"),
                embedding_dimensions=public_payload.get("embedding_dimensions"),
            ),
            runtime=ChunkApplyRuntimeConfig(
                resume=runtime_resume,
                cache_path=cache_path,
                staging_path=staging_path,
                recursion_limit=runtime_payload.get("recursion_limit"),
                shard_count=max(1, int(runtime_payload.get("shard_count", payload.get("shard_count", 1)))),
                reference_bytes=max(1, int(runtime_payload.get("reference_bytes", payload.get("reference_bytes", 4000)))),
                max_retries=max(1, int(runtime_payload.get("max_retries", payload.get("max_retries", 3)))),
                max_workers=runtime_payload.get("max_workers", payload.get("max_workers", 1)),
            ),
            chunking=MiddlewareRuningConfig.model_validate(active_chunking_payload),
            resume=runtime_resume,
            cache_path=cache_path,
            staging_path=staging_path,
            checkpoint_path=checkpoint_path,
            neo4j_config_path=neo4j_config_path,
            document_edge_distance=float(payload.get("document_edge_distance", 0.3)),
            persist_keyword_embeddings=bool(payload.get("persist_keyword_embeddings", True)),
        )

    @classmethod
    def load_config_chunk_apply_tool(cls, source=None) -> "ChunkApplyToolConfig":
        """Load the demo-style top-level chunk_apply context from dict/json/defaults."""

        if source is None:
            return cls.load()
        return config_from_external(cls, source)


class ChunkApplyIdentityOverride(BaseModel):
    base_run_id: str | None = Field(default=None, description="覆盖 chunk_apply 的根 run_id 前缀。")
    base_thread_id: str | None = Field(default=None, description="覆盖 chunk_apply 的根 thread_id 前缀。")


class ChunkApplyPublicOverride(BaseModel):
    checkpoint_path: Path | None = Field(default=None, description="覆盖共享 checkpoint SQLite 路径。")
    neo4j: Neo4jConnectionConfig | dict[str, Any] | None = Field(default=None, description="覆盖显式 Neo4j 连接参数。")
    neo4j_config_path: Path | None = Field(default=None, description="覆盖 Neo4j 配置文件路径。")
    embedding_provider: str | None = Field(default=None, description="覆盖 embedding provider。")
    embedding_model: str | None = Field(default=None, description="覆盖 embedding model。")
    embedding_base_url: str | None = Field(default=None, description="覆盖 embedding base_url。")
    embedding_api_key: str | None = Field(default=None, description="覆盖 embedding api_key。")
    embedding_dimensions: int | None = Field(default=None, description="覆盖 embedding dimensions。")


class ChunkApplyRuntimeOverride(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    resume: bool | None = Field(default=None, description="覆盖默认 resume 行为。")
    cache_path: Path | None = Field(default=None, description="覆盖 chunk cache 路径。")
    staging_path: Path | None = Field(default=None, description="覆盖 chunk staging 路径。")
    recursion_limit: int | None = Field(default=None, description="覆盖预留递归上限。")
    shard_count: int | None = Field(default=None, ge=1, description="覆盖预留分片数量。")
    reference_bytes: int | None = Field(default=None, ge=1, description="覆盖预留分片参考窗口大小。")
    max_retries: int | None = Field(default=None, ge=1, description="覆盖 chunk 最大重试次数。")
    max_workers: int | None = Field(default=None, description="覆盖 worker 上限。")
    cache: SQLiteChunkCache | None = Field(default=None, description="覆盖 chunk cache 实例。")
    staging_store: SQLiteChunkStagingStore | None = Field(default=None, description="覆盖 staging store 实例。")
    stream_callback: Callable[[dict[str, Any]], None] | None = Field(default=None, description="覆盖流式事件回调。")
    progress_callback: Callable[[dict[str, Any]], None] | None = Field(default=None, description="覆盖进度回调。")


class ChunkingMiddlewareOverride(BaseModel):
    history_line_count: int | None = Field(default=None, ge=0, description="覆盖窗口历史区域行数。")
    active_line_count: int | None = Field(default=None, ge=1, description="覆盖当前可切分区域行数。")
    preview_line_count: int | None = Field(default=None, ge=0, description="覆盖窗口预览区域行数。")
    line_wrap_width: int | None = Field(default=None, ge=1, description="覆盖窗口自动换行宽度。")
    window_back_bytes: int | None = Field(default=None, ge=1, description="覆盖 Passed 区域字节窗口上限。")
    window_forward_bytes: int | None = Field(default=None, ge=1, description="覆盖 Window + Preview 字节窗口上限。")
    trace_limit: int | None = Field(default=None, ge=1, description="覆盖 trace 保留条数。")
    max_retries: int | None = Field(default=None, ge=1, description="覆盖切分默认最大重试次数。")


class ChunkApplyOverrides(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    identity: ChunkApplyIdentityOverride = Field(default_factory=ChunkApplyIdentityOverride)
    public: ChunkApplyPublicOverride = Field(default_factory=ChunkApplyPublicOverride)
    runtime: ChunkApplyRuntimeOverride = Field(default_factory=ChunkApplyRuntimeOverride)
    chunking: ChunkingMiddlewareOverride = Field(default_factory=ChunkingMiddlewareOverride)
    document_edge_distance: float | None = Field(default=None, ge=0.0, description="覆盖文档内部顺序边距离。")
    persist_keyword_embeddings: bool | None = Field(default=None, description="覆盖关键词向量落库开关。")


def _apply_chunk_apply_overrides(
    config: ChunkApplyToolConfig,
    overrides: ChunkApplyOverrides | dict[str, Any] | None,
) -> ChunkApplyToolConfig:
    if overrides is None:
        return config
    if isinstance(overrides, dict):
        overrides = ChunkApplyOverrides.model_validate(overrides)
    override_payload = overrides.model_dump(exclude_none=True)
    public_payload = override_payload.get("public")
    if isinstance(public_payload, dict):
        if "checkpoint_path" in public_payload:
            override_payload["checkpoint_path"] = public_payload["checkpoint_path"]
        if "neo4j_config_path" in public_payload:
            override_payload["neo4j_config_path"] = public_payload["neo4j_config_path"]
    runtime_payload = override_payload.get("runtime")
    if isinstance(runtime_payload, dict):
        if "resume" in runtime_payload:
            override_payload["resume"] = runtime_payload["resume"]
        if "cache_path" in runtime_payload:
            override_payload["cache_path"] = runtime_payload["cache_path"]
        if "staging_path" in runtime_payload:
            override_payload["staging_path"] = runtime_payload["staging_path"]
    return merge_model(config, override_payload)


class ChunkApplyInput(BaseModel):
    path: str = Field(description="要切块并入库的单个文件路径。")
    resume: bool = Field(default=True, description="是否尝试复用缓存与 checkpoint 恢复上次进度。")
    chunking_requirement: str | None = Field(default=None, description="给切块 agent 的补充要求。")
    shard_count: int | None = Field(default=None, ge=1, description="本次单文档切分使用的分片数量；空值使用工具构造配置。")
    max_workers: int | None = Field(default=None, ge=1, description="本次分片切分允许的最大线程数；空值使用工具构造配置。")
    reference_bytes: int | None = Field(default=None, ge=1, description="本次分片参考字节窗口；空值使用工具构造配置。")


class ChunkApplyToolFeedback(BaseModel):
    successText: str = Field(default="chunk_apply 已完成。")
    failureText: str = Field(default="chunk_apply 执行失败：{error}")


class ChunkApplyToolSchema:
    name = "chunk_apply"
    args_schema = ChunkApplyInput
    description = (
        "将单个文件抽取、切分并写入知识库。"
        "业务入参接受 path、resume、chunking_requirement，以及长文档调参 shard_count、max_workers、reference_bytes。"
    )
    toolfeedback = ChunkApplyToolFeedback


class ChunkApplySubState(AgentState, total=False):
    chunkApplyStats: dict[str, int]


class _ChunkApplyRuntimeTool(BaseTool):
    name: str = "chunk_apply"
    description: str = (
        "将单个文件抽取、切分并写入知识库。"
        "业务入参接受 path、resume、chunking_requirement，以及长文档调参 shard_count、max_workers、reference_bytes。"
    )
    args_schema: type[BaseModel] = ChunkApplyInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _config: ChunkApplyToolConfig = PrivateAttr()
    _middleware: ChunkingCapabilityMiddleware = PrivateAttr()
    _agent: Any = PrivateAttr()
    _cache: SQLiteChunkCache = PrivateAttr()
    _staging_store: SQLiteChunkStagingStore = PrivateAttr()
    _checkpoint: SQLiteChunkCheckpoint = PrivateAttr()
    _chunk_store: ChunkStore = PrivateAttr()
    _owns_cache: bool = PrivateAttr(default=False)
    _owns_staging_store: bool = PrivateAttr(default=False)
    _owns_checkpoint: bool = PrivateAttr(default=True)

    def __init__(
        self,
        *,
        config: ChunkApplyToolConfig | None = None,
        middleware: ChunkingCapabilityMiddleware | None = None,
        overrides: ChunkApplyOverrides | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        base_config = config or ChunkApplyToolConfig.load()
        base_config = _apply_chunk_apply_overrides(base_config, overrides)
        self._config = base_config
        base_chunking_config = merge_model(self.config.chunking, {"max_retries": self.config.runtime.max_retries})
        chunking_override = None
        if isinstance(overrides, dict):
            chunking_override = ChunkApplyOverrides.model_validate(overrides).chunking
        elif overrides is not None:
            chunking_override = overrides.chunking
        if middleware is not None:
            active_chunking_config = merge_model(base_chunking_config, chunking_override)
            self._middleware = ChunkingCapabilityMiddleware(
                runing_config=active_chunking_config,
                split_tool=middleware.split_tool,
            )
        else:
            active_chunking_config = merge_model(base_chunking_config, chunking_override)
            self._middleware = ChunkingCapabilityMiddleware(runing_config=active_chunking_config)
        self._agent = build_chunk_apply_agent(middleware=self.middleware)
        self._owns_cache = self.config.runtime.cache is None
        self._owns_staging_store = self.config.runtime.staging_store is None
        self._cache = self.config.runtime.cache or SQLiteChunkCache(self.config.resolved_cache_path)
        self._staging_store = self.config.runtime.staging_store or SQLiteChunkStagingStore(self.config.resolved_staging_path)
        self._checkpoint = SQLiteChunkCheckpoint(self.config.resolved_checkpoint_path)
        neo4j = self.config.resolved_neo4j
        self._chunk_store = ChunkStore(
            uri=neo4j.uri,
            username=neo4j.username,
            password=neo4j.password,
            database=neo4j.database,
            document_edge_distance=self.config.document_edge_distance,
            persist_keyword_embeddings=self.config.persist_keyword_embeddings,
            embedding_config_override=_embedding_override_from_public_config(self.config.public),
        )

    @property
    def config(self) -> ChunkApplyToolConfig:
        return self._config

    @property
    def middleware(self) -> ChunkingCapabilityMiddleware:
        return self._middleware

    @staticmethod
    def document_name_for_source(source_path: Path) -> str:
        return source_path.stem.strip() or source_path.stem or source_path.name or "doc"

    def resolve_execution_identity(self, source_path: Path, *, resume: bool) -> tuple[str, str]:
        digest = sha1(self.document_name_for_source(source_path).encode("utf-8")).hexdigest()[:12]
        run_id = (
            f"{self.config.identity.base_run_id}-{digest}"
            if self.config.identity.derive_document_run_id
            else self.config.identity.base_run_id
        )
        thread_id = f"{self.config.identity.base_thread_id}-{digest}"
        return run_id, thread_id

    @staticmethod
    def _error_result(
        *,
        source_path: Path,
        document_name: str,
        message: str,
        chunking_requirement: str | None,
        resume: bool,
        shard_count: int | None = None,
        max_workers: int | None = None,
        reference_bytes: int | None = None,
    ) -> dict[str, Any]:
        return {
            "operation": "chunk_apply",
            "status": "error",
            "message": message,
            "path": str(source_path),
            "path_kind": "file",
            "resume": resume,
            "shard_count": shard_count,
            "max_workers": max_workers,
            "reference_bytes": reference_bytes,
            "chunking_requirement": chunking_requirement,
            "document_count": 1,
            "success_count": 0,
            "failure_count": 1,
            "succeeded_documents": [],
            "failed_documents": [document_name],
            "results": [
                {
                    "ok": False,
                    "source_path": str(source_path),
                    "document_name": document_name,
                    "message": message,
                }
            ],
        }

    @staticmethod
    def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
        payload = dict(result)
        payload.pop("text", None)
        compact_chunks = []
        for chunk in payload.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            compact_chunks.append(
                {
                    "id": chunk.get("id"),
                    "document_name": chunk.get("document_name"),
                    "chunk_index": chunk.get("chunk_index"),
                    "summary": chunk.get("summary"),
                    "keywords": chunk.get("keywords") or [],
                }
            )
        limited_chunks, total_chunks, chunks_truncated = limit_items(compact_chunks, MAX_RETURN_CHUNKS)
        payload["chunks"] = limited_chunks
        payload["chunk_count"] = total_chunks
        payload["returned_chunk_count"] = len(limited_chunks)
        payload["chunks_truncated"] = chunks_truncated
        return strip_internal_run_context(payload)

    @staticmethod
    def _format_apply_result(
        *,
        source_path: Path,
        resume: bool,
        chunking_requirement: str | None,
        result: dict[str, Any],
        document_name: str,
        shard_count: int,
        max_workers: int,
        reference_bytes: int,
    ) -> dict[str, Any]:
        success_count = 1 if result.get("ok") else 0
        failure_count = 0 if result.get("ok") else 1
        failed_documents = [str(result.get("document_name", document_name))] if failure_count else []
        succeeded_documents = [str(result.get("document_name", document_name))] if success_count else []
        summarized_result = _ChunkApplyRuntimeTool._summarize_result(result)
        return strip_internal_run_context({
            "operation": "chunk_apply",
            "status": "success" if success_count else "error",
            "message": "Processed 1/1 document(s) successfully." if success_count else "Chunk apply failed for 1 document.",
            "path": str(source_path),
            "path_kind": "file",
            "resume": resume,
            "shard_count": shard_count,
            "max_workers": max_workers,
            "reference_bytes": reference_bytes,
            "chunking_requirement": chunking_requirement,
            "document_count": 1,
            "success_count": success_count,
            "failure_count": failure_count,
            "succeeded_documents": succeeded_documents,
            "failed_documents": failed_documents,
            "results": [summarized_result],
        })

    def _run_workflow(
        self,
        *,
        tool_input: ChunkApplyInput,
        stream_writer: Callable[[dict[str, Any]], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        source_path = Path(tool_input.path).expanduser().resolve()
        active_resume = tool_input.resume if tool_input.resume is not None else self.config.resolved_resume
        active_shard_count = max(1, int(tool_input.shard_count or self.config.runtime.shard_count))
        active_max_workers = max(1, int(tool_input.max_workers or self.config.runtime.max_workers or 1))
        active_reference_bytes = max(1, int(tool_input.reference_bytes or self.config.runtime.reference_bytes))
        active_stream_writer = stream_writer or self.config.runtime.stream_callback
        active_progress_callback = progress_callback or self.config.runtime.progress_callback
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        if not source_path.is_file():
            raise ValueError("chunk_apply only accepts a single file path.")

        document_name = self.document_name_for_source(source_path)
        run_id, thread_id = self.resolve_execution_identity(source_path, resume=active_resume)
        document_exists = self.chunk_store.document_exists(document_name=document_name, run_id=run_id)
        cached_record = self.cache.load(document_name=document_name, run_id=run_id, thread_id=thread_id) if active_resume else None
        source_fingerprint = build_source_fingerprint(source_path)
        if cached_record is not None and cached_record.get("completed"):
            cached_fingerprint = (cached_record.get("state") or {}).get("_source_fingerprint")
            if cached_fingerprint == source_fingerprint and document_exists:
                result = {
                    "ok": True,
                    "resumed": True,
                    "source_path": str(source_path),
                    "document_name": document_name,
                    "chunks": self.chunk_store.list_chunks(run_id=run_id, document_name=document_name),
                }
                return self._format_apply_result(
                    source_path=source_path,
                    resume=active_resume,
                    chunking_requirement=tool_input.chunking_requirement,
                    result=result,
                    document_name=document_name,
                    shard_count=active_shard_count,
                    max_workers=active_max_workers,
                    reference_bytes=active_reference_bytes,
                )
            if cached_fingerprint != source_fingerprint:
                if document_exists:
                    return self._error_result(
                        source_path=source_path,
                        document_name=document_name,
                        message="当前文档已经在memory中",
                        chunking_requirement=tool_input.chunking_requirement,
                        resume=active_resume,
                        shard_count=active_shard_count,
                        max_workers=active_max_workers,
                        reference_bytes=active_reference_bytes,
                    )
                active_resume = False
        elif document_exists:
            return self._error_result(
                source_path=source_path,
                document_name=document_name,
                message="当前文档已经在memory中",
                chunking_requirement=tool_input.chunking_requirement,
                resume=active_resume,
                shard_count=active_shard_count,
                max_workers=active_max_workers,
                reference_bytes=active_reference_bytes,
            )
        result = self.agent.process_document(
            source_path=source_path,
            run_id=run_id,
            thread_id=thread_id,
            cache=self.cache,
            staging_store=self.staging_store,
            checkpoint=self.checkpoint,
            chunk_store=self.chunk_store,
            resume=active_resume,
            chunking_requirement=tool_input.chunking_requirement,
            shard_count=active_shard_count,
            reference_bytes=active_reference_bytes,
            max_workers=active_max_workers,
            stream_writer=active_stream_writer,
            progress_callback=active_progress_callback,
        )
        return self._format_apply_result(
            source_path=source_path,
            resume=active_resume,
            chunking_requirement=tool_input.chunking_requirement,
            result=result,
            document_name=document_name,
            shard_count=active_shard_count,
            max_workers=active_max_workers,
            reference_bytes=active_reference_bytes,
        )

    def _run(
        self,
        path: str,
        resume: bool = True,
        chunking_requirement: str | None = None,
        shard_count: int | None = None,
        max_workers: int | None = None,
        reference_bytes: int | None = None,
        stream_writer: Callable[[dict[str, Any]], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        **_: Any,
        ) -> dict[str, Any]:
        return self._run_workflow(
            tool_input=ChunkApplyInput(
                path=path,
                resume=resume,
                chunking_requirement=chunking_requirement,
                shard_count=shard_count,
                max_workers=max_workers,
                reference_bytes=reference_bytes,
            ),
            stream_writer=stream_writer,
            progress_callback=progress_callback,
        )

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> dict[str, Any]:
        payload = input
        if isinstance(input, ChunkApplyInput):
            payload = input.model_dump()
        elif isinstance(input, dict) and "args" in input and isinstance(input.get("args"), dict):
            payload = input["args"]
        if not isinstance(payload, dict):
            raise TypeError("chunk_apply expects dict-like input with path/resume/chunking_requirement")
        tool_input = ChunkApplyInput(
            path=str(payload.get("path") or ""),
            resume=_coerce_bool(payload.get("resume", True)),
            chunking_requirement=payload.get("chunking_requirement"),
            shard_count=_coerce_positive_int(payload.get("shard_count")),
            max_workers=_coerce_positive_int(payload.get("max_workers")),
            reference_bytes=_coerce_positive_int(payload.get("reference_bytes")),
        )
        return self._run(
            path=tool_input.path,
            resume=tool_input.resume,
            chunking_requirement=tool_input.chunking_requirement,
            shard_count=tool_input.shard_count,
            max_workers=tool_input.max_workers,
            reference_bytes=tool_input.reference_bytes,
            stream_writer=kwargs.get("stream_writer"),
            progress_callback=kwargs.get("progress_callback"),
        )

    def close(self) -> None:
        self.chunk_store.close()
        if self._owns_cache:
            self.cache.close()
        if self._owns_staging_store:
            self.staging_store.close()
        if self._owns_checkpoint:
            self.checkpoint.close()

    @property
    def agent(self):
        return self._agent

    @property
    def cache(self):
        return self._cache

    @property
    def staging_store(self):
        return self._staging_store

    @property
    def checkpoint(self):
        return self._checkpoint

    @property
    def chunk_store(self):
        return self._chunk_store


class ChunkApplyTool:
    """Demo-style wrapper for the top-level chunk_apply workflow tool."""

    name = ChunkApplyToolSchema.name
    config = ChunkApplyToolConfig
    substate = ChunkApplySubState
    toolschema = ChunkApplyToolSchema

    def __init__(
        self,
        *,
        config: ChunkApplyToolConfig | None = None,
        middleware: ChunkingCapabilityMiddleware | None = None,
        overrides: ChunkApplyOverrides | dict[str, Any] | None = None,
    ) -> None:
        self.config = config or self.config.load()
        self._middleware_override = middleware
        self._overrides = overrides
        self._runner = self._build_runner(self.config)
        self.config = self._runner.config
        self.tool = self.create_tool()

    def _build_runner(self, config: ChunkApplyToolConfig) -> _ChunkApplyRuntimeTool:
        return _ChunkApplyRuntimeTool(
            config=config,
            middleware=self._middleware_override,
            overrides=self._overrides,
        )

    def create_tool(self):
        current_config = self.config
        current_toolschema = self.toolschema
        current_feedback_cls = current_toolschema.toolfeedback
        wrapper = self

        @tool(
            current_toolschema.name,
            args_schema=current_toolschema.args_schema,
            description=current_toolschema.description,
        )
        def chunk_apply(
            runtime: ToolRuntime[ChunkApplyToolConfig, ChunkApplySubState],
            path: str,
            resume: bool = True,
            chunking_requirement: str | None = None,
            shard_count: int | None = None,
            max_workers: int | None = None,
            reference_bytes: int | None = None,
        ) -> Command:
            feedback = current_feedback_cls()
            context = runtime.context or current_config
            runner = wrapper._runner if context == wrapper.config else wrapper._build_runner(context)
            emit(
                runtime.stream_writer,
                {
                    "type": "tool",
                    "tool": current_toolschema.name,
                    "event": "start",
                    "path": path,
                },
            )
            try:
                result = runner.invoke(
                    {
                        "path": path,
                        "resume": resume,
                        "chunking_requirement": chunking_requirement,
                        "shard_count": shard_count,
                        "max_workers": max_workers,
                        "reference_bytes": reference_bytes,
                    },
                    stream_writer=runtime.stream_writer,
                )
            except Exception as exc:
                message_text = feedback.failureText.format(error=str(exc))
                emit(
                    runtime.stream_writer,
                    {
                        "type": "tool",
                        "tool": current_toolschema.name,
                        "event": "error",
                        "error": message_text,
                    },
                )
                if runner is not wrapper._runner:
                    runner.close()
                return Command(
                    update={
                        "messages": [ToolMessage(content=message_text, tool_call_id=runtime.tool_call_id)],
                    }
                )
            if runner is not wrapper._runner:
                runner.close()
            emit(
                runtime.stream_writer,
                {
                    "type": "tool",
                    "tool": current_toolschema.name,
                    "event": "success",
                    "status": result.get("status"),
                    "success_count": int(result.get("success_count", 0)),
                    "failure_count": int(result.get("failure_count", 0)),
                },
            )
            message_text = json.dumps(result, ensure_ascii=False, default=str)
            return Command(
                update={
                    "messages": [ToolMessage(content=message_text, tool_call_id=runtime.tool_call_id)],
                    "chunkApplyStats": {
                        **(runtime.state.get("chunkApplyStats") or {}),
                        "lastSuccessCount": int(result.get("success_count", 0)),
                        "lastFailureCount": int(result.get("failure_count", 0)),
                    },
                }
            )

        return chunk_apply

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> dict[str, Any]:
        return self._runner.invoke(input, config=config, **kwargs)

    def run(self, tool_input: ChunkApplyInput | dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        if isinstance(tool_input, ChunkApplyInput):
            return self._runner._run_workflow(
                tool_input=tool_input,
                stream_writer=kwargs.get("stream_writer"),
                progress_callback=kwargs.get("progress_callback"),
            )
        if isinstance(tool_input, dict):
            return self.invoke(tool_input, **kwargs)
        return self._runner.run(tool_input=tool_input, **kwargs)

    def close(self) -> None:
        self._runner.close()

    @property
    def middleware(self):
        return self._runner.middleware

    @property
    def agent(self):
        return self._runner.agent

    @property
    def cache(self):
        return self._runner.cache

    @property
    def staging_store(self):
        return self._runner.staging_store

    @property
    def checkpoint(self):
        return self._runner.checkpoint

    @property
    def chunk_store(self):
        return self._runner.chunk_store


Config = ChunkApplyToolConfig
SubState = ChunkApplySubState
Input = ChunkApplyInput
ToolFeedback = ChunkApplyToolFeedback
ToolSchema = ChunkApplyToolSchema
Tool = ChunkApplyTool


def _embedding_override_from_public_config(config: ChunkApplyPublicConfig) -> dict[str, Any] | None:
    override = {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "base_url": config.embedding_base_url,
        "api_key": config.embedding_api_key,
        "dimensions": config.embedding_dimensions,
    }
    cleaned = {
        key: value
        for key, value in override.items()
        if value is not None and (not isinstance(value, str) or value.strip())
    }
    return cleaned or None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _coerce_positive_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be greater than 0")
    return parsed
