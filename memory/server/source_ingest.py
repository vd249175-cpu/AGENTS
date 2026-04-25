"""Single-file source ingestion for chunk_apply."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PLAIN_TEXT_SUFFIXES = {".txt", ".md"}


def load_single_source_document(source_path: str | Path) -> dict[str, Any]:
    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    if not path.is_file():
        raise ValueError("chunk_apply only accepts a single file path.")

    if path.suffix.lower() in PLAIN_TEXT_SUFFIXES:
        return {
            "source_path": str(path),
            "document_name": path.stem.strip() or path.stem or path.name or "doc",
            "text": path.read_text(encoding="utf-8"),
            "metadata": {"file_path": str(path)},
        }

    try:
        from llama_index.core import SimpleDirectoryReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading non-text files requires llama-index-core. "
            "Run `uv sync` to install package dependencies."
        ) from exc

    documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
    texts: list[str] = []
    merged_metadata: dict[str, Any] = {"file_path": str(path), "document_count": len(documents)}
    for index, document in enumerate(documents):
        text = str(getattr(document, "text", "") or "")
        if not text.strip():
            continue
        texts.append(text)
        metadata = getattr(document, "metadata", None)
        if index == 0 and isinstance(metadata, dict):
            merged_metadata.update(metadata)
    if not texts:
        raise ValueError(f"No readable text extracted from source file: {path}")
    return {
        "source_path": str(path),
        "document_name": path.stem.strip() or path.stem or path.name or "doc",
        "text": "\n\n".join(texts),
        "metadata": merged_metadata,
    }


__all__ = ["PLAIN_TEXT_SUFFIXES", "load_single_source_document"]
