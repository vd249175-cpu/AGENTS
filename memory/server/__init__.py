"""Server package."""

from .source_ingest import PLAIN_TEXT_SUFFIXES, load_single_source_document

__all__ = [
    "PLAIN_TEXT_SUFFIXES",
    "load_single_source_document",
]
