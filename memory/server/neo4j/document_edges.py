"""Document-internal edge helpers for Chunk records."""

from __future__ import annotations

from neo4j import ManagedTransaction


DEFAULT_DOCUMENT_EDGE_DISTANCE = 0.3
DOCUMENT_ORDER_RELATIONSHIP = "DOCUMENT_NEXT"


def rebuild_document_order_edges(
    transaction: ManagedTransaction,
    *,
    run_id: str,
    document_name: str,
    distance: float = DEFAULT_DOCUMENT_EDGE_DISTANCE,
) -> None:
    """Rebuild only the document-order edges for one document.

    Normal graph edges deliberately use other relationship types/properties and
    are not matched here.
    """
    transaction.run(
        """
        MATCH (:Chunk {run_id: $run_id, document_name: $document_name})-[edge:DOCUMENT_NEXT]->
              (:Chunk {run_id: $run_id, document_name: $document_name})
        DELETE edge
        """,
        run_id=run_id,
        document_name=document_name,
    )
    transaction.run(
        """
        MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name})
        WITH chunk
        ORDER BY chunk.chunk_index
        WITH collect(chunk) AS chunks
        WHERE size(chunks) > 1
        UNWIND range(0, size(chunks) - 2) AS position
        WITH chunks[position] AS source, chunks[position + 1] AS target, position
        MERGE (source)-[edge:DOCUMENT_NEXT]->(target)
        SET
            edge.edge_key = $run_id + ':' + $document_name + ':' + source.chunk_id + '->' + target.chunk_id,
            edge.run_id = $run_id,
            edge.document_name = $document_name,
            edge.edge_kind = 'document_order',
            edge.source_chunk_id = source.chunk_id,
            edge.target_chunk_id = target.chunk_id,
            edge.source_position = position,
            edge.target_position = position + 1,
            edge.dist = $distance,
            edge.distance = $distance
        """,
        run_id=run_id,
        document_name=document_name,
        distance=float(distance),
    )
