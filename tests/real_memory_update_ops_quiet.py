import json
import subprocess
from typing import Any

from neo4j import GraphDatabase

from real_memory_db_ops_quiet import RUN_ID
from real_memory_ingest_quiet import (
    LOG_DIR,
    JsonlLog,
    Summary,
    compact_value,
    ensure_services,
    healthcheck,
    run_stage,
    short_id,
    stop_started,
)


def neo4j_payload() -> dict[str, Any]:
    return {
        "uri": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "1575338771",
        "database": None,
    }


def cypher_read(*, doc_a: str, doc_b: str, node_id: str, append_id: str) -> dict[str, Any]:
    config = neo4j_payload()
    with GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"])) as driver:
        with driver.session(database=config["database"]) as session:
            doc_a_chunks = list(
                session.run(
                    """
                    MATCH (chunk:Chunk {run_id: $run_id, document_name: $doc_a})
                    OPTIONAL MATCH (chunk)-[:HAS_KEYWORD]->(keyword:KeywordNode {run_id: $run_id})
                    WITH chunk, collect(DISTINCT keyword.keyword) AS keywords
                    RETURN coalesce(chunk.chunk_id, chunk.id) AS id,
                           chunk.chunk_index AS chunk_index,
                           chunk.summary AS summary,
                           coalesce(chunk.body, chunk.text, '') AS body,
                           keywords
                    ORDER BY chunk.chunk_index
                    """,
                    run_id=RUN_ID,
                    doc_a=doc_a,
                )
            )
            doc_b_chunk = session.run(
                """
                MATCH (chunk:Chunk {run_id: $run_id, document_name: $doc_b, chunk_index: 0})
                RETURN coalesce(chunk.chunk_id, chunk.id) AS id,
                       chunk.summary AS summary,
                       coalesce(chunk.body, chunk.text, '') AS body
                """,
                run_id=RUN_ID,
                doc_b=doc_b,
            ).single()
            node = session.run(
                """
                MATCH (node:GraphNode {run_id: $run_id, node_id: $node_id})
                OPTIONAL MATCH (node)-[:HAS_KEYWORD]->(keyword:KeywordNode {run_id: $run_id, owner_label: 'GraphNode', owner_id: $node_id})
                RETURN count(DISTINCT node) AS node_count,
                       collect(DISTINCT node.summary)[0] AS summary,
                       collect(DISTINCT node.body)[0] AS body,
                       collect(DISTINCT keyword.keyword) AS keywords
                """,
                run_id=RUN_ID,
                node_id=node_id,
            ).single()
            edges = list(
                session.run(
                    """
                    MATCH (source)-[edge:LINKS]-(target)
                    WHERE edge.run_id = $run_id
                      AND (
                        coalesce(source.node_id, source.chunk_id, source.id) IN $ids
                        OR coalesce(target.node_id, target.chunk_id, target.id) IN $ids
                      )
                    RETURN coalesce(source.node_id, source.chunk_id, source.id) AS source_id,
                           coalesce(target.node_id, target.chunk_id, target.id) AS target_id,
                           edge.u AS u,
                           edge.v AS v,
                           edge.dist AS dist
                    ORDER BY source_id, target_id
                    """,
                    run_id=RUN_ID,
                    ids=[node_id, append_id],
                )
            )
    return {
        "doc_a_chunks": [
            {
                "id": str(record["id"]),
                "chunk_index": int(record["chunk_index"]),
                "summary": str(record["summary"] or ""),
                "body": str(record["body"] or ""),
                "keywords": [str(keyword) for keyword in list(record["keywords"] or []) if keyword is not None],
            }
            for record in doc_a_chunks
        ],
        "doc_b_chunk": {
            "id": str(doc_b_chunk["id"]) if doc_b_chunk else "",
            "summary": str(doc_b_chunk["summary"] or "") if doc_b_chunk else "",
            "body": str(doc_b_chunk["body"] or "") if doc_b_chunk else "",
        },
        "node": {
            "node_count": int(node["node_count"]) if node else 0,
            "summary": str(node["summary"] or "") if node else "",
            "body": str(node["body"] or "") if node else "",
            "keywords": [str(keyword) for keyword in list(node["keywords"] or []) if keyword is not None] if node else [],
        },
        "edges": [
            {
                "source_id": str(record["source_id"]),
                "target_id": str(record["target_id"]),
                "u": str(record["u"] or ""),
                "v": str(record["v"] or ""),
                "dist": float(record["dist"] or 0),
            }
            for record in edges
        ],
    }


def cleanup_probe(*, doc_a: str, doc_b: str, node_id: str, append_id: str, log: JsonlLog) -> None:
    config = neo4j_payload()
    try:
        with GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"])) as driver:
            with driver.session(database=config["database"]) as session:
                session.run(
                    """
                    MATCH (node {run_id: $run_id})
                    WHERE node.document_name IN $documents
                       OR coalesce(node.node_id, node.chunk_id, node.id) IN $ids
                    DETACH DELETE node
                    """,
                    run_id=RUN_ID,
                    documents=[doc_a, doc_b],
                    ids=[node_id, append_id],
                ).consume()
        log.write("cleanup", {"ok": True, "documents": [doc_a, doc_b], "ids": [node_id, append_id]})
    except Exception as exc:
        log.write("cleanup", {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def has_inner_tool(stage: dict[str, Any], tool_name: str) -> bool:
    milestones = stage.get("milestones") if isinstance(stage.get("milestones"), list) else []
    response_text = json.dumps(compact_value(stage.get("response")), ensure_ascii=False, default=str)
    return any(tool_name in str(item) for item in milestones) or tool_name in response_text


def edge_exists(snapshot: dict[str, Any], left: str, right: str) -> bool:
    wanted = {left, right}
    for edge in snapshot["edges"]:
        ids = {edge["source_id"], edge["target_id"], edge["u"], edge["v"]}
        if wanted <= ids:
            return True
    return False


def verify(
    *,
    doc_a: str,
    doc_b: str,
    node_id: str,
    append_id: str,
    updated_doc_kw: str,
    updated_node_kw: str,
    setup: dict[str, Any],
    update_doc: dict[str, Any],
    update_graph: dict[str, Any],
    append_edges: dict[str, Any],
    snapshot: dict[str, Any],
    log: JsonlLog,
) -> dict[str, Any]:
    doc_a_chunks = snapshot["doc_a_chunks"]
    chunk0 = next((chunk for chunk in doc_a_chunks if chunk["chunk_index"] == 0), {})
    appended = next((chunk for chunk in doc_a_chunks if chunk["id"] == append_id), {})
    doc_b_id = snapshot["doc_b_chunk"]["id"]
    checks = [
        {"name": "setup_stage_ok", "ok": bool(setup.get("ok"))},
        {"name": "setup_created_documents", "ok": len(doc_a_chunks) >= 1 and bool(doc_b_id)},
        {"name": "setup_created_node", "ok": snapshot["node"]["node_count"] == 1},
        {"name": "update_doc_stage_ok", "ok": bool(update_doc.get("ok"))},
        {"name": "update_chunks_called", "ok": has_inner_tool(update_doc, "update_chunks")},
        {"name": "chunk_summary_updated", "ok": "更新后的文档摘要" in str(chunk0.get("summary"))},
        {"name": "chunk_body_updated", "ok": "更新后的实际正文" in str(chunk0.get("body"))},
        {"name": "chunk_keyword_updated", "ok": updated_doc_kw in json.dumps(chunk0, ensure_ascii=False, default=str)},
        {"name": "update_graph_stage_ok", "ok": bool(update_graph.get("ok"))},
        {"name": "graph_update_node_called", "ok": has_inner_tool(update_graph, "graph_update_node")},
        {"name": "node_summary_updated", "ok": "更新后的普通节点摘要" in snapshot["node"]["summary"]},
        {"name": "node_body_updated", "ok": "更新后的普通节点实际内容" in snapshot["node"]["body"]},
        {"name": "node_keyword_updated", "ok": updated_node_kw in json.dumps(snapshot["node"], ensure_ascii=False, default=str)},
        {"name": "node_edge_to_doc_a", "ok": edge_exists(snapshot, node_id, str(chunk0.get("id", "")))},
        {"name": "append_stage_ok", "ok": bool(append_edges.get("ok"))},
        {"name": "insert_chunks_called", "ok": has_inner_tool(append_edges, "insert_chunks")},
        {"name": "append_chunk_exists", "ok": bool(appended)},
        {"name": "append_summary_saved", "ok": "续写新增文档节点" in str(appended.get("summary"))},
        {"name": "append_edge_to_graph_node", "ok": edge_exists(snapshot, append_id, node_id)},
        {"name": "append_edge_to_other_document", "ok": edge_exists(snapshot, append_id, doc_b_id)},
    ]
    failed = [check for check in checks if not check["ok"]]
    result = {
        "ok": not failed,
        "reason": None if not failed else failed[0]["name"],
        "checks": checks,
        "snapshot": snapshot,
        "doc_a": doc_a,
        "doc_b": doc_b,
    }
    log.write("verification", result)
    return result


def main() -> int:
    log = JsonlLog()
    log.path = LOG_DIR / log.path.name.replace("real_memory_ingest_", "real_memory_update_ops_")
    log.full_path = LOG_DIR / log.full_path.name.replace("real_memory_ingest_", "real_memory_update_ops_")
    summary = Summary()
    started: list[subprocess.Popen[str]] = []
    token = short_id()
    doc_a = f"update_doc_a_{token}"
    doc_b = f"update_doc_b_{token}"
    node_id = f"updatenode{token}"
    append_id = f"appendchunk{token}"
    updated_doc_kw = f"updated_doc_kw_{token}"
    updated_node_kw = f"updated_node_kw_{token}"
    final_ok = False

    summary.add("log", str(log.path))
    summary.add("log_tail", f"tail -n 80 {log.path}")
    summary.add("full_log", str(log.full_path))
    try:
        started = ensure_services(log)
        health = healthcheck(log)
        summary.add("setup", f"health_ok={health['ok']} run_id={RUN_ID}")
        cleanup_probe(doc_a=doc_a, doc_b=doc_b, node_id=node_id, append_id=append_id, log=log)

        setup_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"1) 使用 create_chunk_document 创建文档 {doc_a}，summary='原始文档A摘要'，body='原始文档A正文'，keywords=['原始A'];"
            f"2) 使用 create_chunk_document 创建文档 {doc_b}，summary='其他文档B摘要'，body='其他文档B正文'，keywords=['其他B'];"
            f"3) 使用 graph_create_nodes 创建 GraphNode，ids=['{node_id}']，summary='原始普通节点摘要'，body='原始普通节点正文'，"
            "keyword_ops=[{'op': '+', 'keywords': ['原始普通节点']}];"
            "完成后只总结三个创建动作的结果。"
        )
        setup = run_stage("setup_records", setup_prompt, log)
        summary.add("setup_records", f"ok={setup['ok']} milestones={', '.join(setup['milestones'][:8])}")

        update_doc_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"使用 update_chunks 更新文档 {doc_a} 的 chunk_index=0："
            "summary 改为 '更新后的文档摘要'，body 改为 '更新后的实际正文'，"
            f"keyword_ops 使用 [{{'op': 'replace', 'keywords': ['{updated_doc_kw}', '更新关键词']}}]。"
            "完成后用 query_chunk_positions 读取该 chunk，确认摘要、正文和关键词已更新。"
        )
        update_doc = run_stage("update_doc", update_doc_prompt, log)
        summary.add("update_doc", f"ok={update_doc['ok']} milestones={', '.join(update_doc['milestones'][:8])}")

        update_graph_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"使用 graph_update_node 更新 node id {node_id}："
            "summary 改为 '更新后的普通节点摘要'，body 改为 '更新后的普通节点实际内容'，"
            f"keyword_ops 使用 [{{'op': 'replace', 'keywords': ['{updated_node_kw}', '更新节点关键词']}}]，"
            f"edge_ops 添加一条到文档 {doc_a} 的 chunk_index 0 的边："
            f"[{{'op': '+', 'targets': [0], 'document_name': '{doc_a}', 'dist': 0.42}}]。"
            "完成后 read_nodes detail 读取该 node。"
        )
        update_graph = run_stage("update_graph", update_graph_prompt, log)
        summary.add("update_graph", f"ok={update_graph['ok']} milestones={', '.join(update_graph['milestones'][:8])}")

        append_edges_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"先使用 insert_chunks 续写文档 {doc_a}，新增一个 chunk，id 精确为 {append_id}，"
            "summary='续写新增文档节点'，body='续写新增文档节点的实际内容'，keywords=['续写节点'];"
            f"然后使用 graph_update_node 更新刚新增的 Chunk id {append_id}，只添加普通边，"
            f"edge_ops 包含两条：第一条连到普通 GraphNode {node_id}，第二条连到其他文档 {doc_b} 的 chunk_index 0。"
            f"第二条写成 {{'op': '+', 'targets': [0], 'document_name': '{doc_b}', 'dist': 0.44}}。"
            "完成后用 query_chunk_positions 读取新增 chunk。"
        )
        append_edges = run_stage("append_edges", append_edges_prompt, log)
        summary.add("append_edges", f"ok={append_edges['ok']} milestones={', '.join(append_edges['milestones'][:10])}")

        snapshot = cypher_read(doc_a=doc_a, doc_b=doc_b, node_id=node_id, append_id=append_id)
        log.write("neo4j_snapshot", snapshot)
        verification = verify(
            doc_a=doc_a,
            doc_b=doc_b,
            node_id=node_id,
            append_id=append_id,
            updated_doc_kw=updated_doc_kw,
            updated_node_kw=updated_node_kw,
            setup=setup,
            update_doc=update_doc,
            update_graph=update_graph,
            append_edges=append_edges,
            snapshot=snapshot,
            log=log,
        )
        summary.add("verify", f"ok={verification['ok']} reason={verification.get('reason')}")
        final_ok = bool(health["ok"] and verification["ok"])
        summary.add("summary", f"final_ok={final_ok}")
        log.write("summary", {"lines": summary.lines, "final_ok": final_ok})
        return 0 if final_ok else 1
    finally:
        cleanup_probe(doc_a=doc_a, doc_b=doc_b, node_id=node_id, append_id=append_id, log=log)
        stop_started(started, log)


if __name__ == "__main__":
    raise SystemExit(main())
