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


def cypher_read(*, document_name: str, node_id: str, delete_chunk_id: str) -> dict[str, Any]:
    config = neo4j_payload()
    with GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"])) as driver:
        with driver.session(database=config["database"]) as session:
            chunks = list(
                session.run(
                    """
                    MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name})
                    RETURN coalesce(chunk.chunk_id, chunk.id) AS id,
                           chunk.chunk_index AS chunk_index,
                           chunk.summary AS summary
                    ORDER BY chunk.chunk_index
                    """,
                    run_id=RUN_ID,
                    document_name=document_name,
                )
            )
            node = session.run(
                """
                MATCH (node:GraphNode {run_id: $run_id, node_id: $node_id})
                RETURN count(DISTINCT node) AS node_count
                """,
                run_id=RUN_ID,
                node_id=node_id,
            ).single()
            edge_records = list(
                session.run(
                    """
                    MATCH (left)-[edge:LINKS]-(right)
                    WHERE edge.run_id = $run_id
                      AND (
                        coalesce(left.node_id, left.chunk_id, left.id) = $node_id
                        OR coalesce(right.node_id, right.chunk_id, right.id) = $node_id
                        OR coalesce(left.node_id, left.chunk_id, left.id) = $delete_chunk_id
                        OR coalesce(right.node_id, right.chunk_id, right.id) = $delete_chunk_id
                      )
                    RETURN coalesce(left.node_id, left.chunk_id, left.id) AS left_id,
                           coalesce(right.node_id, right.chunk_id, right.id) AS right_id,
                           edge.u AS u,
                           edge.v AS v
                    ORDER BY left_id, right_id
                    """,
                    run_id=RUN_ID,
                    node_id=node_id,
                    delete_chunk_id=delete_chunk_id,
                )
            )
    return {
        "chunks": [
            {
                "id": str(record["id"]),
                "chunk_index": int(record["chunk_index"]),
                "summary": str(record["summary"] or ""),
            }
            for record in chunks
        ],
        "node_count": int(node["node_count"]) if node else 0,
        "edges": [
            {
                "left_id": str(record["left_id"]),
                "right_id": str(record["right_id"]),
                "u": str(record["u"] or ""),
                "v": str(record["v"] or ""),
            }
            for record in edge_records
        ],
    }


def cleanup_probe(*, document_name: str, node_id: str, delete_chunk_id: str, log: JsonlLog) -> None:
    config = neo4j_payload()
    try:
        with GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"])) as driver:
            with driver.session(database=config["database"]) as session:
                session.run(
                    """
                    MATCH (node {run_id: $run_id})
                    WHERE node.document_name = $document_name
                       OR coalesce(node.node_id, node.chunk_id, node.id) IN $ids
                    DETACH DELETE node
                    """,
                    run_id=RUN_ID,
                    document_name=document_name,
                    ids=[node_id, delete_chunk_id],
                ).consume()
        log.write("cleanup", {"ok": True, "document_name": document_name, "ids": [node_id, delete_chunk_id]})
    except Exception as exc:
        log.write("cleanup", {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def has_inner_tool(stage: dict[str, Any], tool_name: str) -> bool:
    milestones = stage.get("milestones") if isinstance(stage.get("milestones"), list) else []
    response_text = json.dumps(compact_value(stage.get("response")), ensure_ascii=False, default=str)
    return any(tool_name in str(item) for item in milestones) or tool_name in response_text


def edge_between(snapshot: dict[str, Any], left: str, right: str) -> bool:
    wanted = {left, right}
    for edge in snapshot["edges"]:
        if wanted <= {edge["left_id"], edge["right_id"], edge["u"], edge["v"]}:
            return True
    return False


def verify(
    *,
    document_name: str,
    node_id: str,
    delete_chunk_id: str,
    setup: dict[str, Any],
    delete_edge: dict[str, Any],
    delete_chunk: dict[str, Any],
    state_bucket: dict[str, Any],
    before: dict[str, Any],
    after_edge: dict[str, Any],
    after_chunk: dict[str, Any],
    log: JsonlLog,
) -> dict[str, Any]:
    chunk0 = next((chunk for chunk in before["chunks"] if chunk["chunk_index"] == 0), {})
    chunk0_id = str(chunk0.get("id") or "")
    checks = [
        {"name": "setup_stage_ok", "ok": bool(setup.get("ok"))},
        {"name": "setup_created_document_chunks", "ok": len(before["chunks"]) >= 2},
        {"name": "setup_created_node", "ok": before["node_count"] == 1},
        {"name": "setup_edge_exists", "ok": edge_between(before, node_id, chunk0_id)},
        {"name": "delete_edge_stage_ok", "ok": bool(delete_edge.get("ok"))},
        {"name": "delete_edge_tool_called", "ok": has_inner_tool(delete_edge, "graph_update_node")},
        {"name": "edge_removed", "ok": not edge_between(after_edge, node_id, chunk0_id)},
        {"name": "delete_chunk_stage_ok", "ok": bool(delete_chunk.get("ok"))},
        {"name": "delete_chunks_tool_called", "ok": has_inner_tool(delete_chunk, "delete_chunks")},
        {"name": "chunk_deleted", "ok": delete_chunk_id not in {chunk["id"] for chunk in after_chunk["chunks"]}},
        {"name": "document_still_has_one_chunk", "ok": len(after_chunk["chunks"]) == 1},
        {"name": "state_stage_ok", "ok": bool(state_bucket.get("ok"))},
        {"name": "mark_useful_called", "ok": has_inner_tool(state_bucket, "graph_mark_useful")},
        {"name": "mark_blocked_called", "ok": has_inner_tool(state_bucket, "graph_mark_blocked")},
        {"name": "clear_blocked_called", "ok": has_inner_tool(state_bucket, "graph_clear_blocked")},
    ]
    failed = [check for check in checks if not check["ok"]]
    result = {
        "ok": not failed,
        "reason": None if not failed else failed[0]["name"],
        "checks": checks,
        "document_name": document_name,
        "node_id": node_id,
        "delete_chunk_id": delete_chunk_id,
        "before": before,
        "after_edge": after_edge,
        "after_chunk": after_chunk,
    }
    log.write("verification", result)
    return result


def main() -> int:
    log = JsonlLog()
    log.path = LOG_DIR / log.path.name.replace("real_memory_ingest_", "real_memory_delete_state_")
    log.full_path = LOG_DIR / log.full_path.name.replace("real_memory_ingest_", "real_memory_delete_state_")
    summary = Summary()
    started: list[subprocess.Popen[str]] = []
    token = short_id()
    document_name = f"delete_state_doc_{token}"
    node_id = f"deletenode{token}"
    delete_chunk_id = f"deletechunk{token}"
    final_ok = False

    summary.add("log", str(log.path))
    summary.add("log_tail", f"tail -n 80 {log.path}")
    summary.add("full_log", str(log.full_path))
    try:
        started = ensure_services(log)
        health = healthcheck(log)
        summary.add("setup", f"health_ok={health['ok']} run_id={RUN_ID}")
        cleanup_probe(document_name=document_name, node_id=node_id, delete_chunk_id=delete_chunk_id, log=log)

        setup_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"1) 使用 create_chunk_document 创建文档 {document_name}，summary='保留chunk摘要'，body='保留chunk正文'，keywords=['保留chunk'];"
            f"2) 使用 insert_chunks 向 {document_name} 插入一个 chunk，id 精确为 {delete_chunk_id}，summary='待删除chunk摘要'，body='待删除chunk正文'，keywords=['待删除chunk'];"
            f"3) 使用 graph_create_nodes 创建 GraphNode，ids=['{node_id}']，summary='删除边测试节点'，body='删除边测试节点正文'，"
            "keyword_ops=[{'op': '+', 'keywords': ['删除边测试节点']}];"
            f"4) 使用 graph_update_node 给 {node_id} 添加一条到文档 {document_name} chunk_index 0 的边。"
            "完成后简短总结。"
        )
        setup = run_stage("setup_records", setup_prompt, log)
        summary.add("setup_records", f"ok={setup['ok']} milestones={', '.join(setup['milestones'][:10])}")
        before = cypher_read(document_name=document_name, node_id=node_id, delete_chunk_id=delete_chunk_id)
        log.write("neo4j_before", before)

        delete_edge_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"使用 graph_update_node 更新 node id {node_id}，只删除普通边，"
            f"edge_ops=[{{'op': '-', 'targets': [0], 'document_name': '{document_name}'}}]。"
            "不要删除任何节点或 chunk。完成后总结删除边结果。"
        )
        delete_edge = run_stage("delete_edge", delete_edge_prompt, log)
        summary.add("delete_edge", f"ok={delete_edge['ok']} milestones={', '.join(delete_edge['milestones'][:8])}")
        after_edge = cypher_read(document_name=document_name, node_id=node_id, delete_chunk_id=delete_chunk_id)
        log.write("neo4j_after_edge", after_edge)

        delete_chunk_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"使用 delete_chunks 从文档 {document_name} 删除 id 为 {delete_chunk_id} 的 chunk。"
            "不要删除 GraphNode，不要删除保留 chunk。完成后用 query_chunk_positions 或 list_chunk_documents 复核。"
        )
        delete_chunk = run_stage("delete_chunk", delete_chunk_prompt, log)
        summary.add("delete_chunk", f"ok={delete_chunk['ok']} milestones={', '.join(delete_chunk['milestones'][:8])}")
        after_chunk = cypher_read(document_name=document_name, node_id=node_id, delete_chunk_id=delete_chunk_id)
        log.write("neo4j_after_chunk", after_chunk)

        state_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"依次调用 graph_mark_useful 标记 {node_id}，rationale='状态桶测试 useful'；"
            f"再调用 graph_mark_blocked 标记 {node_id}，rationale='状态桶测试 blocked'；"
            f"最后调用 graph_clear_blocked 清除 {node_id}。"
            "不要新增或删除数据库节点。完成后总结 useful/blocked 状态桶动作。"
        )
        state_bucket = run_stage("state_bucket", state_prompt, log)
        summary.add("state_bucket", f"ok={state_bucket['ok']} milestones={', '.join(state_bucket['milestones'][:10])}")

        verification = verify(
            document_name=document_name,
            node_id=node_id,
            delete_chunk_id=delete_chunk_id,
            setup=setup,
            delete_edge=delete_edge,
            delete_chunk=delete_chunk,
            state_bucket=state_bucket,
            before=before,
            after_edge=after_edge,
            after_chunk=after_chunk,
            log=log,
        )
        summary.add("verify", f"ok={verification['ok']} reason={verification.get('reason')}")
        final_ok = bool(health["ok"] and verification["ok"])
        summary.add("summary", f"final_ok={final_ok}")
        log.write("summary", {"lines": summary.lines, "final_ok": final_ok})
        return 0 if final_ok else 1
    finally:
        cleanup_probe(document_name=document_name, node_id=node_id, delete_chunk_id=delete_chunk_id, log=log)
        stop_started(started, log)


if __name__ == "__main__":
    raise SystemExit(main())
