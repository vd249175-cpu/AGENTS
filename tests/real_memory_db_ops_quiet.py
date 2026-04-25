import json
import subprocess
from typing import Any

from neo4j import GraphDatabase

from real_memory_ingest_quiet import (
    LOG_DIR,
    ROOT,
    JsonlLog,
    Summary,
    compact_value,
    ensure_services,
    healthcheck,
    run_stage,
    short_id,
    stop_started,
)


RUN_ID = "SeedAgent-knowledge"


def neo4j_payload() -> dict[str, Any]:
    return {
        "uri": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "1575338771",
        "database": None,
    }


def neo4j_read(*, document_name: str, node_id: str) -> dict[str, Any]:
    config = neo4j_payload()
    with GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"])) as driver:
        with driver.session(database=config["database"]) as session:
            doc = session.run(
                """
                MATCH (chunk:Chunk {run_id: $run_id, document_name: $document_name})
                RETURN count(chunk) AS chunk_count,
                       collect(chunk.summary)[0..3] AS summaries,
                       collect(chunk.keywords)[0..3] AS keywords
                """,
                run_id=RUN_ID,
                document_name=document_name,
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
    return {
        "document": {
            "chunk_count": int(doc["chunk_count"]) if doc else 0,
            "summaries": list(doc["summaries"] or []) if doc else [],
            "keywords": list(doc["keywords"] or []) if doc else [],
        },
        "node": {
            "node_count": int(node["node_count"]) if node else 0,
            "summary": str(node["summary"] or "") if node else "",
            "body": str(node["body"] or "") if node else "",
            "keywords": list(node["keywords"] or []) if node and node["keywords"] else [],
        },
    }


def cleanup_probe(*, document_name: str, node_id: str, log: JsonlLog) -> None:
    config = neo4j_payload()
    try:
        with GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"])) as driver:
            with driver.session(database=config["database"]) as session:
                session.run(
                    """
                    MATCH (node {run_id: $run_id})
                    WHERE node.node_id = $node_id OR node.document_name = $document_name
                    DETACH DELETE node
                    """,
                    run_id=RUN_ID,
                    node_id=node_id,
                    document_name=document_name,
                ).consume()
        log.write("cleanup", {"ok": True, "document_name": document_name, "node_id": node_id})
    except Exception as exc:
        log.write("cleanup", {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def has_inner_tool(stage: dict[str, Any], tool_name: str) -> bool:
    milestones = stage.get("milestones") if isinstance(stage.get("milestones"), list) else []
    response_text = json.dumps(compact_value(stage.get("response")), ensure_ascii=False, default=str)
    return any(tool_name in str(item) for item in milestones) or tool_name in response_text


def verify(
    *,
    document_name: str,
    node_id: str,
    doc_keyword: str,
    node_keyword: str,
    create_doc: dict[str, Any],
    create_node: dict[str, Any],
    recall: dict[str, Any],
    delete_node: dict[str, Any],
    before_delete: dict[str, Any],
    after_delete: dict[str, Any],
    log: JsonlLog,
) -> dict[str, Any]:
    recall_text = json.dumps(compact_value(recall.get("response")), ensure_ascii=False, default=str)
    checks = [
        {"name": "create_doc_stage_ok", "ok": bool(create_doc.get("ok"))},
        {"name": "create_chunk_document_called", "ok": has_inner_tool(create_doc, "create_chunk_document")},
        {"name": "document_exists", "ok": before_delete["document"]["chunk_count"] >= 1},
        {
            "name": "document_keyword_saved",
            "ok": doc_keyword in json.dumps(before_delete["document"], ensure_ascii=False, default=str),
        },
        {"name": "create_node_stage_ok", "ok": bool(create_node.get("ok"))},
        {"name": "graph_create_nodes_called", "ok": has_inner_tool(create_node, "graph_create_nodes")},
        {"name": "node_exists_before_delete", "ok": before_delete["node"]["node_count"] == 1},
        {
            "name": "node_keyword_saved",
            "ok": node_keyword in json.dumps(before_delete["node"], ensure_ascii=False, default=str),
        },
        {"name": "recall_stage_ok", "ok": bool(recall.get("ok"))},
        {"name": "keyword_recall_called", "ok": has_inner_tool(recall, "keyword_recall")},
        {"name": "keyword_recall_hit", "ok": node_id in recall_text or node_keyword in recall_text},
        {"name": "delete_stage_ok", "ok": bool(delete_node.get("ok"))},
        {"name": "graph_delete_nodes_called", "ok": has_inner_tool(delete_node, "graph_delete_nodes")},
        {"name": "node_deleted", "ok": after_delete["node"]["node_count"] == 0},
        {"name": "document_still_exists", "ok": after_delete["document"]["chunk_count"] >= 1},
    ]
    failed = [check for check in checks if not check["ok"]]
    result = {
        "ok": not failed,
        "checks": checks,
        "reason": None if not failed else failed[0]["name"],
        "before_delete": before_delete,
        "after_delete": after_delete,
    }
    log.write("verification", result)
    return result


def main() -> int:
    log = JsonlLog()
    log.path = LOG_DIR / log.path.name.replace("real_memory_ingest_", "real_memory_db_ops_")
    log.full_path = LOG_DIR / log.full_path.name.replace("real_memory_ingest_", "real_memory_db_ops_")
    summary = Summary()
    started: list[subprocess.Popen[str]] = []
    token = short_id()
    document_name = f"db_ops_doc_{token}"
    node_id = f"dbopsnode{token}"
    doc_keyword = f"dbops_doc_kw_{token}"
    node_keyword = f"dbops_node_kw_{token}"
    final_ok = False

    summary.add("log", str(log.path))
    summary.add("log_tail", f"tail -n 80 {log.path}")
    summary.add("full_log", str(log.full_path))
    try:
        started = ensure_services(log)
        health = healthcheck(log)
        summary.add("setup", f"health_ok={health['ok']} run_id={RUN_ID}")
        cleanup_probe(document_name=document_name, node_id=node_id, log=log)

        create_doc_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"使用 create_chunk_document 创建新文档 {document_name}，"
            f"summary 写 数据库操作文档 {token}，body 写 文档创建测试 {doc_keyword}，"
            f"keywords 精确包含 ['{doc_keyword}', '文档创建测试']。"
            "完成后总结 create_chunk_document 的结果。"
        )
        create_doc = run_stage("create_doc", create_doc_prompt, log)
        summary.add("create_doc", f"ok={create_doc['ok']} milestones={', '.join(create_doc['milestones'][:6])}")

        create_node_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"使用 graph_create_nodes 创建一个 GraphNode，ids 精确传 ['{node_id}']，"
            f"summary 和 body 都包含 节点创建测试 {node_keyword}，"
            f"keyword_ops 使用 [{{'op': '+', 'keywords': ['{node_keyword}', '节点创建测试']}}]。"
            "完成后总结 graph_create_nodes 的结果，并原样写出真实 node id。"
        )
        create_node = run_stage("create_node", create_node_prompt, log)
        summary.add("create_node", f"ok={create_node['ok']} milestones={', '.join(create_node['milestones'][:6])}")

        recall_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"使用 keyword_recall 查询关键词 ['{node_keyword}']，top_k=5，detail_mode='detail'，"
            f"确认结果里能看到 GraphNode {node_id} 或关键词 {node_keyword}。"
            "完成后简短总结召回命中情况。"
        )
        recall = run_stage("recall", recall_prompt, log)
        summary.add("recall", f"ok={recall['ok']} milestones={', '.join(recall['milestones'][:6])}")

        before_delete = neo4j_read(document_name=document_name, node_id=node_id)
        log.write("neo4j_before_delete", before_delete)

        delete_prompt = (
            "必须调用 manage_knowledge 工具一次。target 请写成："
            f"使用 graph_delete_nodes 删除 GraphNode id {node_id}。"
            "只删除这个 GraphNode，不要删除文档 chunk。完成后总结删除结果。"
        )
        delete_node = run_stage("delete_node", delete_prompt, log)
        summary.add("delete_node", f"ok={delete_node['ok']} milestones={', '.join(delete_node['milestones'][:6])}")

        after_delete = neo4j_read(document_name=document_name, node_id=node_id)
        log.write("neo4j_after_delete", after_delete)

        verification = verify(
            document_name=document_name,
            node_id=node_id,
            doc_keyword=doc_keyword,
            node_keyword=node_keyword,
            create_doc=create_doc,
            create_node=create_node,
            recall=recall,
            delete_node=delete_node,
            before_delete=before_delete,
            after_delete=after_delete,
            log=log,
        )
        summary.add("verify", f"ok={verification['ok']} reason={verification.get('reason')}")
        final_ok = bool(health["ok"] and verification["ok"])
        summary.add("summary", f"final_ok={final_ok}")
        log.write("summary", {"lines": summary.lines, "final_ok": final_ok})
        return 0 if final_ok else 1
    finally:
        cleanup_probe(document_name=document_name, node_id=node_id, log=log)
        stop_started(started, log)


if __name__ == "__main__":
    raise SystemExit(main())
