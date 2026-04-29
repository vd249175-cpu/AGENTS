import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class MainServerAdminConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        os.environ["MAIN_SERVER_AGENT_CONFIG"] = f"{self.tempdir.name}/agents.json"

        from MainServer import main_server

        self.main_server = main_server
        self.main_server._registry.clear()
        self.main_server._mailboxes.clear()
        self.main_server._mail_log.clear()
        self.client = TestClient(main_server.app)

    def tearDown(self) -> None:
        os.environ.pop("MAIN_SERVER_AGENT_CONFIG", None)

    def test_communication_spaces_create_venn_style_peer_sets(self) -> None:
        response = self.client.put(
            "/admin/communication",
            json={
                "spaces": [
                    {"id": "left", "name": "Left", "members": ["AgentA", "AgentBridge"]},
                    {"id": "right", "name": "Right", "members": ["AgentBridge", "AgentB"]},
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        edges = {(edge["from"], edge["to"]) for edge in response.json()["edges"]}
        self.assertEqual(edges, {("AgentA", "AgentBridge"), ("AgentB", "AgentBridge")})

        for agent_name in ("AgentA", "AgentBridge", "AgentB"):
            response = self.client.post("/agents/register", json={"agent_name": agent_name})
            self.assertEqual(response.status_code, 200)

        response = self.client.get("/agents/peers/AgentBridge")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["peers"], ["AgentA", "AgentB"])

        response = self.client.get("/agents/peers/AgentA")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["peers"], ["AgentBridge"])

    def test_send_and_recv_use_global_communication_spaces(self) -> None:
        self.client.put(
            "/admin/communication",
            json={"spaces": [{"id": "main", "members": ["SeedAgent", "WorkerAgent"]}]},
        )

        for agent_name in ("SeedAgent", "WorkerAgent", "OtherAgent"):
            response = self.client.post("/agents/register", json={"agent_name": agent_name})
            self.assertEqual(response.status_code, 200)

        allowed = {
            "message_id": "msg-1",
            "from": "SeedAgent",
            "to": "WorkerAgent",
            "type": "message",
            "content": "hello",
            "attachments": [],
        }
        response = self.client.post("/send", json=allowed)
        self.assertEqual(response.status_code, 200)

        blocked = dict(allowed, message_id="msg-2", to="OtherAgent")
        response = self.client.post("/send", json=blocked)
        self.assertEqual(response.status_code, 403)

        response = self.client.get("/recv/WorkerAgent")
        self.assertEqual(response.status_code, 200)
        messages = response.json()["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["from"], "SeedAgent")

    def test_send_wakes_registered_receiver_service(self) -> None:
        calls = []

        def fake_post_json(url, payload, timeout):
            calls.append({"url": url, "payload": payload, "timeout": timeout})
            return {"ok": True, "reply": "handled inbox"}

        original_post_json = self.main_server.post_json
        self.main_server.post_json = fake_post_json
        self.addCleanup(setattr, self.main_server, "post_json", original_post_json)

        self.client.put(
            "/admin/communication",
            json={"spaces": [{"id": "main", "members": ["AgentA", "AgentB"]}]},
        )
        self.client.post("/agents/register", json={"agent_name": "AgentA"})
        self.client.post(
            "/agents/register",
            json={"agent_name": "AgentB", "metadata": {"service_url": "http://127.0.0.1:8011"}},
        )
        self.client.post(
            "/agents/AgentB/status",
            json={"status": "running", "phase": "ready", "step": "waiting"},
        )
        self.client.put(
            "/user/chat/config/AgentB",
            json={
                "thread_id": "receiver-thread",
                "run_id": "receiver-run",
                "stream_mode": ["messages", "updates", "custom"],
                "version": "v2",
            },
        )

        response = self.client.post(
            "/send",
            json={
                "message_id": "wake-1",
                "from": "AgentA",
                "to": "AgentB",
                "type": "message",
                "content": "please handle this",
                "attachments": [],
                "metadata": {"thread_id": "sender-thread", "run_id": "sender-run"},
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["wake_scheduled"])
        self.assertEqual(calls[0]["url"], "http://127.0.0.1:8011/invoke")
        self.assertEqual(calls[0]["payload"]["session_id"], "receiver-thread")
        self.assertEqual(calls[0]["payload"]["run_id"], "receiver-run")
        self.assertIn("mail_metadata_thread_id=sender-thread", calls[0]["payload"]["messages"][0]["content"])
        self.assertEqual(calls[0]["payload"]["stream_mode"], ["updates", "custom", "messages"])
        events = self.client.get("/agents/AgentB").json()["events"]
        wake_success = [event for event in events if event.get("event") == "mail_wake_success"][-1]
        self.assertEqual(wake_success["reply"], "handled inbox")
        self.assertTrue(wake_success["has_reply"])

    def test_empty_communication_spaces_default_to_all_registered_peers(self) -> None:
        self.client.post(
            "/agents/register",
            json={"agent_name": "AgentA", "scope": [[], "AgentB", []]},
        )
        self.client.post("/agents/register", json={"agent_name": "AgentB"})
        self.client.post("/agents/register", json={"agent_name": "AgentC"})

        response = self.client.get("/agents/peers/AgentA")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["peers"], ["AgentB", "AgentC"])

    def test_peers_endpoint_includes_online_status_and_agent_card(self) -> None:
        agent_b_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / "AgentB"
        shutil.rmtree(agent_b_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_b_root, True)
        card_path = agent_b_root / "AgentServer" / "AgentCard.json"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(
            json.dumps(
                {
                    "agent_name": "AgentB",
                    "capabilities": [{"title": "Search", "content": "Find useful information."}],
                }
            ),
            encoding="utf-8",
        )
        self.client.put(
            "/admin/communication",
            json={"spaces": [{"id": "main", "members": ["AgentA", "AgentB"]}]},
        )
        self.client.post("/agents/register", json={"agent_name": "AgentA"})
        self.client.post(
            "/agents/register",
            json={
                "agent_name": "AgentB",
                "metadata": {"service_url": "http://127.0.0.1:8011"},
            },
        )
        self.client.post("/agents/AgentB/status", json={"status": "running", "phase": "ready"})

        response = self.client.get("/agents/peers/AgentA")

        self.assertEqual(response.status_code, 200)
        detail = response.json()["peer_details"][0]
        self.assertEqual(detail["agent_name"], "AgentB")
        self.assertTrue(detail["online"])
        self.assertEqual(detail["status"], "running")
        self.assertEqual(detail["service_url"], "http://127.0.0.1:8011")
        self.assertEqual(detail["card"]["capabilities"][0]["title"], "Search")

    def test_user_chat_mail_builds_agent_mail(self) -> None:
        self.client.put(
            "/user/chat/config/SeedAgent",
            json={
                "thread_id": "mail-thread",
                "run_id": "mail-run",
                "stream_mode": ["updates", "custom"],
                "version": "v2",
            },
        )
        response = self.client.post(
            "/user/chat",
            json={
                "agent_name": "SeedAgent",
                "mode": "mail",
                "from": "user",
                "text": "hello from user",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["mode"], "mail")

        response = self.client.get("/recv/SeedAgent")
        self.assertEqual(response.status_code, 200)
        messages = response.json()["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["from"], "user")
        self.assertEqual(messages[0]["to"], "SeedAgent")
        self.assertEqual(messages[0]["content"], "hello from user")
        self.assertEqual(messages[0]["attachments"], [])
        self.assertEqual(messages[0]["metadata"]["thread_id"], "mail-thread")
        self.assertEqual(messages[0]["metadata"]["run_id"], "mail-run")

    def test_user_chat_mail_uses_effective_default_run_id(self) -> None:
        self.client.put(
            "/user/chat/config/SeedAgent",
            json={
                "thread_id": "mail-thread",
                "run_id": None,
                "stream_mode": ["updates", "custom"],
                "version": "v2",
            },
        )
        response = self.client.post(
            "/user/chat",
            json={
                "agent_name": "SeedAgent",
                "mode": "mail",
                "from": "user",
                "text": "hello from user",
            },
        )

        self.assertEqual(response.status_code, 200)
        response = self.client.get("/recv/SeedAgent")
        messages = response.json()["messages"]
        self.assertEqual(messages[0]["metadata"]["thread_id"], "mail-thread")
        self.assertEqual(messages[0]["metadata"]["run_id"], "mail-thread-run")
        self.assertEqual(messages[0]["metadata"]["conversation_thread_id"], "mail-thread")
        self.assertEqual(messages[0]["metadata"]["conversation_run_id"], "mail-thread-run")

    def test_user_chat_mail_decodes_browser_data_url_attachments(self) -> None:
        agent_name = "MailUploadAgent"
        agent_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / agent_name
        shutil.rmtree(agent_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_root, True)

        response = self.client.post(
            "/user/chat",
            json={
                "agent_name": agent_name,
                "mode": "mail",
                "from": "user",
                "text": "see attached note",
                "attachments": [
                    {
                        "link": "data:text/plain;charset=utf-8,hello%20mail",
                        "name": "note.txt",
                        "summary": "note.txt",
                        "mime_type": "text/plain",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        response = self.client.get(f"/recv/{agent_name}")
        self.assertEqual(response.status_code, 200)
        attachment = response.json()["messages"][0]["attachments"][0]
        self.assertEqual(attachment["_routed"], "decoded")
        self.assertEqual(attachment["visible_link"], "/workspace/mail/user__" + response.json()["messages"][0]["message_id"] + "/note.txt")
        decoded_file = Path(attachment["link"])
        self.assertTrue(decoded_file.exists())
        self.assertEqual(decoded_file.read_text(encoding="utf-8"), "hello mail")
        message_md = decoded_file.parent / "message.md"
        self.assertIn("`/workspace/mail/", message_md.read_text(encoding="utf-8"))

    def test_user_chat_direct_proxies_invoke_payload_with_multimodal_user_message(self) -> None:
        calls = []

        def fake_post_json(url, payload, timeout):
            calls.append({"url": url, "payload": payload, "timeout": timeout})
            return {"ok": True, "chunks": []}

        original_post_json = self.main_server.post_json
        self.main_server.post_json = fake_post_json
        self.addCleanup(setattr, self.main_server, "post_json", original_post_json)

        self.client.post(
            "/agents/register",
            json={
                "agent_name": "SeedAgent",
                "metadata": {"service_url": "http://127.0.0.1:8010"},
            },
        )
        response = self.client.post(
            "/user/chat",
            json={
                "agent_name": "SeedAgent",
                "mode": "direct",
                "text": "describe this",
                "attachments": [
                    {
                        "link": "data:image/png;base64,abcd",
                        "summary": "inline image",
                        "mime_type": "image/png",
                    }
                ],
                "session_id": "chat-1",
                "run_id": "chat-1-run",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls[0]["url"], "http://127.0.0.1:8010/invoke")
        invoke_payload = calls[0]["payload"]
        self.assertEqual(invoke_payload["session_id"], "chat-1")
        self.assertEqual(invoke_payload["run_id"], "chat-1-run")
        user_message = invoke_payload["messages"][-1]
        self.assertEqual(user_message["role"], "user")
        self.assertIsInstance(user_message["content"], list)
        self.assertEqual(user_message["content"][-1]["type"], "image_url")
        self.assertEqual(response.json()["user_message"], user_message)

    def test_user_chat_config_supplies_default_thread_and_run_ids(self) -> None:
        calls = []

        def fake_post_json(url, payload, timeout):
            calls.append({"url": url, "payload": payload, "timeout": timeout})
            return {"ok": True, "chunks": []}

        original_post_json = self.main_server.post_json
        self.main_server.post_json = fake_post_json
        self.addCleanup(setattr, self.main_server, "post_json", original_post_json)

        self.client.post(
            "/agents/register",
            json={
                "agent_name": "SeedAgent",
                "metadata": {"service_url": "http://127.0.0.1:8010"},
            },
        )
        response = self.client.put(
            "/user/chat/config/SeedAgent",
            json={
                "thread_id": "frontend-thread",
                "run_id": "frontend-run",
                "stream_mode": ["updates", "custom"],
                "version": "v2",
            },
        )
        self.assertEqual(response.status_code, 200)

        response = self.client.post(
            "/user/chat",
            json={"agent_name": "SeedAgent", "mode": "direct", "text": "hello"},
        )

        self.assertEqual(response.status_code, 200)
        invoke_payload = calls[0]["payload"]
        self.assertEqual(invoke_payload["session_id"], "frontend-thread")
        self.assertEqual(invoke_payload["run_id"], "frontend-run")
        self.assertEqual(invoke_payload["stream_mode"], ["updates", "custom"])
        self.assertEqual(response.json()["chat"]["thread_id"], "frontend-thread")

    def test_user_chat_direct_uses_effective_default_run_id(self) -> None:
        calls = []

        def fake_post_json(url, payload, timeout):
            calls.append({"url": url, "payload": payload, "timeout": timeout})
            return {"ok": True, "chunks": []}

        original_post_json = self.main_server.post_json
        self.main_server.post_json = fake_post_json
        self.addCleanup(setattr, self.main_server, "post_json", original_post_json)

        self.client.post(
            "/agents/register",
            json={
                "agent_name": "SeedAgent",
                "metadata": {"service_url": "http://127.0.0.1:8010"},
            },
        )
        response = self.client.put(
            "/user/chat/config/SeedAgent",
            json={
                "thread_id": "frontend-thread",
                "run_id": None,
                "stream_mode": ["updates", "custom"],
                "version": "v2",
            },
        )
        self.assertEqual(response.status_code, 200)

        response = self.client.post(
            "/user/chat",
            json={"agent_name": "SeedAgent", "mode": "direct", "text": "hello"},
        )

        self.assertEqual(response.status_code, 200)
        invoke_payload = calls[0]["payload"]
        self.assertEqual(invoke_payload["session_id"], "frontend-thread")
        self.assertEqual(invoke_payload["run_id"], "frontend-thread-run")
        self.assertEqual(response.json()["chat"]["run_id"], "frontend-thread-run")

    def test_admin_clear_runtime_removes_store_and_mail_but_keeps_knowledge_by_default(self) -> None:
        agent_name = "RuntimeCleanupAgent"
        agent_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / agent_name
        shutil.rmtree(agent_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_root, True)

        store_file = agent_root / "Agent" / "store" / "memory" / "cache" / "chunk_cache.sqlite3"
        mail_file = agent_root / "workspace" / "mail" / "user__test" / "message.md"
        knowledge_file = agent_root / "workspace" / "knowledge" / "keep.txt"
        for path in (store_file, mail_file, knowledge_file):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data", encoding="utf-8")

        response = self.client.post(f"/admin/agents/{agent_name}/runtime/clear", json={})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(store_file.exists())
        self.assertFalse(mail_file.exists())
        self.assertTrue(knowledge_file.exists())
        self.assertTrue((agent_root / "Agent" / "store" / "memory" / "cache").is_dir())

    def test_admin_clear_runtime_can_remove_only_checkpoints(self) -> None:
        agent_name = "CheckpointCleanupAgent"
        agent_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / agent_name
        shutil.rmtree(agent_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_root, True)

        checkpoint_file = agent_root / "Agent" / "store" / "checkpoints" / "graph.sqlite"
        memory_checkpoint_file = agent_root / "Agent" / "store" / "memory" / "checkpoint" / "memory.sqlite"
        cache_file = agent_root / "Agent" / "store" / "memory" / "cache" / "chunk_cache.sqlite3"
        mail_file = agent_root / "workspace" / "mail" / "user__test" / "message.md"
        knowledge_file = agent_root / "workspace" / "knowledge" / "keep.txt"
        for path in (checkpoint_file, memory_checkpoint_file, cache_file, mail_file, knowledge_file):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data", encoding="utf-8")

        response = self.client.post(
            f"/admin/agents/{agent_name}/runtime/clear",
            json={
                "include_store": False,
                "include_mail": False,
                "include_knowledge": False,
                "include_checkpoints": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(checkpoint_file.exists())
        self.assertFalse(memory_checkpoint_file.exists())
        self.assertTrue(cache_file.exists())
        self.assertTrue(mail_file.exists())
        self.assertTrue(knowledge_file.exists())
        self.assertTrue(checkpoint_file.parent.is_dir())
        self.assertTrue(memory_checkpoint_file.parent.is_dir())
        self.assertFalse(response.json()["reload"]["attempted"])
        self.assertEqual(response.json()["reload"]["reason"], "agent_not_registered")

    def test_admin_clear_runtime_reloads_registered_agent_after_checkpoint_delete(self) -> None:
        agent_name = "CheckpointReloadAgent"
        agent_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / agent_name
        shutil.rmtree(agent_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_root, True)
        checkpoint_file = agent_root / "Agent" / "store" / "checkpoints" / "langgraph.sqlite3"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file.write_text("data", encoding="utf-8")
        calls = []

        def fake_post_json(url, payload, timeout):
            calls.append({"url": url, "payload": payload, "timeout": timeout})
            return {"ok": True}

        original_post_json = self.main_server.post_json
        self.main_server.post_json = fake_post_json
        self.addCleanup(setattr, self.main_server, "post_json", original_post_json)

        self.client.post(
            "/agents/register",
            json={"agent_name": agent_name, "metadata": {"service_url": "http://127.0.0.1:8898"}},
        )

        response = self.client.post(
            f"/admin/agents/{agent_name}/runtime/clear",
            json={
                "include_store": False,
                "include_mail": False,
                "include_knowledge": False,
                "include_checkpoints": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(checkpoint_file.exists())
        self.assertEqual(calls[0]["url"], "http://127.0.0.1:8898/reload-config")
        self.assertEqual(calls[0]["payload"], {})
        self.assertTrue(response.json()["reload"]["ok"])

    def test_admin_clear_runtime_clears_frontend_agent_history(self) -> None:
        agent_name = "HistoryCleanupAgent"
        agent_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / agent_name
        shutil.rmtree(agent_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_root, True)
        checkpoint_file = agent_root / "Agent" / "store" / "checkpoints" / "graph.sqlite"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file.write_text("data", encoding="utf-8")

        self.client.post("/agents/register", json={"agent_name": agent_name})
        self.client.post(
            f"/agents/{agent_name}/event",
            json={"event": "mail_wake_success", "reply": "done"},
        )
        self.client.post(
            "/send",
            json={
                "message_id": "history-1",
                "from": agent_name,
                "to": "OtherAgent",
                "type": "message",
                "content": "hello",
                "attachments": [],
            },
        )
        self.client.put(
            "/admin/ui-state",
            json={
                "agent_positions": {},
                "chat_sessions": {agent_name: {"messages": [{"role": "user", "content": "hello"}]}},
            },
        )

        response = self.client.post(
            f"/admin/agents/{agent_name}/runtime/clear",
            json={
                "include_store": False,
                "include_mail": False,
                "include_knowledge": False,
                "include_checkpoints": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(checkpoint_file.exists())
        monitor = self.client.get("/admin/monitor").json()
        agent = next(item for item in monitor["agents"] if item["agent_name"] == agent_name)
        self.assertEqual(agent["events"], [])
        self.assertEqual(monitor["recent_mail"], [])
        ui = self.client.get("/admin/ui-state").json()["ui"]
        self.assertNotIn(agent_name, ui["chat_sessions"])

    def test_runtime_config_endpoint_updates_runtime_fields(self) -> None:
        agent_name = "RuntimeConfigAgent"
        agent_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / agent_name
        config_file = agent_root / "Agent" / f"{agent_name}Config.example.json"
        shutil.rmtree(agent_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_root, True)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            '{"knowledgeRunId":"RuntimeConfigAgent-knowledge"}',
            encoding="utf-8",
        )

        response = self.client.patch(
            f"/admin/agents/{agent_name}/runtime-config",
            json={
                "knowledgeRunId": "RuntimeConfigAgent-knowledge-v2",
                "enableSendMessagesMiddleware": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["uses_local"])
        self.assertEqual(payload["config"]["knowledgeRunId"], "RuntimeConfigAgent-knowledge-v2")
        self.assertFalse(payload["config"]["enableSendMessagesMiddleware"])
        self.assertFalse(payload["reload"]["attempted"])
        self.assertEqual(payload["reload"]["reason"], "agent_not_registered")

    def test_agent_config_endpoint_merges_and_updates_runtime_fields(self) -> None:
        agent_name = "UnifiedConfigAgent"
        agent_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / agent_name
        config_file = agent_root / "Agent" / f"{agent_name}Config.example.json"
        shutil.rmtree(agent_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_root, True)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            '{"chatModel":"gpt-old","knowledgeRunId":"UnifiedConfigAgent-knowledge"}',
            encoding="utf-8",
        )

        response = self.client.get(f"/admin/agents/{agent_name}/config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["config"]["chatModel"], "gpt-old")

        response = self.client.put(
            f"/admin/agents/{agent_name}/config",
            json={
                "template": "SeedAgent",
                "scope": ["SeedAgent"],
                "chatModel": "gpt-new",
                "chatApiKey": "test-key",
                "knowledgeRunId": "UnifiedConfigAgent-knowledge-v2",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["center_config"], {"scope": ["SeedAgent"], "template": "SeedAgent"})
        self.assertEqual(payload["config"]["chatModel"], "gpt-new")
        self.assertEqual(payload["config"]["template"], "SeedAgent")
        self.assertFalse(payload["reload"]["attempted"])
        runtime = json.loads((agent_root / "Agent" / f"{agent_name}Config.local.json").read_text(encoding="utf-8"))
        self.assertEqual(runtime["chatModel"], "gpt-new")
        self.assertEqual(runtime["chatApiKey"], "test-key")
        stored_center = self.client.get(f"/admin/agents/{agent_name}/config").json()["center_config"]
        self.assertNotIn("chatModel", stored_center)

    def test_runtime_config_endpoint_reloads_registered_agent_service(self) -> None:
        agent_name = "RuntimeReloadAgent"
        agent_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / agent_name
        config_file = agent_root / "Agent" / f"{agent_name}Config.example.json"
        shutil.rmtree(agent_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_root, True)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            '{"knowledgeRunId":"RuntimeReloadAgent-knowledge"}',
            encoding="utf-8",
        )
        calls = []

        def fake_post_json(url, payload, timeout):
            calls.append({"url": url, "payload": payload, "timeout": timeout})
            return {"ok": True, "config": {"knowledgeRunId": "RuntimeReloadAgent-knowledge-v2"}}

        original_post_json = self.main_server.post_json
        self.main_server.post_json = fake_post_json
        self.addCleanup(setattr, self.main_server, "post_json", original_post_json)

        self.client.post(
            "/agents/register",
            json={"agent_name": agent_name, "metadata": {"service_url": "http://127.0.0.1:8899"}},
        )
        response = self.client.put(
            f"/admin/agents/{agent_name}/runtime-config",
            json={"knowledgeRunId": "RuntimeReloadAgent-knowledge-v2"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls[0]["url"], "http://127.0.0.1:8899/reload-config")
        self.assertEqual(calls[0]["payload"], {})
        self.assertTrue(response.json()["reload"]["ok"])

    def test_agent_card_endpoint_reads_and_writes_public_card(self) -> None:
        agent_name = "CardConfigAgent"
        agent_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / agent_name
        card_file = agent_root / "AgentServer" / "AgentCard.json"
        shutil.rmtree(agent_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_root, True)
        card_file.parent.mkdir(parents=True, exist_ok=True)
        card_file.write_text(
            '{"agent_name":"CardConfigAgent","capabilities":[{"title":"Old","content":"old public card"}]}',
            encoding="utf-8",
        )

        response = self.client.get(f"/admin/agents/{agent_name}/card")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["card"]["capabilities"][0]["title"], "Old")

        response = self.client.put(
            f"/admin/agents/{agent_name}/card",
            json={
                "agent_name": "WrongAgent",
                "capabilities": [{"title": "New", "content": "new public card"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["card"]["agent_name"], agent_name)
        self.assertEqual(payload["card"]["capabilities"][0]["content"], "new public card")
        self.assertEqual(payload["card"]["agent_name"], self.client.get(f"/admin/agents/{agent_name}/card").json()["card"]["agent_name"])

    def test_brain_prompt_endpoint_reads_and_writes_agents_md(self) -> None:
        agent_name = "BrainPromptAgent"
        agent_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / agent_name
        brain_file = agent_root / "workspace" / "brain" / "AGENTS.md"
        shutil.rmtree(agent_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, agent_root, True)
        brain_file.parent.mkdir(parents=True, exist_ok=True)
        brain_file.write_text("old brain\n", encoding="utf-8")

        response = self.client.get(f"/admin/agents/{agent_name}/brain")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"], "old brain\n")

        response = self.client.put(
            f"/admin/agents/{agent_name}/brain",
            json={"content": "new brain"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["content"], "new brain\n")
        self.assertEqual(brain_file.read_text(encoding="utf-8"), "new brain\n")

    def test_available_agents_includes_filesystem_seedagent_when_not_registered(self) -> None:
        response = self.client.get("/admin/agents/available")

        self.assertEqual(response.status_code, 200)
        names = {agent["agent_name"]: agent for agent in response.json()["agents"]}
        self.assertIn("SeedAgent", names)
        self.assertFalse(names["SeedAgent"]["registered"])
        self.assertEqual(names["SeedAgent"]["status"], "available")

    def test_create_agent_does_not_copy_workspace_business_files(self) -> None:
        source_name = "CopySourceAgent"
        target_name = "CopyTargetAgent"
        source_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / source_name
        target_root = Path(self.main_server.PROJECT_ROOT) / "Deepagents" / target_name
        shutil.rmtree(source_root, ignore_errors=True)
        shutil.rmtree(target_root, ignore_errors=True)
        self.addCleanup(shutil.rmtree, source_root, True)
        self.addCleanup(shutil.rmtree, target_root, True)

        (source_root / "Agent").mkdir(parents=True)
        (source_root / "AgentServer").mkdir(parents=True)
        (source_root / "workspace" / "brain").mkdir(parents=True)
        (source_root / "workspace" / "skills" / "custom-skill").mkdir(parents=True)
        (source_root / "workspace" / "notes").mkdir(parents=True)
        (source_root / "workspace" / "knowledge").mkdir(parents=True)
        (source_root / "workspace" / "brain" / "AGENTS.md").write_text("CopySourceAgent brain", encoding="utf-8")
        (source_root / "workspace" / "brain" / "scratch.txt").write_text("do not copy", encoding="utf-8")
        (source_root / "workspace" / "skills" / "custom-skill" / "SKILL.md").write_text("skill", encoding="utf-8")
        (source_root / "workspace" / "business.txt").write_text("do not copy", encoding="utf-8")
        (source_root / "workspace" / "notes" / "note.md").write_text("do not copy", encoding="utf-8")
        (source_root / "workspace" / "knowledge" / "doc.txt").write_text("do not copy", encoding="utf-8")

        response = self.client.post(
            "/admin/agents/create",
            json={"agent_name": target_name, "source_agent": source_name},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue((target_root / "workspace" / "brain" / "AGENTS.md").exists())
        self.assertIn(target_name, (target_root / "workspace" / "brain" / "AGENTS.md").read_text(encoding="utf-8"))
        self.assertTrue((target_root / "workspace" / "skills" / "custom-skill" / "SKILL.md").exists())
        self.assertFalse((target_root / "workspace" / "brain" / "scratch.txt").exists())
        self.assertFalse((target_root / "workspace" / "business.txt").exists())
        self.assertFalse((target_root / "workspace" / "notes" / "note.md").exists())
        self.assertFalse((target_root / "workspace" / "knowledge" / "doc.txt").exists())
        self.assertTrue((target_root / "workspace" / "notes").is_dir())
        self.assertTrue((target_root / "workspace" / "knowledge").is_dir())


if __name__ == "__main__":
    unittest.main()
