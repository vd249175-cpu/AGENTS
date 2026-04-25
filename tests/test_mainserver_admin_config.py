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
        self.client = TestClient(main_server.app)

    def tearDown(self) -> None:
        os.environ.pop("MAIN_SERVER_AGENT_CONFIG", None)

    def test_nested_scope_is_resolved_by_mainserver_config(self) -> None:
        response = self.client.put(
            "/admin/agents/SeedAgent/scope",
            json={"scope": [[], "WorkerAgent", ["ReviewAgent", []]]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["resolved_scope"], ["WorkerAgent", "ReviewAgent"])

        for agent_name in ("SeedAgent", "WorkerAgent", "ReviewAgent", "OtherAgent"):
            response = self.client.post("/agents/register", json={"agent_name": agent_name})
            self.assertEqual(response.status_code, 200)

        response = self.client.get("/agents/peers/SeedAgent")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["peers"], ["WorkerAgent", "ReviewAgent"])

    def test_send_and_recv_use_nested_scope(self) -> None:
        self.client.put("/admin/agents/SeedAgent/scope", json={"scope": [[], "WorkerAgent", []]})
        self.client.put("/admin/agents/WorkerAgent/scope", json={"scope": [[], "SeedAgent", []]})

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

    def test_registration_scope_still_works_without_admin_config(self) -> None:
        self.client.post(
            "/agents/register",
            json={"agent_name": "AgentA", "scope": [[], "AgentB", []]},
        )
        self.client.post("/agents/register", json={"agent_name": "AgentB"})
        self.client.post("/agents/register", json={"agent_name": "AgentC"})

        response = self.client.get("/agents/peers/AgentA")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["peers"], ["AgentB"])

    def test_user_chat_mail_builds_agent_mail(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
