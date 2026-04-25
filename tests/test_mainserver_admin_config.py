import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
