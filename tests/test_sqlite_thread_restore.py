import sqlite3
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph


def build_memory_graph(checkpointer: SqliteSaver):
    def respond(state: MessagesState) -> dict[str, list[AIMessage]]:
        humans = [message for message in state["messages"] if isinstance(message, HumanMessage)]
        latest = humans[-1].content if humans else ""
        if "刚才" in latest or "what was" in latest.lower():
            remembered = humans[0].content if humans else ""
            return {"messages": [AIMessage(content=f"remembered: {remembered}")]}
        return {"messages": [AIMessage(content="stored")]}

    graph = StateGraph(MessagesState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)


class SQLiteThreadRestoreTests(unittest.TestCase):
    def test_same_thread_id_restores_previous_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_path = Path(tempdir) / "checkpoints.sqlite3"
            connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
            try:
                checkpointer = SqliteSaver(connection)
                graph = build_memory_graph(checkpointer)
                config = {"configurable": {"thread_id": "thread-a"}}

                first = graph.invoke({"messages": [HumanMessage(content="secret=blue-cup")]}, config=config)
                second = graph.invoke({"messages": [HumanMessage(content="刚才我说的 secret 是什么？")]}, config=config)

                self.assertEqual(first["messages"][-1].content, "stored")
                self.assertEqual(second["messages"][-1].content, "remembered: secret=blue-cup")
                self.assertEqual(len(second["messages"]), 4)
            finally:
                connection.close()

    def test_different_thread_id_does_not_restore_previous_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_path = Path(tempdir) / "checkpoints.sqlite3"
            connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
            try:
                checkpointer = SqliteSaver(connection)
                graph = build_memory_graph(checkpointer)

                graph.invoke(
                    {"messages": [HumanMessage(content="secret=blue-cup")]},
                    config={"configurable": {"thread_id": "thread-a"}},
                )
                second = graph.invoke(
                    {"messages": [HumanMessage(content="刚才我说的 secret 是什么？")]},
                    config={"configurable": {"thread_id": "thread-b"}},
                )

                self.assertEqual(second["messages"][-1].content, "remembered: 刚才我说的 secret 是什么？")
                self.assertEqual(len(second["messages"]), 2)
            finally:
                connection.close()

    def test_same_thread_id_restores_even_when_run_id_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_path = Path(tempdir) / "checkpoints.sqlite3"
            connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
            try:
                checkpointer = SqliteSaver(connection)
                graph = build_memory_graph(checkpointer)

                graph.invoke(
                    {"messages": [HumanMessage(content="secret=blue-cup")]},
                    config={"configurable": {"thread_id": "thread-a", "run_id": "run-1"}},
                )
                second = graph.invoke(
                    {"messages": [HumanMessage(content="刚才我说的 secret 是什么？")]},
                    config={"configurable": {"thread_id": "thread-a", "run_id": "run-2"}},
                )

                self.assertEqual(second["messages"][-1].content, "remembered: secret=blue-cup")
                self.assertEqual(len(second["messages"]), 4)
            finally:
                connection.close()

    def test_langvideo_run_and_thread_combine_into_checkpoint_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint_path = Path(tempdir) / "checkpoints.sqlite3"
            connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
            try:
                checkpointer = SqliteSaver(connection)
                graph = build_memory_graph(checkpointer)

                def config(run_id: str, thread_id: str) -> dict:
                    return {"configurable": {"thread_id": f"{run_id}:{thread_id}"}}

                graph.invoke(
                    {"messages": [HumanMessage(content="secret=blue-cup")]},
                    config=config("run-a", "chat"),
                )
                restored = graph.invoke(
                    {"messages": [HumanMessage(content="刚才我说的 secret 是什么？")]},
                    config=config("run-a", "chat"),
                )
                isolated = graph.invoke(
                    {"messages": [HumanMessage(content="刚才我说的 secret 是什么？")]},
                    config=config("run-b", "chat"),
                )

                self.assertEqual(restored["messages"][-1].content, "remembered: secret=blue-cup")
                self.assertEqual(isolated["messages"][-1].content, "remembered: 刚才我说的 secret 是什么？")
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
