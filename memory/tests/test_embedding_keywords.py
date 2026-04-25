import unittest
from unittest.mock import patch

from server.embedding_keywords import embed_keywords
from server.neo4j.chunk_store_writes import create_schema


class EmbeddingKeywordTests(unittest.TestCase):
    def test_embed_keywords_uses_langchain_init_embeddings(self) -> None:
        captured: dict[str, object] = {}

        class FakeEmbeddings:
            def embed_documents(self, texts):
                captured["texts"] = list(texts)
                return [[0.1, 0.2] for _ in texts]

        def fake_init_embeddings(model, *, provider=None, **kwargs):
            captured["model"] = model
            captured["provider"] = provider
            captured["kwargs"] = kwargs
            return FakeEmbeddings()

        with patch("server.embedding_keywords.init_embeddings", side_effect=fake_init_embeddings):
            vectors = embed_keywords(
                ["alpha", "beta"],
                salt="doc:",
                config_override={
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "sk-test",
                    "dimensions": 64,
                },
            )

        self.assertEqual(captured["model"], "text-embedding-3-small")
        self.assertEqual(captured["provider"], "openai")
        self.assertEqual(captured["kwargs"]["base_url"], "https://example.invalid/v1")
        self.assertEqual(captured["kwargs"]["dimensions"], 64)
        self.assertEqual(captured["texts"], ["doc:alpha", "doc:beta"])
        self.assertEqual(vectors, [[0.1, 0.2], [0.1, 0.2]])

    def test_chunk_schema_uses_override_index_name_and_dimensions(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        class FakeTransaction:
            def run(self, statement, **kwargs):
                calls.append((statement, kwargs))

        override = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "base_url": "https://example.invalid/v1",
            "api_key": "sk-test",
            "dimensions": 256,
        }

        create_schema(FakeTransaction(), statements=[], embedding_config_override=override)

        vector_statement, vector_kwargs = calls[-1]
        self.assertIn("chunk_keyword_embedding_index__openai__text_embedding_3_small__256", vector_statement)
        self.assertEqual(vector_kwargs["dimensions"], 256)


if __name__ == "__main__":
    unittest.main()
