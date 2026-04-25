from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class ImportSideEffectTests(unittest.TestCase):
    def test_public_imports_do_not_open_neo4j_drivers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        code = textwrap.dedent(
            """
            from unittest.mock import patch

            with patch("neo4j.GraphDatabase.driver", side_effect=AssertionError("driver opened during import")):
                import tools
                import middleware
                from tools import ChunkApplyToolConfig
                from middleware import KnowledgeManagerMiddlewareConfig
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr[-1000:])


if __name__ == "__main__":
    unittest.main()
