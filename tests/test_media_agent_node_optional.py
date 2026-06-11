import asyncio
import importlib
import unittest
from unittest.mock import patch

module = importlib.import_module("AgentCoordinator.graph.nodes.media_agent_node")


class MediaAgentNodeOptionalTests(unittest.TestCase):
    def test_skips_quietly_when_unconfigured(self):
        with patch.object(module, "_missing_media_config", return_value=["MEDIA_ENGINE_API_KEY"]):
            result = asyncio.run(module.media_agent_node({"query": "test topic"}))

        media_run = result["media_run"]
        self.assertTrue(media_run["success"])
        self.assertIsNone(media_run["text_output"])
        self.assertIsNone(media_run["error"])
        self.assertIn("skipped_unconfigured", result["coordinator_trace"][0])

    def test_uses_live_media_output_when_available(self):
        async def fake_run(_query):
            return "# Live Media Report\n\n- Real media evidence"

        with patch.object(module, "_missing_media_config", return_value=[]), patch.object(module, "_run_media_agent", fake_run):
            result = asyncio.run(module.media_agent_node({"query": "test topic"}))

        media_run = result["media_run"]
        self.assertTrue(media_run["success"])
        self.assertEqual(media_run["text_output"], "# Live Media Report\n\n- Real media evidence")
        self.assertIn("live", result["coordinator_trace"][0])


if __name__ == "__main__":
    unittest.main()
