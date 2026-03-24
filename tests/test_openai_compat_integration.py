# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Integration tests for OpenAICompatPromptExpander with MiniMax API.

These tests require a live MINIMAX_API_KEY and network access.
Skip automatically when the key is not set.
"""

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock

# Stub heavy dependencies to avoid needing torch etc.
for mod_name in ("dashscope", "torch"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

_PE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "phantom_wan", "utils", "prompt_extend.py"
)
_spec = importlib.util.spec_from_file_location("prompt_extend", _PE_PATH)
prompt_extend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prompt_extend)

OpenAICompatPromptExpander = prompt_extend.OpenAICompatPromptExpander
PromptOutput = prompt_extend.PromptOutput

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestOpenAICompatIntegration(unittest.TestCase):
    """Live integration tests against MiniMax API."""

    def setUp(self):
        self.expander = OpenAICompatPromptExpander(
            model_name="MiniMax-M2.7",
            retry_times=2,
        )

    def test_extend_english_prompt(self):
        result = self.expander("A cat sitting on a surfboard", tar_lang="en")
        self.assertTrue(result.status, f"API call failed: {result.message}")
        self.assertGreater(len(result.prompt), 20)

    def test_extend_chinese_prompt(self):
        result = self.expander("一只猫坐在冲浪板上", tar_lang="ch")
        self.assertTrue(result.status, f"API call failed: {result.message}")
        self.assertGreater(len(result.prompt), 10)

    def test_extend_returns_prompt_output(self):
        result = self.expander("sunset on the beach", tar_lang="en")
        self.assertIsInstance(result, PromptOutput)
        self.assertTrue(result.status)
        self.assertIsInstance(result.prompt, str)
        self.assertIsInstance(result.seed, int)
        self.assertIsInstance(result.message, str)


if __name__ == "__main__":
    unittest.main()
