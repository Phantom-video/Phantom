# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Unit tests for OpenAICompatPromptExpander."""

import importlib.util
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Load prompt_extend module directly to avoid phantom_wan/__init__.py imports
# ---------------------------------------------------------------------------

# Stub only the direct dependencies of prompt_extend.py
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
LM_CH_SYS_PROMPT = prompt_extend.LM_CH_SYS_PROMPT
LM_EN_SYS_PROMPT = prompt_extend.LM_EN_SYS_PROMPT
VL_CH_SYS_PROMPT = prompt_extend.VL_CH_SYS_PROMPT
VL_EN_SYS_PROMPT = prompt_extend.VL_EN_SYS_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(content="expanded prompt"):
    """Build a fake openai chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_expander(is_vl=False, model_name=None, base_url=None):
    """Create an expander with a mocked OpenAI client."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
        with patch("openai.OpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat = MagicMock()
            instance.chat.completions = MagicMock()
            exp = OpenAICompatPromptExpander(
                model_name=model_name,
                base_url=base_url,
                is_vl=is_vl,
            )
            return exp, instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOpenAICompatPromptExpanderInit(unittest.TestCase):
    """Test initialization and configuration."""

    def test_default_model_and_url(self):
        exp, _ = _make_expander()
        self.assertEqual(exp.model, "MiniMax-M2.7")

    def test_custom_model_name(self):
        exp, _ = _make_expander(model_name="gpt-4o")
        self.assertEqual(exp.model, "gpt-4o")

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            for key in ("MINIMAX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
                os.environ.pop(key, None)
            with self.assertRaises(ValueError):
                with patch("openai.OpenAI"):
                    OpenAICompatPromptExpander()

    def test_minimax_api_key_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with patch("openai.OpenAI"):
                exp = OpenAICompatPromptExpander()
                self.assertIsNotNone(exp.client)

    def test_openai_api_key_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "oai-key"}, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            with patch("openai.OpenAI"):
                exp = OpenAICompatPromptExpander()
                self.assertIsNotNone(exp.client)

    def test_is_vl_flag(self):
        exp, _ = _make_expander(is_vl=True)
        self.assertTrue(exp.is_vl)

    def test_retry_times_default(self):
        exp, _ = _make_expander()
        self.assertEqual(exp.retry_times, 4)


class TestStripThinkTags(unittest.TestCase):
    """Test the _strip_think_tags helper."""

    def setUp(self):
        self.exp, _ = _make_expander()

    def test_no_think_tags(self):
        self.assertEqual(self.exp._strip_think_tags("hello world"), "hello world")

    def test_single_think_block(self):
        text = "<think>some reasoning</think>the answer"
        self.assertEqual(self.exp._strip_think_tags(text), "the answer")

    def test_multiline_think_block(self):
        text = "<think>\nline1\nline2\n</think>\nresult"
        self.assertEqual(self.exp._strip_think_tags(text), "result")

    def test_multiple_think_blocks(self):
        text = "<think>a</think>middle<think>b</think>end"
        self.assertEqual(self.exp._strip_think_tags(text), "middleend")


class TestExtend(unittest.TestCase):
    """Test text-only prompt extension."""

    def test_successful_extend(self):
        exp, client = _make_expander()
        client.chat.completions.create.return_value = _make_mock_response(
            "expanded prompt"
        )
        result = exp("test prompt", tar_lang="en")
        self.assertIsInstance(result, PromptOutput)
        self.assertTrue(result.status)
        self.assertEqual(result.prompt, "expanded prompt")

    def test_extend_chinese(self):
        exp, client = _make_expander()
        client.chat.completions.create.return_value = _make_mock_response(
            "扩展后的提示词"
        )
        result = exp("测试提示词", tar_lang="ch")
        self.assertTrue(result.status)
        self.assertEqual(result.prompt, "扩展后的提示词")

    def test_extend_strips_think_tags(self):
        exp, client = _make_expander()
        client.chat.completions.create.return_value = _make_mock_response(
            "<think>reasoning</think>clean prompt"
        )
        result = exp("test", tar_lang="en")
        self.assertTrue(result.status)
        self.assertEqual(result.prompt, "clean prompt")

    def test_extend_failure_returns_original(self):
        exp, client = _make_expander()
        client.chat.completions.create.side_effect = RuntimeError("API error")
        exp.retry_times = 2
        result = exp("original prompt", tar_lang="en")
        self.assertFalse(result.status)
        self.assertEqual(result.prompt, "original prompt")
        self.assertIn("API error", result.message)

    def test_extend_retries_on_failure(self):
        exp, client = _make_expander()
        exp.retry_times = 3
        client.chat.completions.create.side_effect = [
            RuntimeError("fail1"),
            RuntimeError("fail2"),
            _make_mock_response("success"),
        ]
        result = exp("test", tar_lang="en")
        self.assertTrue(result.status)
        self.assertEqual(result.prompt, "success")
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_extend_uses_correct_system_prompt_en(self):
        exp, client = _make_expander()
        client.chat.completions.create.return_value = _make_mock_response("ok")
        exp("test", tar_lang="en")
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        self.assertEqual(messages[0]["content"], LM_EN_SYS_PROMPT)

    def test_extend_uses_correct_system_prompt_ch(self):
        exp, client = _make_expander()
        client.chat.completions.create.return_value = _make_mock_response("ok")
        exp("test", tar_lang="ch")
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        self.assertEqual(messages[0]["content"], LM_CH_SYS_PROMPT)

    def test_extend_uses_model_name(self):
        exp, client = _make_expander(model_name="MiniMax-M2.5")
        client.chat.completions.create.return_value = _make_mock_response("ok")
        exp("test", tar_lang="en")
        call_args = client.chat.completions.create.call_args
        model = call_args.kwargs.get("model", call_args[1].get("model"))
        self.assertEqual(model, "MiniMax-M2.5")

    def test_extend_message_contains_json(self):
        exp, client = _make_expander()
        client.chat.completions.create.return_value = _make_mock_response("out")
        result = exp("test", tar_lang="en")
        msg = json.loads(result.message)
        self.assertEqual(msg["content"], "out")
        self.assertEqual(msg["model"], "MiniMax-M2.7")

    def test_extend_sets_temperature(self):
        exp, client = _make_expander()
        client.chat.completions.create.return_value = _make_mock_response("ok")
        exp("test", tar_lang="en")
        call_args = client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs.get("temperature"), 0.7)


class TestExtendWithImg(unittest.TestCase):
    """Test image+text prompt extension."""

    def _make_test_image(self):
        import io
        img = MagicMock()
        img.width = 100
        img.height = 100
        img.convert.return_value = img

        def resize_fn(size):
            resized = MagicMock()
            resized.width = size[0]
            resized.height = size[1]
            buf_content = b'\x89PNG\r\n\x1a\n' + b'\x00' * 50
            resized.save = MagicMock(
                side_effect=lambda b, format: b.write(buf_content)
            )
            return resized
        img.resize = resize_fn
        return img

    def test_extend_with_img_success(self):
        exp, client = _make_expander(is_vl=True)
        client.chat.completions.create.return_value = _make_mock_response(
            "image-based expansion"
        )
        result = exp("describe this", tar_lang="en", image=self._make_test_image())
        self.assertTrue(result.status)
        self.assertIn("image-based expansion", result.prompt)

    def test_extend_with_img_sends_base64(self):
        exp, client = _make_expander(is_vl=True)
        client.chat.completions.create.return_value = _make_mock_response("ok")
        exp("test", tar_lang="en", image=self._make_test_image())
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        user_content = messages[1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(len(user_content), 2)
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(
            user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        )

    def test_extend_with_img_failure(self):
        exp, client = _make_expander(is_vl=True)
        client.chat.completions.create.side_effect = RuntimeError("VL error")
        exp.retry_times = 1
        result = exp("test", tar_lang="en", image=self._make_test_image())
        self.assertFalse(result.status)
        self.assertIn("VL error", result.message)

    def test_extend_with_img_uses_vl_system_prompt(self):
        exp, client = _make_expander(is_vl=True)
        client.chat.completions.create.return_value = _make_mock_response("ok")
        exp("test", tar_lang="ch", image=self._make_test_image())
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages"))
        self.assertEqual(messages[0]["content"], VL_CH_SYS_PROMPT)


class TestDecideSystemPrompt(unittest.TestCase):
    """Test system prompt selection logic."""

    def test_lm_ch(self):
        exp, _ = _make_expander(is_vl=False)
        self.assertEqual(exp.decide_system_prompt("ch"), LM_CH_SYS_PROMPT)

    def test_lm_en(self):
        exp, _ = _make_expander(is_vl=False)
        self.assertEqual(exp.decide_system_prompt("en"), LM_EN_SYS_PROMPT)

    def test_vl_ch(self):
        exp, _ = _make_expander(is_vl=True)
        self.assertEqual(exp.decide_system_prompt("ch"), VL_CH_SYS_PROMPT)

    def test_vl_en(self):
        exp, _ = _make_expander(is_vl=True)
        self.assertEqual(exp.decide_system_prompt("en"), VL_EN_SYS_PROMPT)


if __name__ == "__main__":
    unittest.main()
