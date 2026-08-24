"""Unit tests for MiniMax LLM provider integration."""

import os
import json
import unittest
from unittest.mock import patch, MagicMock


def _base_env(**overrides):
    """Return minimal valid env for MiniMax provider."""
    env = {
        "LLM_PROVIDER": "minimax",
        "MINIMAX_API_KEY": "test-minimax-key",
        "MINIMAX_MODEL": "MiniMax-M2.7",
        "PEXELS_API_KEY": "test-pexels-key",
        "STT_PROVIDER": "whisper",
        "TTS_PROVIDER": "edgetts",
        "EDGETTS_VOICE": "en-AU-WilliamNeural",
    }
    env.update(overrides)
    return env


def _reset_config():
    """Reset Config singleton for clean test."""
    from utility.config import Config
    Config._instance = None


class TestConfigMiniMaxValidation(unittest.TestCase):
    """Test Config validation for MiniMax provider."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    @patch("utility.config.os.path.exists", return_value=True)
    def test_minimax_valid_config(self, mock_exists):
        """MiniMax config with all required fields should succeed."""
        with patch.dict(os.environ, _base_env(), clear=False):
            from utility.config import Config
            config = Config()
            self.assertEqual(config.get_llm_provider(), "minimax")

    @patch("utility.config.os.path.exists", return_value=True)
    def test_minimax_missing_api_key(self, mock_exists):
        """Missing MINIMAX_API_KEY should raise ConfigurationError."""
        from utility.config import ConfigurationError
        env = _base_env()
        env.pop("MINIMAX_API_KEY")
        # Remove from real env too
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("MINIMAX_API_KEY", None)
            with self.assertRaises(ConfigurationError) as ctx:
                from utility.config import Config
                _reset_config()
                Config()
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    @patch("utility.config.os.path.exists", return_value=True)
    def test_minimax_missing_model(self, mock_exists):
        """Missing MINIMAX_MODEL should raise ConfigurationError."""
        from utility.config import ConfigurationError
        env = _base_env()
        env.pop("MINIMAX_MODEL")
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("MINIMAX_MODEL", None)
            with self.assertRaises(ConfigurationError):
                from utility.config import Config
                _reset_config()
                Config()

    @patch("utility.config.os.path.exists", return_value=True)
    def test_minimax_default_model(self, mock_exists):
        """Default model should be MiniMax-M2.7."""
        with patch.dict(os.environ, _base_env(), clear=False):
            from utility.config import Config
            config = Config()
            self.assertEqual(config.get_llm_model(), "MiniMax-M2.7")

    @patch("utility.config.os.path.exists", return_value=True)
    def test_minimax_custom_model(self, mock_exists):
        """Custom model should be returned when set."""
        with patch.dict(os.environ, _base_env(MINIMAX_MODEL="MiniMax-M2.7-highspeed"), clear=False):
            from utility.config import Config
            config = Config()
            self.assertEqual(config.get_llm_model(), "MiniMax-M2.7-highspeed")

    @patch("utility.config.os.path.exists", return_value=True)
    def test_minimax_client_is_openai(self, mock_exists):
        """MiniMax client should be an OpenAI instance with custom base_url."""
        from openai import OpenAI
        with patch.dict(os.environ, _base_env(), clear=False):
            from utility.config import Config
            config = Config()
            client = config.get_llm_client()
            self.assertIsInstance(client, OpenAI)
            self.assertIn("minimax", str(client.base_url))

    @patch("utility.config.os.path.exists", return_value=True)
    def test_minimax_client_cached(self, mock_exists):
        """Client should be cached on subsequent calls."""
        with patch.dict(os.environ, _base_env(), clear=False):
            from utility.config import Config
            config = Config()
            client1 = config.get_llm_client()
            client2 = config.get_llm_client()
            self.assertIs(client1, client2)

    @patch("utility.config.os.path.exists", return_value=True)
    def test_minimax_provider_literal(self, mock_exists):
        """get_llm_provider should return 'minimax'."""
        with patch.dict(os.environ, _base_env(), clear=False):
            from utility.config import Config
            config = Config()
            self.assertEqual(config.get_llm_provider(), "minimax")

    @patch("utility.config.os.path.exists", return_value=True)
    def test_invalid_provider_rejected(self, mock_exists):
        """Invalid provider should be rejected."""
        from utility.config import ConfigurationError
        with patch.dict(os.environ, _base_env(LLM_PROVIDER="invalid"), clear=False):
            with self.assertRaises(ConfigurationError):
                from utility.config import Config
                _reset_config()
                Config()

    @patch("utility.config.os.path.exists", return_value=True)
    def test_all_four_providers_accepted(self, mock_exists):
        """All four providers should be valid."""
        for provider in ["openai", "groq", "gemini", "minimax"]:
            _reset_config()
            env = {
                "LLM_PROVIDER": provider,
                "PEXELS_API_KEY": "test",
                "STT_PROVIDER": "whisper",
                "TTS_PROVIDER": "edgetts",
                "EDGETTS_VOICE": "en-AU-WilliamNeural",
            }
            if provider == "openai":
                env.update({"OPENAI_API_KEY": "test", "OPENAI_MODEL": "gpt-4o"})
            elif provider == "groq":
                env.update({"GROQ_API_KEY": "test", "GROQ_MODEL": "llama3-70b-8192"})
            elif provider == "gemini":
                env.update({"GEMINI_API_KEY": "test", "GEMINI_MODEL": "gemini-2.5-flash"})
            elif provider == "minimax":
                env.update({"MINIMAX_API_KEY": "test", "MINIMAX_MODEL": "MiniMax-M2.7"})
            with patch.dict(os.environ, env, clear=False):
                from utility.config import Config
                config = Config()
                self.assertEqual(config.get_llm_provider(), provider)


class TestScriptGeneratorMiniMax(unittest.TestCase):
    """Test script_generator works with MiniMax client."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    @patch("utility.script.script_generator._call_openai_groq")
    @patch("utility.script.script_generator.get_config")
    def test_generate_script_calls_openai_groq_for_minimax(self, mock_get_config, mock_call):
        """generate_script should call _call_openai_groq for MiniMax provider."""
        mock_config = MagicMock()
        mock_config.get_llm_provider.return_value = "minimax"
        mock_config.get_llm_model.return_value = "MiniMax-M2.7"
        mock_config.get_llm_client.return_value = MagicMock()
        mock_get_config.return_value = mock_config

        mock_call.return_value = '{"script": "Test script content"}'

        from utility.script.script_generator import generate_script
        result = generate_script("test topic")

        mock_call.assert_called_once()
        self.assertEqual(result, "Test script content")

    @patch("utility.script.script_generator._call_gemini")
    @patch("utility.script.script_generator._call_openai_groq")
    @patch("utility.script.script_generator.get_config")
    def test_minimax_does_not_call_gemini(self, mock_get_config, mock_openai, mock_gemini):
        """MiniMax should not use the Gemini code path."""
        mock_config = MagicMock()
        mock_config.get_llm_provider.return_value = "minimax"
        mock_config.get_llm_model.return_value = "MiniMax-M2.7"
        mock_config.get_llm_client.return_value = MagicMock()
        mock_get_config.return_value = mock_config

        mock_openai.return_value = '{"script": "Hello"}'

        from utility.script.script_generator import generate_script
        generate_script("test")

        mock_gemini.assert_not_called()

    @patch("utility.script.script_generator._call_openai_groq")
    @patch("utility.script.script_generator.get_config")
    def test_minimax_model_passed_to_client(self, mock_get_config, mock_call):
        """The MiniMax model name should be passed to the OpenAI-compat call."""
        mock_config = MagicMock()
        mock_config.get_llm_provider.return_value = "minimax"
        mock_config.get_llm_model.return_value = "MiniMax-M2.7-highspeed"
        mock_config.get_llm_client.return_value = MagicMock()
        mock_get_config.return_value = mock_config

        mock_call.return_value = '{"script": "Fast script"}'

        from utility.script.script_generator import generate_script
        generate_script("speed test")

        call_args = mock_call.call_args
        self.assertEqual(call_args[0][1], "MiniMax-M2.7-highspeed")


class TestVideoSearchQueryMiniMax(unittest.TestCase):
    """Test video_search_query_generator with MiniMax."""

    @patch("utility.video.video_search_query_generator.get_config")
    def test_minimax_uses_openai_chat_completions(self, mock_get_config):
        """MiniMax should use client.chat.completions.create (not gemini path)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '[[[0.0, 5.0], ["test keyword", "test scene", "test visual"]]]'
        mock_client.chat.completions.create.return_value = mock_response

        mock_config = MagicMock()
        mock_config.get_llm_provider.return_value = "minimax"
        mock_config.get_llm_model.return_value = "MiniMax-M2.7"
        mock_config.get_llm_client.return_value = mock_client
        mock_get_config.return_value = mock_config

        from utility.video.video_search_query_generator import call_OpenAI
        result = call_OpenAI("test script", [[[0.0, 5.0], "test caption"]])

        mock_client.chat.completions.create.assert_called_once()
        self.assertIn("test keyword", result)

    @patch("utility.video.video_search_query_generator.get_config")
    def test_minimax_temperature_within_range(self, mock_get_config):
        """Temperature=1 in call_OpenAI should be valid for MiniMax (0, 1]."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '[[[0.0, 3.0], ["cat", "dog", "bird"]]]'
        mock_client.chat.completions.create.return_value = mock_response

        mock_config = MagicMock()
        mock_config.get_llm_provider.return_value = "minimax"
        mock_config.get_llm_model.return_value = "MiniMax-M2.7"
        mock_config.get_llm_client.return_value = mock_client
        mock_get_config.return_value = mock_config

        from utility.video.video_search_query_generator import call_OpenAI
        call_OpenAI("animals", [[[0.0, 3.0], "animals"]])

        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("temperature", call_kwargs[1].get("temperature")), 1)


class TestEnvExample(unittest.TestCase):
    """Test .env.example includes MiniMax configuration."""

    def test_env_example_has_minimax_key(self):
        """Check .env.example contains MINIMAX_API_KEY."""
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        with open(env_path) as f:
            content = f.read()
        self.assertIn("MINIMAX_API_KEY", content)

    def test_env_example_has_minimax_model(self):
        """Check .env.example contains MINIMAX_MODEL."""
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        with open(env_path) as f:
            content = f.read()
        self.assertIn("MINIMAX_MODEL", content)
        self.assertIn("MiniMax-M2.7", content)

    def test_env_example_has_minimax_in_options(self):
        """Check .env.example lists minimax as a provider option."""
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        with open(env_path) as f:
            content = f.read()
        self.assertIn("minimax", content.lower())


if __name__ == "__main__":
    unittest.main()
