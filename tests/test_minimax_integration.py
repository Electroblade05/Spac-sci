"""Integration tests for MiniMax LLM provider.

These tests require a valid MINIMAX_API_KEY environment variable.
Skip automatically when the key is not set.
"""

import os
import json
import re
import unittest

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
SKIP_REASON = "MINIMAX_API_KEY not set"


def strip_think_tags(text):
    """Strip <think>...</think> tags from MiniMax M2.7 responses."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def extract_json(text):
    """Extract JSON from response, handling think tags and markdown fences."""
    text = strip_think_tags(text)
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    json_start = text.find("{")
    json_end = text.rfind("}")
    if json_start >= 0 and json_end >= 0:
        text = text[json_start : json_end + 1]
    return json.loads(text)


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that hit the real MiniMax API."""

    def test_chat_completions_basic(self):
        """MiniMax should respond to a basic chat completion request."""
        from openai import OpenAI

        client = OpenAI(
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
        )
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "user", "content": "Say hello in one word."}
            ],
            max_tokens=200,
        )
        self.assertTrue(len(response.choices) > 0)
        content = strip_think_tags(response.choices[0].message.content)
        self.assertTrue(len(content) > 0)

    def test_json_output(self):
        """MiniMax should return valid JSON when prompted."""
        from openai import OpenAI

        client = OpenAI(
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
        )
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "system", "content": "Output valid JSON only. No markdown. No thinking."},
                {"role": "user", "content": 'Return {"greeting": "hello"}'},
            ],
            max_tokens=50,
        )
        content = response.choices[0].message.content.strip()
        parsed = extract_json(content)
        self.assertIn("greeting", parsed)

    def test_script_generation_format(self):
        """MiniMax should generate a script in the expected JSON format."""
        from openai import OpenAI

        client = OpenAI(
            api_key=MINIMAX_API_KEY,
            base_url="https://api.minimax.io/v1",
        )
        prompt = (
            "You are a content writer. Generate a very short 2-sentence fact. "
            'Output ONLY a JSON object: {"script": "your text here"}'
        )
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "space facts"},
            ],
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        parsed = extract_json(content)
        self.assertIn("script", parsed)
        self.assertTrue(len(parsed["script"]) > 10)


if __name__ == "__main__":
    unittest.main()
