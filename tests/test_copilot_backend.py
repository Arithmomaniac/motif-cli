"""Tests for the Copilot SDK backend JSON parsing."""

import pytest
from motif.llm.copilot_backend import _parse_json_response


class TestParseJsonResponse:
    """Tests for _parse_json_response()."""

    def test_raw_json(self):
        text = '{"archetype": {"name": "The Architect"}, "superpowers": []}'
        result = _parse_json_response(text)
        assert result["archetype"]["name"] == "The Architect"

    def test_raw_json_with_whitespace(self):
        text = '  \n  {"key": "value"}  \n  '
        result = _parse_json_response(text)
        assert result["key"] == "value"

    def test_json_in_code_fence(self):
        text = 'Here is the analysis:\n```json\n{"skills": [{"name": "deploy"}]}\n```\nDone.'
        result = _parse_json_response(text)
        assert result["skills"][0]["name"] == "deploy"

    def test_json_in_plain_fence(self):
        text = '```\n{"rules": []}\n```'
        result = _parse_json_response(text)
        assert result["rules"] == []

    def test_json_with_trailing_text(self):
        text = '{"communication_style": "terse"}\n\nLet me know if you need changes.'
        result = _parse_json_response(text)
        assert result["communication_style"] == "terse"

    def test_json_with_leading_text(self):
        text = 'Based on my analysis:\n\n{"result": true}'
        result = _parse_json_response(text)
        assert result["result"] is True

    def test_nested_json(self):
        text = '```json\n{"a": {"b": {"c": [1, 2, 3]}}}\n```'
        result = _parse_json_response(text)
        assert result["a"]["b"]["c"] == [1, 2, 3]

    def test_multiline_json_in_fence(self):
        text = '```json\n{\n  "name": "test",\n  "items": [\n    "one",\n    "two"\n  ]\n}\n```'
        result = _parse_json_response(text)
        assert result["name"] == "test"
        assert len(result["items"]) == 2

    def test_malformed_json_raises(self):
        text = "This is not JSON at all, just regular text."
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            _parse_json_response(text)

    def test_partial_json_raises(self):
        text = '{"key": "value"'  # missing closing brace
        with pytest.raises((ValueError, Exception)):
            _parse_json_response(text)

    def test_multiple_fences_picks_valid(self):
        text = (
            '```\nnot json\n```\n'
            '```json\n{"valid": true}\n```\n'
            '```\nalso not json\n```'
        )
        result = _parse_json_response(text)
        assert result["valid"] is True

    def test_complex_vibe_report_schema(self):
        """Test with a realistic vibe-report analysis structure."""
        text = '''```json
{
  "archetype": {"name": "The Debugger", "description": "Finds bugs like a truffle pig."},
  "superpowers": [{"name": "Pattern Recognition", "description": "Spots issues fast."}],
  "communication_style": "Terse and direct.",
  "blind_spots": [],
  "questioning_behavior": {
    "question_ratio": "~20%",
    "dominant_type": "diagnostic"
  }
}
```'''
        result = _parse_json_response(text)
        assert result["archetype"]["name"] == "The Debugger"
        assert len(result["superpowers"]) == 1
        assert result["questioning_behavior"]["dominant_type"] == "diagnostic"
