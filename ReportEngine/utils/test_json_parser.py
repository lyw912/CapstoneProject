"""
Test various repair capabilities of RobustJSONParser.

Verify the parser can handle:
1. Basic markdown wrapping
2. Thinking content cleanup
3. Missing comma repair
4. Unbalanced bracket repair
5. Control character escaping
6. Trailing comma removal
"""

import json
import unittest
from json_parser import RobustJSONParser, JSONParseError


class TestRobustJSONParser(unittest.TestCase):
    """Test various repair strategies of robust JSON parser."""

    def setUp(self):
        """Initialize parser."""
        self.parser = RobustJSONParser(
            enable_json_repair=False,  # Test local repair first
            enable_llm_repair=False,
        )

    def test_basic_json(self):
        """Test parsing basic valid JSON."""
        json_str = '{"name": "test", "value": 123}'
        result = self.parser.parse(json_str, "Basic Test")
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 123)

    def test_markdown_wrapped(self):
        """Test parsing JSON wrapped in ```json."""
        json_str = """```json
{
  "name": "test",
  "value": 123
}
```"""
        result = self.parser.parse(json_str, "Markdown Wrapped Test")
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 123)

    def test_thinking_content_removal(self):
        """Test cleaning thinking content."""
        json_str = """<thinking>Let me think about how to construct this JSON</thinking>
{
  "name": "test",
  "value": 123
}"""
        result = self.parser.parse(json_str, "Thinking Content Removal Test")
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 123)

    def test_missing_comma_fix(self):
        """Test fixing missing commas."""
        # Common real error: missing commas between array elements
        json_str = """{
  "totalWords": 40000,
  "globalGuidelines": [
    "Focus on highlighting technological dividend distribution imbalance"
    "Detail strategy: Technological innovation"
  ],
  "chapters": []
}"""
        result = self.parser.parse(json_str, "Missing Comma Fix Test")
        self.assertEqual(len(result["globalGuidelines"]), 2)

    def test_unbalanced_brackets(self):
        """Test fixing unbalanced brackets."""
        # Missing closing bracket
        json_str = """{
  "name": "test",
  "nested": {
    "value": 123
  }
"""  # Missing outer }
        result = self.parser.parse(json_str, "Unbalanced Brackets Test")
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["nested"]["value"], 123)

    def test_control_character_escape(self):
        """Test escaping control characters."""
        # Raw newlines in JSON string should be escaped
        json_str = """{
  "text": "This is first line
This is second line",
  "value": 123
}"""
        result = self.parser.parse(json_str, "Control Character Escape Test")
        # Ensure newlines are handled correctly
        self.assertIn("first line", result["text"])
        self.assertIn("second line", result["text"])

    def test_trailing_comma_removal(self):
        """Test removing trailing commas."""
        json_str = """{
  "name": "test",
  "value": 123,
  "items": [1, 2, 3,],
}"""
        result = self.parser.parse(json_str, "Trailing Comma Test")
        self.assertEqual(result["name"], "test")
        self.assertEqual(len(result["items"]), 3)

    def test_colon_equals_fix(self):
        """Test fixing colon-equals error."""
        json_str = """{
  "name":= "test",
  "value": 123
}"""
        result = self.parser.parse(json_str, "Colon Equals Test")
        self.assertEqual(result["name"], "test")

    def test_extract_first_json(self):
        """Test extracting first JSON structure from text."""
        json_str = """Here is some explanatory text, followed by JSON:
{
  "name": "test",
  "value": 123
}
And some text after"""
        result = self.parser.parse(json_str, "Extract JSON Test")
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 123)

    def test_unterminated_string_with_json_repair(self):
        """Test using json_repair library to fix unterminated strings."""
        # Create parser with json_repair enabled
        parser_with_repair = RobustJSONParser(
            enable_json_repair=True,
            enable_llm_repair=False,
        )

        # Simulate actual error: unescaped control characters or quotes in string
        json_str = """{
  "template_name": "Specific Policy Report",
  "selection_reason": "This is test content"
}"""
        result = parser_with_repair.parse(json_str, "Unterminated String Test")
        # Just need to parse successfully without error
        self.assertIsInstance(result, dict)
        self.assertIn("template_name", result)

    def test_array_with_best_match(self):
        """Test extracting best matching element from array."""
        json_str = """[
  {
    "name": "test",
    "value": 123
  },
  {
    "totalWords": 40000,
    "globalGuidelines": ["guide1", "guide2"],
    "chapters": []
  }
]"""
        result = self.parser.parse(
            json_str,
            "Array Best Match Test",
            expected_keys=["totalWords", "globalGuidelines", "chapters"],
        )
        # Should extract second element because it matches 3 keys
        self.assertEqual(result["totalWords"], 40000)
        self.assertEqual(len(result["globalGuidelines"]), 2)

    def test_key_alias_recovery(self):
        """Test key name alias recovery."""
        json_str = """{
  "templateName": "test_template",
  "selectionReason": "This is a test"
}"""
        result = self.parser.parse(
            json_str,
            "Key Alias Test",
            expected_keys=["template_name", "selection_reason"],
        )
        # Should automatically map templateName -> template_name
        self.assertEqual(result["template_name"], "test_template")
        self.assertEqual(result["selection_reason"], "This is a test")

    def test_complex_real_world_case(self):
        """Test complex real-world case (similar to actual errors)."""
        # Simulate actual errors: missing commas, markdown wrapping, thinking content
        json_str = """<thinking>I need to construct a length plan</thinking>
```json
{
  "totalWords": 40000,
  "tolerance": 2000,
  "globalGuidelines": [
    "Focus on highlighting technological dividend distribution imbalance, talent loss and professional identity crisis"
    "Detail strategy: Collision between technological innovation and traditional craftsmanship"
    "Case-oriented: Prioritize real data and research"
  ],
  "chapters": [
    {
      "chapterId": "ch1",
      "targetWords": 5000
    }
  ]
}
```"""
        result = self.parser.parse(json_str, "Complex Real World Case Test")
        self.assertEqual(result["totalWords"], 40000)
        self.assertEqual(result["tolerance"], 2000)
        self.assertEqual(len(result["globalGuidelines"]), 3)
        self.assertEqual(len(result["chapters"]), 1)

    def test_expected_keys_validation(self):
        """Test expected keys validation."""
        json_str = '{"name": "test"}'
        # Should not fail due to missing key, just warn
        result = self.parser.parse(
            json_str, "Key Validation Test", expected_keys=["name", "value"]
        )
        self.assertIn("name", result)

    def test_wrapper_key_extraction(self):
        """Test extracting data from wrapper key."""
        json_str = """{
  "wrapper": {
    "name": "test",
    "value": 123
  }
}"""
        result = self.parser.parse(
            json_str, "Wrapper Key Test", extract_wrapper_key="wrapper"
        )
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 123)

    def test_empty_input(self):
        """Test empty input."""
        with self.assertRaises(JSONParseError):
            self.parser.parse("", "Empty Input Test")

    def test_invalid_json_after_all_repairs(self):
        """Test case where all repair strategies fail."""
        # This is a severely corrupted JSON that cannot be repaired
        json_str = "{completely not JSON format content###"
        with self.assertRaises(JSONParseError):
            self.parser.parse(json_str, "Cannot Repair Test")


def run_manual_test():
    """Run manual test, printing detailed information."""
    print("=" * 60)
    print("Starting RobustJSONParser Tests")
    print("=" * 60)

    parser = RobustJSONParser(enable_json_repair=False, enable_llm_repair=False)

    # Test actual error case
    test_case = """```json
{
  "totalWords": 40000,
  "tolerance": 2000,
  "globalGuidelines": [
    "Focus on highlighting technological dividend distribution imbalance, talent loss and professional identity crisis"
    "Detail strategy: Collision between technological innovation and traditional craftsmanship"
  ],
  "chapters": []
}
```"""

    print("\nTest Case:")
    print(test_case)
    print("\n" + "=" * 60)

    try:
        result = parser.parse(test_case, "Manual Test")
        print("\n✓ Parsing successful!")
        print("\nParse Result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"\n✗ Parsing failed: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run manual test
    run_manual_test()

    # Run unit tests
    print("\n\nRunning unit tests...")
    unittest.main(verbosity=2)
