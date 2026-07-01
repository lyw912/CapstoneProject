import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "sensitive_input_filter",
    PROJECT_ROOT / "utils" / "sensitive_input_filter.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class SensitiveInputFilterTests(unittest.TestCase):
    def test_empty_input_is_allowed(self):
        self.assertFalse(module.check_sensitive_input("", enabled=True, words_file=Path("missing.txt")))

    def test_disabled_filter_allows_everything(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as handle:
            handle.write("blockedterm\n")
            words_file = Path(handle.name)
        try:
            self.assertFalse(
                module.check_sensitive_input("contains blockedterm here", enabled=False, words_file=words_file)
            )
        finally:
            words_file.unlink(missing_ok=True)

    def test_detects_case_insensitive_match(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as handle:
            handle.write("BadWord\n")
            words_file = Path(handle.name)
        try:
            module._cached_words.cache_clear()
            self.assertTrue(module.check_sensitive_input("hello BADword world", words_file=words_file))
        finally:
            words_file.unlink(missing_ok=True)

    def test_detects_fullwidth_variants(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as handle:
            handle.write("测试词\n")
            words_file = Path(handle.name)
        try:
            module._cached_words.cache_clear()
            self.assertTrue(module.check_sensitive_input("包含测　试词的内容", words_file=words_file))
        finally:
            words_file.unlink(missing_ok=True)

    def test_check_sensitive_fields_returns_field_name(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as handle:
            handle.write("secret\n")
            words_file = Path(handle.name)
        try:
            module._cached_words.cache_clear()
            result = module.check_sensitive_fields(
                {"query": "safe topic", "custom_template": "uses secret word"},
                words_file=words_file,
            )
            self.assertTrue(result.blocked)
            self.assertEqual(result.field, "custom_template")
        finally:
            words_file.unlink(missing_ok=True)

    def test_payload_shape(self):
        payload = module.sensitive_input_payload("query")
        self.assertEqual(payload["error_code"], "sensitive_input")
        self.assertIn("blocked terms", payload["message"])


if __name__ == "__main__":
    unittest.main()
