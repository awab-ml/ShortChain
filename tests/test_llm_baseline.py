"""Tests for the cost-bound LLM tool-selection baseline helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_llm_baseline", _ROOT / "scripts" / "run_llm_baseline.py"
)
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)


class TestParseRanked:
    def test_plain_json_array(self):
        assert lb.parse_ranked('["a","b","c"]') == ["a", "b", "c"]

    def test_code_fenced(self):
        assert lb.parse_ranked('```json\n["a","b"]\n```') == ["a", "b"]

    def test_dict_with_list_key(self):
        assert lb.parse_ranked('{"ranked": ["a","b"]}') == ["a", "b"]

    def test_list_of_objects(self):
        assert lb.parse_ranked('[{"name":"a"},{"name":"b"}]') == ["a", "b"]

    def test_reasoning_text_fallback(self):
        # reasoning-only content (JSON array embedded in prose)
        text = "The answer is: [\"a\", \"b\"]\nbecause ..."
        assert lb.parse_ranked(text) == ["a", "b"]

    def test_none_and_garbage(self):
        assert lb.parse_ranked(None) == []
        assert lb.parse_ranked("no tools at all") == []


class TestTaskMetrics:
    def test_perfect(self):
        m = lb.task_metrics(["a", "b", "c"], {"a", "b", "c"}, [1, 3])
        assert m["r_precision"] == 1.0
        assert m["mrr"] == 1.0
        assert m["recall_at_1"] == 1 / 3
        assert m["recall_at_3"] == 1.0

    def test_partial(self):
        m = lb.task_metrics(["x", "b", "y"], {"a", "b"}, [1, 2, 5])
        assert m["r_precision"] == 0.5  # top-R=2 contains one relevant
        assert m["mrr"] == 0.5
        assert m["recall_at_1"] == 0.0
        assert m["recall_at_2"] == 0.5

    def test_empty_relevant(self):
        m = lb.task_metrics(["a"], set(), [1])
        assert m["r_precision"] == 0.0 and m["mrr"] == 0.0


class TestToolList:
    def test_build_tool_list_deterministic(self):
        tl = lb.build_tool_list({"b__x": "desc b", "a__y": "desc a"}, {})
        assert tl.index("a__y") < tl.index("b__x")
        assert "a__y: desc a [0 args]" in tl
