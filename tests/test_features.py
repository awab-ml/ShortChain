"""Tests for the Phase 2 features module.

Covers: CorpusStats, ContextFeatureBuilder, ToolFeatureBuilder,
TfidfEncoder, DenseEncoder (fallback), and FeaturePipeline.
"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tabagent.config import FeaturesConfig
from tabagent.features.context import ContextFeatureBuilder
from tabagent.features.encoders import TfidfEncoder, DenseEncoder, create_encoder
from tabagent.features.pipeline import FeaturePipeline
from tabagent.features.stats import CorpusStats
from tabagent.features.tool import ToolFeatureBuilder
from tabagent.ingest.schema import Step, Trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trajectory(
    task_id: str = "t1",
    intent: str = "Send an email",
    app_name: str = "gmail",
    tools: list[str] | None = None,
) -> Trajectory:
    """Create a minimal Trajectory for testing."""
    tools = tools or ["search_emails", "send_email"]
    steps = [
        Step(agent_name="agent", action=t, observation=f"did {t}", thoughts=f"thinking about {t}")
        for t in tools
    ]
    return Trajectory(
        task_id=task_id,
        intent=intent,
        app_name=app_name,
        steps=steps,
    )


def _make_trajectories() -> list[Trajectory]:
    """Create a small corpus of trajectories for testing."""
    return [
        _make_trajectory("t1", "Send email", "gmail", ["search_emails", "send_email"]),
        _make_trajectory("t2", "Read email", "gmail", ["search_emails", "get_email"]),
        _make_trajectory("t3", "Make call", "phone", ["search_contacts", "make_call"]),
        _make_trajectory("t4", "Play song", "spotify", ["search_tracks", "play_tracks"]),
        _make_trajectory("t5", "Create note", "simplenote", ["create_note"]),
    ]


# ---------------------------------------------------------------------------
# CorpusStats
# ---------------------------------------------------------------------------

class TestCorpusStats(unittest.TestCase):

    def setUp(self):
        self.trajs = _make_trajectories()
        self.stats = CorpusStats.from_trajectories(self.trajs)

    def test_total_trajectories(self):
        self.assertEqual(self.stats.total_trajectories, 5)

    def test_tool_frequency(self):
        # search_emails appears in t1 and t2
        self.assertEqual(self.stats.tool_frequency["search_emails"], 2)
        self.assertEqual(self.stats.tool_frequency["create_note"], 1)

    def test_app_tools(self):
        gmail_tools = set(self.stats.app_tools["gmail"])
        self.assertEqual(gmail_tools, {"search_emails", "send_email", "get_email"})

    def test_app_tool_count(self):
        self.assertEqual(self.stats.app_tool_count["gmail"], 3)
        self.assertEqual(self.stats.app_tool_count["phone"], 2)

    def test_co_occurrence(self):
        # search_emails and send_email co-occur in t1
        co = self.stats.get_co_occurring_tools("search_emails")
        self.assertIn("send_email", co)
        self.assertGreater(co["send_email"], 0)

    def test_get_same_app_tools(self):
        tools = self.stats.get_same_app_tools("spotify")
        self.assertIn("search_tracks", tools)
        self.assertIn("play_tracks", tools)

    def test_get_same_app_tools_unknown(self):
        tools = self.stats.get_same_app_tools("unknown_app")
        self.assertEqual(tools, [])

    def test_serialization(self):
        """CorpusStats should be serializable via Pydantic."""
        data = self.stats.model_dump()
        restored = CorpusStats.model_validate(data)
        self.assertEqual(restored.total_trajectories, 5)
        self.assertEqual(restored.tool_frequency, self.stats.tool_frequency)


# ---------------------------------------------------------------------------
# ContextFeatureBuilder
# ---------------------------------------------------------------------------

class TestContextFeatureBuilder(unittest.TestCase):

    def setUp(self):
        self.trajs = _make_trajectories()
        self.stats = CorpusStats.from_trajectories(self.trajs)
        self.builder = ContextFeatureBuilder(
            corpus_stats=self.stats,
            include_state=True,
            include_dependencies=True,
        )

    def test_core_features_present(self):
        features = self.builder.build(self.trajs[0])
        self.assertIn("task_id", features)
        self.assertIn("intent", features)
        self.assertIn("app_name", features)
        self.assertIn("n_steps", features)
        self.assertIn("previous_tools", features)
        self.assertIn("last_thought", features)

    def test_state_features_present(self):
        features = self.builder.build(self.trajs[0])
        self.assertIn("step_index", features)
        self.assertIn("last_action", features)
        self.assertIn("last_observation", features)
        self.assertIn("unique_tools_so_far", features)
        self.assertIn("history_summary", features)

    def test_dependency_features_present(self):
        features = self.builder.build(self.trajs[0])
        self.assertIn("tool_diversity", features)
        self.assertIn("app_tool_count", features)

    def test_step_index_none_default(self):
        """step_index=None gives trajectory-level features."""
        features = self.builder.build(self.trajs[0], step_index=None)
        self.assertEqual(features["n_steps"], 2)
        self.assertEqual(features["step_index"], 2)  # final position

    def test_step_index_specific(self):
        """step_index=0 gives features up to step 0."""
        features = self.builder.build(self.trajs[0], step_index=0)
        self.assertEqual(features["n_steps"], 1)
        self.assertEqual(features["step_index"], 0)

    def test_state_disabled(self):
        builder = ContextFeatureBuilder(include_state=False, include_dependencies=True)
        features = builder.build(self.trajs[0])
        self.assertNotIn("step_index", features)
        self.assertNotIn("last_action", features)

    def test_dependencies_disabled(self):
        builder = ContextFeatureBuilder(include_state=True, include_dependencies=False)
        features = builder.build(self.trajs[0])
        self.assertNotIn("tool_diversity", features)
        self.assertNotIn("app_tool_count", features)

    def test_app_tool_count_from_stats(self):
        features = self.builder.build(self.trajs[0])
        # gmail has 3 tools in corpus
        self.assertEqual(features["app_tool_count"], 3)


# ---------------------------------------------------------------------------
# ToolFeatureBuilder
# ---------------------------------------------------------------------------

class TestToolFeatureBuilder(unittest.TestCase):

    def setUp(self):
        self.trajs = _make_trajectories()
        self.stats = CorpusStats.from_trajectories(self.trajs)
        self.builder = ToolFeatureBuilder(corpus_stats=self.stats)

    def test_core_features(self):
        features = self.builder.build("send_email", {"description": "Send an email message"})
        self.assertEqual(features["tool_name"], "send_email")
        self.assertEqual(features["tool_description"], "Send an email message")
        self.assertEqual(features["tool_name_length"], len("send_email"))
        self.assertTrue(features["has_description"])

    def test_no_description(self):
        features = self.builder.build("send_email")
        self.assertFalse(features["has_description"])
        self.assertEqual(features["tool_description"], "")

    def test_app_match_with_context(self):
        context = {"app_name": "gmail", "previous_tools": "search_emails"}
        features = self.builder.build("send_email", context=context)
        self.assertEqual(features["tool_app_match"], 1)

    def test_app_no_match(self):
        context = {"app_name": "phone", "previous_tools": ""}
        features = self.builder.build("send_email", context=context)
        self.assertEqual(features["tool_app_match"], 0)

    def test_tool_frequency(self):
        features = self.builder.build("search_emails")
        self.assertEqual(features["tool_frequency"], 2)

    def test_co_occurrence_score(self):
        context = {"app_name": "gmail", "previous_tools": "search_emails"}
        features = self.builder.build("send_email", context=context)
        self.assertGreater(features["tool_co_occurrence"], 0)

    def test_no_corpus_stats(self):
        builder = ToolFeatureBuilder(corpus_stats=None)
        features = builder.build("send_email")
        self.assertNotIn("tool_frequency", features)
        self.assertNotIn("tool_co_occurrence", features)


# ---------------------------------------------------------------------------
# TfidfEncoder
# ---------------------------------------------------------------------------

class TestTfidfEncoder(unittest.TestCase):

    def test_fit_transform(self):
        enc = TfidfEncoder(max_features=100)
        texts = ["hello world", "world of code", "hello code"]
        result = enc.fit_transform(texts)
        self.assertEqual(result.shape[0], 3)
        self.assertGreater(result.shape[1], 0)

    def test_transform_after_fit(self):
        enc = TfidfEncoder(max_features=100)
        enc.fit(["hello world", "world of code"])
        result = enc.transform(["hello code"])
        self.assertEqual(result.shape[0], 1)

    def test_empty_corpus(self):
        enc = TfidfEncoder()
        result = enc.fit_transform(["", "", ""])
        self.assertEqual(result.shape[0], 3)

    def test_output_dim(self):
        enc = TfidfEncoder(max_features=100)
        enc.fit(["hello world", "world of code"])
        self.assertGreater(enc.output_dim, 0)

    def test_output_dim_before_fit(self):
        enc = TfidfEncoder()
        self.assertEqual(enc.output_dim, 0)


# ---------------------------------------------------------------------------
# DenseEncoder (fallback)
# ---------------------------------------------------------------------------

class TestDenseEncoder(unittest.TestCase):

    def test_fallback_to_tfidf(self):
        """Without sentence-transformers, should fall back to TF-IDF."""
        enc = DenseEncoder(model_name="nonexistent/model")
        self.assertTrue(enc._is_fallback)

    def test_fallback_fit_transform(self):
        enc = DenseEncoder(model_name="nonexistent/model")
        texts = ["hello world", "world of code"]
        result = enc.fit_transform(texts)
        self.assertEqual(result.shape[0], 2)


# ---------------------------------------------------------------------------
# create_encoder factory
# ---------------------------------------------------------------------------

class TestCreateEncoder(unittest.TestCase):

    def test_tfidf(self):
        enc = create_encoder("tfidf")
        self.assertIsInstance(enc, TfidfEncoder)

    def test_auto_fallback(self):
        enc = create_encoder("auto")
        # Should work regardless of sentence-transformers availability
        result = enc.fit_transform(["hello", "world"])
        self.assertEqual(result.shape[0], 2)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            create_encoder("invalid_encoder")


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------

class TestFeaturePipeline(unittest.TestCase):

    def _sample_df(self):
        return pd.DataFrame([
            {
                "intent": "Send email",
                "app_name": "gmail",
                "n_steps": 2,
                "previous_tools": "search_emails",
                "last_thought": "compose message",
                "tool_name": "send_email",
                "tool_description": "Send an email",
            },
            {
                "intent": "Make call",
                "app_name": "phone",
                "n_steps": 1,
                "previous_tools": "",
                "last_thought": "dial number",
                "tool_name": "make_call",
                "tool_description": "Make a phone call",
            },
        ])

    def test_fit_transform_dataframe(self):
        pipe = FeaturePipeline()
        df = self._sample_df()
        result = pipe.fit_transform(df)
        self.assertEqual(result.shape[0], 2)
        self.assertGreater(result.shape[1], 0)

    def test_fit_transform_list_of_dicts(self):
        pipe = FeaturePipeline()
        data = self._sample_df().to_dict("records")
        result = pipe.fit_transform(data)
        self.assertEqual(result.shape[0], 2)
        self.assertGreater(result.shape[1], 0)

    def test_transform_after_fit(self):
        pipe = FeaturePipeline()
        df = self._sample_df()
        pipe.fit_transform(df)
        result = pipe.transform(df)
        self.assertEqual(result.shape[0], 2)

    def test_transform_unfitted_raises(self):
        pipe = FeaturePipeline()
        with self.assertRaises(RuntimeError):
            pipe.transform(self._sample_df())

    def test_save_load_roundtrip(self):
        pipe = FeaturePipeline()
        df = self._sample_df()
        original = pipe.fit_transform(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline.pkl"
            pipe.save(path)
            loaded = FeaturePipeline.load(path)
            restored = loaded.transform(df)

        np.testing.assert_array_almost_equal(original, restored)

    def test_invalid_input_type_raises(self):
        pipe = FeaturePipeline()
        with self.assertRaises(TypeError):
            pipe.fit_transform("not a dataframe")

    def test_with_extra_numeric_columns(self):
        """Pipeline should handle Phase 2 numeric columns gracefully."""
        pipe = FeaturePipeline()
        df = self._sample_df()
        df["step_index"] = [2, 1]
        df["unique_tools_so_far"] = [2, 1]
        df["tool_diversity"] = [1.0, 1.0]
        df["app_tool_count"] = [3, 2]
        df["tool_name_length"] = [10, 9]
        df["tool_frequency"] = [1, 1]
        df["tool_co_occurrence"] = [0.5, 0.0]
        df["has_description"] = [True, True]
        df["tool_app_match"] = [1, 1]
        result = pipe.fit_transform(df)
        self.assertEqual(result.shape[0], 2)
        self.assertGreater(result.shape[1], 10)  # more features now


if __name__ == "__main__":
    unittest.main()
