"""Tests for negative sampling strategies."""

from __future__ import annotations

import unittest

from shortchain.config import NegativeSamplingConfig
from shortchain.dataset.negatives import (
    RandomSampler,
    HardNegativeSampler,
    MixedSampler,
    create_sampler,
)
from shortchain.features.stats import CorpusStats
from shortchain.ingest.schema import Span, Trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trajectories() -> list[Trajectory]:
    """Create a small corpus for testing negative sampling."""
    def _traj(tid, intent, app, tools):
        spans = [
            Span(agent_name="a", action=t, observation=f"obs_{t}")
            for t in tools
        ]
        return Trajectory(task_id=tid, intent=intent, app_name=app, spans=spans)

    return [
        _traj("t1", "Send email", "gmail", ["search_emails", "send_email"]),
        _traj("t2", "Read email", "gmail", ["search_emails", "get_email", "reply_to_email"]),
        _traj("t3", "Make call", "phone", ["search_contacts", "make_call"]),
        _traj("t4", "Send text", "phone", ["search_contacts", "send_message"]),
        _traj("t5", "Play song", "spotify", ["search_tracks", "play_tracks"]),
        _traj("t6", "Create note", "simplenote", ["create_note", "search_web"]),
    ]


def _make_catalog() -> dict[str, str]:
    """All tools in the corpus."""
    return {
        "search_emails": "Search emails",
        "send_email": "Send email",
        "get_email": "Get email",
        "reply_to_email": "Reply to email",
        "search_contacts": "Search contacts",
        "make_call": "Make call",
        "send_message": "Send message",
        "search_tracks": "Search tracks",
        "play_tracks": "Play tracks",
        "create_note": "Create note",
        "search_web": "Search web",
    }


# ---------------------------------------------------------------------------
# RandomSampler
# ---------------------------------------------------------------------------

class TestRandomSampler(unittest.TestCase):

    def setUp(self):
        self.catalog = _make_catalog()
        self.sampler = RandomSampler(self.catalog, random_state=42)

    def test_no_overlap_with_positives(self):
        positives = {"search_emails", "send_email"}
        result = self.sampler.sample(positives, "gmail", 5)
        for tool in result:
            self.assertNotIn(tool, positives)

    def test_correct_count(self):
        positives = {"search_emails"}
        result = self.sampler.sample(positives, "gmail", 5)
        self.assertEqual(len(result), 5)

    def test_max_pool_size(self):
        """Cannot sample more than the pool size."""
        positives = {"search_emails"}
        pool_size = len(self.catalog) - 1  # minus positive
        result = self.sampler.sample(positives, "gmail", 100)
        self.assertEqual(len(result), pool_size)

    def test_deterministic_with_seed(self):
        s1 = RandomSampler(self.catalog, random_state=42)
        s2 = RandomSampler(self.catalog, random_state=42)
        positives = {"search_emails"}
        r1 = s1.sample(positives, "gmail", 5)
        r2 = s2.sample(positives, "gmail", 5)
        self.assertEqual(r1, r2)

    def test_empty_pool(self):
        positives = set(self.catalog.keys())
        result = self.sampler.sample(positives, "gmail", 5)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# HardNegativeSampler
# ---------------------------------------------------------------------------

class TestHardNegativeSampler(unittest.TestCase):

    def setUp(self):
        self.trajs = _make_trajectories()
        self.catalog = _make_catalog()
        self.stats = CorpusStats.from_trajectories(self.trajs)
        self.sampler = HardNegativeSampler(
            self.catalog,
            corpus_stats=self.stats,
            random_state=42,
            same_app_weight=0.4,
            co_usage_weight=0.3,
            similarity_weight=0.3,
        )

    def test_no_overlap_with_positives(self):
        positives = {"search_emails", "send_email"}
        result = self.sampler.sample(positives, "gmail", 5)
        for tool in result:
            self.assertNotIn(tool, positives)

    def test_same_app_preference(self):
        """Hard negatives should include same-app tools when available."""
        positives = {"send_email"}
        result = self.sampler.sample(positives, "gmail", 5)
        gmail_tools = set(self.stats.get_same_app_tools("gmail")) - positives
        # At least one same-app tool should appear
        overlap = gmail_tools & set(result)
        self.assertGreater(len(overlap), 0, f"Expected same-app tools in {result}")

    def test_correct_count(self):
        positives = {"search_emails"}
        result = self.sampler.sample(positives, "gmail", 5)
        self.assertEqual(len(result), 5)

    def test_deterministic(self):
        s1 = HardNegativeSampler(self.catalog, self.stats, random_state=42)
        s2 = HardNegativeSampler(self.catalog, self.stats, random_state=42)
        positives = {"send_email"}
        r1 = s1.sample(positives, "gmail", 5)
        r2 = s2.sample(positives, "gmail", 5)
        self.assertEqual(r1, r2)


# ---------------------------------------------------------------------------
# MixedSampler
# ---------------------------------------------------------------------------

class TestMixedSampler(unittest.TestCase):

    def setUp(self):
        self.trajs = _make_trajectories()
        self.catalog = _make_catalog()
        self.stats = CorpusStats.from_trajectories(self.trajs)
        self.sampler = MixedSampler(
            self.catalog,
            corpus_stats=self.stats,
            random_state=42,
            hard_ratio=0.5,
        )

    def test_no_overlap_with_positives(self):
        positives = {"search_emails", "send_email"}
        result = self.sampler.sample(positives, "gmail", 6)
        for tool in result:
            self.assertNotIn(tool, positives)

    def test_correct_count(self):
        positives = {"search_emails"}
        result = self.sampler.sample(positives, "gmail", 6)
        self.assertEqual(len(result), 6)

    def test_no_duplicates(self):
        positives = {"search_emails"}
        result = self.sampler.sample(positives, "gmail", 6)
        self.assertEqual(len(result), len(set(result)))


# ---------------------------------------------------------------------------
# create_sampler factory
# ---------------------------------------------------------------------------

class TestCreateSampler(unittest.TestCase):

    def test_random(self):
        sampler = create_sampler(
            NegativeSamplingConfig(strategy="random"),
            catalog=_make_catalog(),
        )
        self.assertIsInstance(sampler, RandomSampler)

    def test_hard(self):
        sampler = create_sampler(
            NegativeSamplingConfig(strategy="hard"),
            catalog=_make_catalog(),
        )
        self.assertIsInstance(sampler, HardNegativeSampler)

    def test_mixed(self):
        sampler = create_sampler(
            NegativeSamplingConfig(strategy="mixed"),
            catalog=_make_catalog(),
        )
        self.assertIsInstance(sampler, MixedSampler)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            create_sampler(NegativeSamplingConfig(strategy="invalid"))


if __name__ == "__main__":
    unittest.main()
