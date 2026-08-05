"""Tests for the dataset construction module."""

from __future__ import annotations

import pytest
import pandas as pd

from shortchain.config import DatasetConfig
from shortchain.dataset.builder import DatasetBuilder, build_dataset
from shortchain.dataset.splitter import GroupStratifiedSplitter
from shortchain.features.stats import CorpusStats
from shortchain.ingest.schema import Span, Trajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_trajectories() -> list[Trajectory]:
    """Create a small set of trajectories for testing."""
    return [
        Trajectory(
            task_id="t1",
            intent="Send an email to John",
            app_name="gmail",
            spans=[
                Span(action="search_contacts", thoughts="Looking up John"),
                Span(action="send_email", thoughts="Sending the email"),
            ],
        ),
        Trajectory(
            task_id="t2",
            intent="Play a song on Spotify",
            app_name="spotify",
            spans=[
                Span(action="search_tracks", thoughts="Finding songs"),
                Span(action="play_tracks", thoughts="Playing music"),
            ],
        ),
        Trajectory(
            task_id="t3",
            intent="Order something from Amazon",
            app_name="amazon",
            spans=[
                Span(action="search_products", thoughts="Searching products"),
                Span(action="add_to_cart", thoughts="Adding to cart"),
                Span(action="place_order", thoughts="Ordering"),
            ],
        ),
        Trajectory(
            task_id="t4",
            intent="Reply to an email",
            app_name="gmail",
            spans=[
                Span(action="search_emails", thoughts="Finding email"),
                Span(action="reply_to_email", thoughts="Replying"),
            ],
        ),
        Trajectory(
            task_id="t5",
            intent="Create a playlist",
            app_name="spotify",
            spans=[
                Span(action="search_tracks", thoughts="Finding tracks"),
                Span(action="create_playlist", thoughts="Creating playlist"),
                Span(action="add_tracks_to_playlist", thoughts="Adding tracks"),
            ],
        ),
    ]


@pytest.fixture
def tool_catalog() -> dict[str, str]:
    return {
        "search_contacts": "Search for contacts",
        "send_email": "Send an email",
        "search_emails": "Search inbox",
        "reply_to_email": "Reply to an email",
        "search_tracks": "Search for music",
        "play_tracks": "Play music",
        "create_playlist": "Create a new playlist",
        "add_tracks_to_playlist": "Add tracks to playlist",
        "search_products": "Search Amazon products",
        "add_to_cart": "Add item to cart",
        "place_order": "Place an order",
        "make_call": "Make a phone call",
    }


# ---------------------------------------------------------------------------
# DatasetBuilder tests
# ---------------------------------------------------------------------------

class TestDatasetBuilder:
    def test_build_creates_dataframe(self, sample_trajectories, tool_catalog):
        builder = DatasetBuilder(tool_catalog=tool_catalog)
        df = builder.build(sample_trajectories)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self, sample_trajectories, tool_catalog):
        df = build_dataset(sample_trajectories, tool_catalog=tool_catalog)
        required = {"task_id", "intent", "app_name", "n_spans", "tool_name", "label"}
        assert required.issubset(set(df.columns))

    def test_positive_labels_correct(self, sample_trajectories, tool_catalog):
        df = build_dataset(sample_trajectories, tool_catalog=tool_catalog)
        # For task t1, positives should be search_contacts and send_email
        t1_pos = df[(df["task_id"] == "t1") & (df["label"] == 1)]
        assert set(t1_pos["tool_name"]) == {"search_contacts", "send_email"}

    def test_negative_ratio(self, sample_trajectories, tool_catalog):
        config = DatasetConfig(negative_ratio=2)
        df = build_dataset(sample_trajectories, config=config, tool_catalog=tool_catalog)
        # For each task, negatives <= positives * ratio
        for tid in df["task_id"].unique():
            task_df = df[df["task_id"] == tid]
            n_pos = (task_df["label"] == 1).sum()
            n_neg = (task_df["label"] == 0).sum()
            assert n_neg <= n_pos * 2

    def test_derived_catalog(self, sample_trajectories):
        """When no explicit catalog, should derive from trajectories."""
        df = build_dataset(sample_trajectories)
        assert len(df) > 0
        assert "tool_name" in df.columns

    def test_context_features_populated(self, sample_trajectories, tool_catalog):
        df = build_dataset(sample_trajectories, tool_catalog=tool_catalog)
        # Intent should never be empty for our test data
        assert (df["intent"] != "").all()
        # n_spans is 0 at pre-execution (no lookahead leakage)
        assert (df["n_spans"] >= 0).all()

    def test_no_target_lookahead_leakage(self, sample_trajectories, tool_catalog):
        """Negative samples must never contain the candidate tool in previous_tools."""
        df = build_dataset(sample_trajectories, tool_catalog=tool_catalog)
        negatives = df[df["label"] == 0]
        for _, row in negatives.iterrows():
            prev = str(row.get("previous_tools", ""))
            if prev:
                prev_tools = [t.strip() for t in prev.split("|")]
                assert row["tool_name"] not in prev_tools, (
                    f"Leakage: tool '{row['tool_name']}' found in "
                    f"previous_tools for negative sample (task={row['task_id']})"
                )

    def test_build_reuses_frozen_corpus_stats(self, sample_trajectories):
        """Corpus stats must be frozen on the first build (the training set).

        Reusing the same builder on an evaluation set must NOT recompute
        tool_frequency / co-occurrence / app-counts from that set — that would
        leak the evaluation answers into the scored features.
        """
        builder = DatasetBuilder()
        train_df = builder.build(sample_trajectories)
        assert "brand_new_tool" not in set(train_df["tool_name"])
        assert builder.corpus_stats is not None

        eval_like = Trajectory(
            task_id="eval_1",
            intent="Use the brand new tool",
            app_name="newapp",
            spans=[Span(action="brand_new_tool", thoughts="")],
        )
        eval_df = builder.build([eval_like])

        row = eval_df[eval_df["tool_name"] == "brand_new_tool"].iloc[0]
        # Frequency must reflect the TRAIN corpus (0), not the evaluation set (1).
        assert row["tool_frequency"] == 0, (
            "Corpus stats were recomputed from evaluation data (leak)."
        )
        # The frozen stats object is still the train-derived one.
        assert builder.corpus_stats.get_tool_freq("send_email") == 1

    def test_corpus_stats_injectable(self, sample_trajectories):
        """An explicitly injected CorpusStats object is used and never recomputed."""
        injected = CorpusStats(tool_frequency={"send_email": 7})
        builder = DatasetBuilder(corpus_stats=injected)
        df = builder.build(sample_trajectories)

        assert builder.corpus_stats is injected
        row = df[df["tool_name"] == "send_email"].iloc[0]
        assert row["tool_frequency"] == 7

    def test_build_candidates_requires_frozen_stats(self, sample_trajectories):
        builder = DatasetBuilder()
        with pytest.raises(ValueError, match="frozen corpus"):
            builder.build_candidates(
                sample_trajectories[0],
                [{"tool_name": "x", "tool_description": ""}],
            )

    def test_build_candidates_labels_explicit_pool(self, sample_trajectories):
        builder = DatasetBuilder()
        train_df = builder.build(sample_trajectories)

        eval_builder = DatasetBuilder(corpus_stats=builder.corpus_stats)
        traj = Trajectory(
            task_id="eval_9",
            intent="Rank these tools",
            app_name="spotify",
            spans=[],
            success=True,
        )
        rows = eval_builder.build_candidates(
            traj,
            [
                {"tool_name": "send_email", "tool_description": "send"},
                {"tool_name": "search_tracks", "tool_description": "find music"},
                {"tool_name": "brand_new_tool", "tool_description": "new"},
            ],
            relevant_tools={"search_tracks", "brand_new_tool"},
        )
        assert len(rows) == 3
        by_tool = {r["tool_name"]: r for r in rows}
        assert by_tool["search_tracks"]["label"] == 1
        assert by_tool["brand_new_tool"]["label"] == 1
        assert by_tool["send_email"]["label"] == 0
        # Unseen tool must not receive train-derived frequency (no leak).
        assert by_tool["brand_new_tool"]["tool_frequency"] == 0
        # Schema columns match the training DataFrame.
        assert set(rows[0].keys()) >= set(train_df.columns)


# ---------------------------------------------------------------------------
# GroupStratifiedSplitter tests
# ---------------------------------------------------------------------------

class TestGroupStratifiedSplitter:
    def test_train_test_split(self, sample_trajectories, tool_catalog):
        df = build_dataset(sample_trajectories, tool_catalog=tool_catalog)
        splitter = GroupStratifiedSplitter()
        train, test = splitter.train_test_split(df)

        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) == len(df)

    def test_no_task_leakage(self, sample_trajectories, tool_catalog):
        """No task should appear in both train and test."""
        df = build_dataset(sample_trajectories, tool_catalog=tool_catalog)
        splitter = GroupStratifiedSplitter()
        train, test = splitter.train_test_split(df)

        train_tasks = set(train["task_id"].unique())
        test_tasks = set(test["task_id"].unique())
        assert train_tasks.isdisjoint(test_tasks), "Task leakage detected!"

    def test_kfold_no_leakage(self, sample_trajectories, tool_catalog):
        df = build_dataset(sample_trajectories, tool_catalog=tool_catalog)
        from shortchain.config import SplitterConfig
        splitter = GroupStratifiedSplitter(SplitterConfig(n_folds=3))

        for train_fold, val_fold in splitter.kfold_split(df):
            train_tasks = set(train_fold["task_id"].unique())
            val_tasks = set(val_fold["task_id"].unique())
            assert train_tasks.isdisjoint(val_tasks)
