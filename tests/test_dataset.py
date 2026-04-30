"""Tests for the dataset construction module."""

from __future__ import annotations

import pytest
import pandas as pd

from tabagent.config import DatasetConfig
from tabagent.dataset.builder import DatasetBuilder, build_dataset
from tabagent.dataset.splitter import GroupStratifiedSplitter
from tabagent.ingest.schema import Step, Trajectory


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
            steps=[
                Step(action="search_contacts", thoughts="Looking up John"),
                Step(action="send_email", thoughts="Sending the email"),
            ],
        ),
        Trajectory(
            task_id="t2",
            intent="Play a song on Spotify",
            app_name="spotify",
            steps=[
                Step(action="search_tracks", thoughts="Finding songs"),
                Step(action="play_tracks", thoughts="Playing music"),
            ],
        ),
        Trajectory(
            task_id="t3",
            intent="Order something from Amazon",
            app_name="amazon",
            steps=[
                Step(action="search_products", thoughts="Searching products"),
                Step(action="add_to_cart", thoughts="Adding to cart"),
                Step(action="place_order", thoughts="Ordering"),
            ],
        ),
        Trajectory(
            task_id="t4",
            intent="Reply to an email",
            app_name="gmail",
            steps=[
                Step(action="search_emails", thoughts="Finding email"),
                Step(action="reply_to_email", thoughts="Replying"),
            ],
        ),
        Trajectory(
            task_id="t5",
            intent="Create a playlist",
            app_name="spotify",
            steps=[
                Step(action="search_tracks", thoughts="Finding tracks"),
                Step(action="create_playlist", thoughts="Creating playlist"),
                Step(action="add_tracks_to_playlist", thoughts="Adding tracks"),
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
        required = {"task_id", "intent", "app_name", "n_steps", "tool_name", "label"}
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
        # n_steps should be > 0
        assert (df["n_steps"] > 0).all()


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
        from tabagent.config import SplitterConfig
        splitter = GroupStratifiedSplitter(SplitterConfig(n_folds=3))

        for train_fold, val_fold in splitter.kfold_split(df):
            train_tasks = set(train_fold["task_id"].unique())
            val_tasks = set(val_fold["task_id"].unique())
            assert train_tasks.isdisjoint(val_tasks)
