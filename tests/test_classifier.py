"""Tests for the classifier module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabagent.config import ClassifierConfig
from tabagent.dataset.builder import build_dataset
from tabagent.head.classifier import TabAgentClassifier
from tabagent.ingest.schema import Step, Trajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Build a small training set."""
    trajectories = [
        Trajectory(
            task_id=f"t{i}",
            intent=f"Task intent number {i} for {app}",
            app_name=app,
            steps=[Step(action=t) for t in tools],
        )
        for i, (app, tools) in enumerate([
            ("gmail", ["search_contacts", "send_email"]),
            ("spotify", ["search_tracks", "play_tracks"]),
            ("amazon", ["search_products", "add_to_cart", "place_order"]),
            ("gmail", ["search_emails", "reply_to_email"]),
            ("spotify", ["search_tracks", "create_playlist"]),
            ("phone", ["make_call"]),
            ("gmail", ["search_contacts", "create_draft", "send_email"]),
            ("amazon", ["search_products", "get_product_details"]),
        ])
    ]

    catalog = {
        "search_contacts": "Search contacts",
        "send_email": "Send email",
        "search_emails": "Search inbox",
        "reply_to_email": "Reply to email",
        "create_draft": "Create draft",
        "search_tracks": "Search music",
        "play_tracks": "Play music",
        "create_playlist": "Create playlist",
        "search_products": "Search products",
        "add_to_cart": "Add to cart",
        "place_order": "Place order",
        "get_product_details": "Get product details",
        "make_call": "Make call",
    }

    df = build_dataset(trajectories, tool_catalog=catalog)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

class TestTabAgentClassifier:
    def test_fit_predict(self, training_data):
        X, y = training_data
        clf = TabAgentClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X),)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_binary(self, training_data):
        X, y = training_data
        clf = TabAgentClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_shortlist(self, training_data):
        X, y = training_data
        clf = TabAgentClassifier()
        clf.fit(X, y)
        results = clf.shortlist(X, top_k=3)
        assert len(results) > 0
        for task_shortlist in results:
            assert len(task_shortlist) <= 3
            for name, score in task_shortlist:
                assert isinstance(name, str)
                assert 0 <= score <= 1

    def test_save_load(self, training_data, tmp_path):
        X, y = training_data
        clf = TabAgentClassifier()
        clf.fit(X, y)
        proba_before = clf.predict_proba(X)

        model_path = tmp_path / "model.pkl"
        clf.save(model_path)

        loaded = TabAgentClassifier.load(model_path)
        proba_after = loaded.predict_proba(X)

        np.testing.assert_array_almost_equal(proba_before, proba_after)

    def test_random_forest_backend(self, training_data):
        X, y = training_data
        config = ClassifierConfig(model_type="random_forest")
        clf = TabAgentClassifier(config)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X),)

    def test_logistic_backend(self, training_data):
        X, y = training_data
        config = ClassifierConfig(model_type="logistic")
        clf = TabAgentClassifier(config)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X),)

    def test_unfitted_raises(self):
        clf = TabAgentClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict_proba(pd.DataFrame({"intent": ["test"]}))

    def test_unknown_model_raises(self):
        config = ClassifierConfig(model_type="nonexistent")
        clf = TabAgentClassifier(config)
        with pytest.raises(ValueError, match="Unknown model type"):
            clf.fit(pd.DataFrame({"intent": ["x"]}), pd.Series([1]))
