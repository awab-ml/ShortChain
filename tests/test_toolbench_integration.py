"""Tests for ToolBench benchmark integration and split audit."""

import json
from pathlib import Path
import pytest

from shortchain.ingest.schema import Span, Trajectory
from shortchain.integrations.toolbench import (
    ToolBenchAdapter,
    canonical_api_name,
    normalize_tool_name,
)
from shortchain.dataset.builder import DatasetBuilder
from shortchain.head.classifier import ShortChainClassifier
from shortchain.evaluation.metrics import compute_metrics


@pytest.fixture
def sample_toolbench_trajectories() -> list[Trajectory]:
    return [
        Trajectory(
            task_id="tb_task_001",
            intent="Search for weather forecast in San Francisco",
            app_name="WeatherAPI",
            success=True,
            spans=[
                Span(action="get_current_weather", thoughts="Looking up SF weather"),
                Span(action="get_weekly_forecast", thoughts="Getting 7 day forecast"),
            ],
        ),
        Trajectory(
            task_id="tb_task_002",
            intent="Convert 100 USD to JPY and calculate total",
            app_name="FinanceAPI",
            success=True,
            spans=[
                Span(action="convert_currency", thoughts="Converting USD to JPY"),
            ],
        ),
        Trajectory(
            task_id="tb_task_003",
            intent="Find flights from NYC to London for tomorrow",
            app_name="TravelAPI",
            success=True,
            spans=[
                Span(action="search_flights", thoughts="Searching flight availability"),
            ],
        ),
    ]


@pytest.fixture
def sample_test_trajectories() -> list[Trajectory]:
    return [
        Trajectory(
            task_id="tb_task_004",
            intent="Get restaurant recommendations in Chicago",
            app_name="DiningAPI",
            success=True,
            spans=[
                Span(action="search_restaurants", thoughts="Finding Chicago spots"),
            ],
        ),
    ]


class TestToolBenchAdapter:
    def test_load_json_file(self, tmp_path: Path):
        data = [
            {
                "task_id": "tb_101",
                "query": "Book a table for 2",
                "category": "Dining",
                "steps": [
                    {"action": "reserve_table", "thoughts": "Booking..."}
                ]
            }
        ]
        json_file = tmp_path / "toolbench_sample.json"
        json_file.write_text(json.dumps(data))

        adapter = ToolBenchAdapter()
        trajs = adapter.load_trajectories(json_file)
        assert len(trajs) == 1
        assert trajs[0].task_id == "tb_101"
        assert trajs[0].intent == "Book a table for 2"
        assert trajs[0].app_name == "Dining"
        assert "reserve_table" in trajs[0].tools_used

    def test_audit_split_compliance_compliant(self, sample_toolbench_trajectories, sample_test_trajectories):
        adapter = ToolBenchAdapter()
        report = adapter.audit_split_compliance(sample_toolbench_trajectories, sample_test_trajectories)
        assert report["compliant"] is True
        assert report["task_id_leakage_count"] == 0
        assert report["intent_leakage_count"] == 0
        assert report["train_size"] == 3
        assert report["test_size"] == 1

    def test_audit_split_compliance_leaking(self, sample_toolbench_trajectories):
        adapter = ToolBenchAdapter()
        # Train and test share task_001
        train_trajs = sample_toolbench_trajectories
        test_trajs = [sample_toolbench_trajectories[0]]

        report = adapter.audit_split_compliance(train_trajs, test_trajs)
        assert report["compliant"] is False
        assert report["task_id_leakage_count"] == 1
        assert report["intent_leakage_count"] == 1

    def test_end_to_end_toolbench_pipeline(self, sample_toolbench_trajectories, sample_test_trajectories):
        builder = DatasetBuilder()
        train_df = builder.build(sample_toolbench_trajectories)
        test_df = builder.build(sample_test_trajectories)

        assert "intent" in train_df.columns
        assert "label" in train_df.columns
        assert len(train_df) > 0
        assert len(test_df) > 0

        from shortchain.config import ClassifierConfig
        clf = ShortChainClassifier(ClassifierConfig(model_type="random_forest"))
        clf.fit(train_df.drop(columns=["label"]), train_df["label"])

        probas = clf.predict_proba(test_df.drop(columns=["label"]))
        assert len(probas) == len(test_df)

        metrics = compute_metrics(test_df["label"].values, probas, X_val=test_df, k_values=[1, 3])
        assert "r_precision" in metrics
        assert "recall_at_1" in metrics


class TestCanonicalNaming:
    def test_normalize_tool_name(self):
        assert normalize_tool_name("Simple YouTube Search") == "simple_youtube_search"
        assert normalize_tool_name("  Temu.com Shopping API (Realtime) ") == "temu_com_shopping_api_realtime"
        assert normalize_tool_name("") == ""

    def test_canonical_api_name(self):
        # Matches the full-API naming used in training traces (*_for_*).
        assert canonical_api_name("Search", "Simple YouTube Search") == "search_for_simple_youtube_search"
        assert canonical_api_name("related_videos", "YouTube v3 Alternative") == "related_videos_for_youtube_v3_alternative"
        # Already-composite api names are not doubled.
        assert canonical_api_name("calculate_expenses_for_expense_data", "Expense Data") == "calculate_expenses_for_expense_data"


class TestFaithfulEvalTasks:
    @pytest.fixture
    def eval_fixture(self, tmp_path: Path):
        records = [
            {
                "query_id": "394",
                "query": "I need a YouTube tutorial search.",
                "api_list": [
                    {"tool_name": "Simple YouTube Search", "api_name": "Search",
                     "api_description": "Make youtube search", "category_name": "Data"},
                    {"tool_name": "Simple YouTube Search", "api_name": "Video",
                     "api_description": "Get video info", "category_name": "Data"},
                    {"tool_name": "Simple YouTube Search", "api_name": "Playlist",
                     "api_description": "Get playlist info", "category_name": "Data"},
                ],
                "relevant APIs": [["Simple YouTube Search", "Search"], ["Simple YouTube Search", "Video"]],
            },
            {
                "query_id": "995",
                "query": "Search weather for tomorrow.",
                "api_list": [
                    {"tool_name": "Weather API", "api_name": "forecast",
                     "api_description": "7-day forecast", "category_name": "Weather"},
                ],
                "relevant APIs": [["Weather API", "forecast"]],
            },
        ]
        test_dir = tmp_path / "data" / "test_instruction"
        test_dir.mkdir(parents=True)
        (test_dir / "G1_tool.json").write_text(json.dumps(records))
        return tmp_path

    def test_load_eval_tasks_uses_api_list_pool(self, eval_fixture):
        adapter = ToolBenchAdapter()
        tasks = adapter.load_eval_tasks(eval_fixture, subsets=["G1_tool"])
        assert len(tasks) == 2
        task = tasks[0]
        assert task["source"] == "G1_tool"
        assert len(task["candidates"]) == 3  # candidate pool = api_list, no sampling
        assert task["relevant_tools"] == {
            "search_for_simple_youtube_search",
            "video_for_simple_youtube_search",
        }
        assert task["n_relevant"] == 2
        assert task["relevant_missing"] == 0

    def test_candidate_descriptions_present(self, eval_fixture):
        adapter = ToolBenchAdapter()
        task = adapter.load_eval_tasks(eval_fixture, subsets=["G1_tool"])[0]
        desc = {c["tool_name"]: c["tool_description"] for c in task["candidates"]}
        assert desc["search_for_simple_youtube_search"] == "Make youtube search"


class TestFrozenCorpusStats:
    def test_build_candidates_requires_frozen_stats(self, sample_toolbench_trajectories):
        builder = DatasetBuilder()
        with pytest.raises(ValueError, match="frozen corpus"):
            builder.build_candidates(
                sample_toolbench_trajectories[0],
                [{"tool_name": "x", "tool_description": ""}],
            )

    def test_eval_rows_use_train_stats_not_test(self, sample_toolbench_trajectories):
        # Ranked tool "unseen_api" never appears in train; build train first.
        train_builder = DatasetBuilder()
        train_df = train_builder.build(sample_toolbench_trajectories)
        assert "unseen_api" not in set(train_df["tool_name"])

        eval_builder = DatasetBuilder(corpus_stats=train_builder.corpus_stats)
        evald_traj = Trajectory(
            task_id="tb_task_999",
            intent="Call the unseen api",
            app_name="NewAPI",
            spans=[],
            success=True,
        )
        rows = eval_builder.build_candidates(
            evald_traj,
            [
                {"tool_name": "unseen_api", "tool_description": "does something"},
                {"tool_name": "search_flights", "tool_description": "find flights"},
            ],
            relevant_tools={"unseen_api"},
        )
        assert len(rows) == 2
        row_by_tool = {r["tool_name"]: r for r in rows}
        # Unseen candidate must have zero train-derived frequency (no leak).
        assert row_by_tool["unseen_api"]["tool_frequency"] == 0
        assert row_by_tool["unseen_api"]["label"] == 1
        assert row_by_tool["search_flights"]["label"] == 0
        # Schema columns must match the training DataFrame.
        assert set(rows[0].keys()) >= set(train_df.columns)

    def test_build_second_call_keeps_train_stats(self, sample_toolbench_trajectories):
        builder = DatasetBuilder()
        df1 = builder.build(sample_toolbench_trajectories[:2])
        df2 = builder.build(sample_toolbench_trajectories)
        # corpus stats recomputed on first build only (frozen thereafter)
        tool = "convert_currency"
        assert df1[df1["tool_name"] == tool]["tool_frequency"].max() == \
               df2[df2["tool_name"] == tool]["tool_frequency"].max()


class TestUnseenAudit:
    def test_audit_eval_task_unseenness(self, sample_toolbench_trajectories):
        adapter = ToolBenchAdapter()
        tasks = [
            {"task_id": "t1", "source": "G1_tool", "relevant_tools": {"never_seen_api"}},
            {"task_id": "t2", "source": "G1_tool", "relevant_tools": {"get_current_weather"}},
            {"task_id": "t3", "source": "G1_category", "relevant_tools": {"another_unseen"}},
        ]
        report = adapter.audit_eval_task_unseenness(sample_toolbench_trajectories, tasks)
        assert report["valid_tasks"] == 3
        assert report["strictly_unseen_tasks"] == 2  # t2 uses a seen tool
        assert report["per_subset"]["G1_tool"]["all_unseen"] == 1
        assert report["per_subset"]["G1_category"]["all_unseen"] == 1


class TestReservoirSampling:
    def test_same_seed_is_deterministic(self, tmp_path: Path):
        records = [{"task_id": str(i), "query": f"q{i}", "conversations": [
            {"from": "user", "value": f"q{i}"},
            {"from": "assistant", "value": "Thought: hi\nAction: do_the_thing"},
        ]} for i in range(50)]
        f = tmp_path / "trains.json"
        f.write_text(json.dumps(records))
        adapter = ToolBenchAdapter()
        a = adapter._load_single_file(f, limit=10, random_state=7)
        b = adapter._load_single_file(f, limit=10, random_state=7)
        assert [t.task_id for t in a] == [t.task_id for t in b]
        assert len(a) == 10
        # A different seed draws a different sample.
        c = adapter._load_single_file(f, limit=10, random_state=99)
        assert {t.task_id for t in a} != {t.task_id for t in c}
        # Reservoir sampling must not simply take the first N records.
        first_n = {str(i) for i in range(10)}
        assert {t.task_id for t in a} != first_n

