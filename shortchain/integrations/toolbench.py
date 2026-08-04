"""ToolBench integration adapter for ShortChain."""

from __future__ import annotations

import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Iterator

from shortchain.ingest.schema import Span, Trajectory
from shortchain.integrations.base import BaseBenchmarkAdapter
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


def normalize_tool_name(name: str) -> str:
    """Normalize a tool/API name into the canonical ``lower_underscore`` form.

    Mirrors the normalization ToolBench uses when building its full API names
    (``<api>_for_<tool>``) that appear in execution traces, e.g.
    ``Simple YouTube Search`` -> ``simple_youtube_search``.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name or "")).strip("_").lower()


def canonical_api_name(api_name: str, tool_name: str) -> str:
    """Reconstruct the canonical ToolBench full-API identifier.

    ToolBench's execution traces reference APIs by their *full* name,
    ``<normalized_api>_for_<normalized_tool>`` (e.g. ``search_for_soundcloud``).
    The ``test_instruction`` records store ``[tool_name, api_name]`` pairs that
    must be expanded back to this canonical form so that seen/unseen auditing
    happens in the same identifier space used during training.
    """
    api = normalize_tool_name(api_name)
    tool = normalize_tool_name(tool_name)
    if not api and not tool:
        return ""
    if not tool:
        return api
    if not api:
        return tool
    # Guard against api_name already being a full composite name.
    if "_for_" in api and api not in tool:
        return api
    return f"{api}_for_{tool}"


# The official ToolBench test-instruction subsets. The first three are the
# "unseen" G1/G2 tool & category generalization sets; the last three test
# instruction-level generalization over seen tool sets.
_TEST_INSTRUCTION_SUBSETS = {
    "G1_tool",
    "G1_category",
    "G2_category",
    "G1_instruction",
    "G2_instruction",
    "G3_instruction",
}

_TEST_INSTRUCTION_SUBSETS_ORDER = [
    "G1_tool",
    "G1_category",
    "G2_category",
    "G1_instruction",
    "G2_instruction",
    "G3_instruction",
]


class ToolBenchAdapter(BaseBenchmarkAdapter):
    """Adapter for loading ToolBench dataset and auditing split compliance."""

    def __init__(self, field_mapping: dict[str, str] | None = None) -> None:
        self.field_mapping = field_mapping or {
            "task_id": "task_id",
            "intent": "query",
            "app_name": "category",
            "steps": "steps",
            "action": "action",
        }

    def load_trajectories(
        self,
        path: str | Path,
        sample_size: int | None = 10000,
        random_state: int | None = None,
    ) -> list[Trajectory]:
        """Load ToolBench trajectories from a JSON file, directory, or data.zip archive.

        Supports standard ToolBench execution trace formats (ToolLlama DFS JSON, JSONL).

        Parameters
        ----------
        path
            File / directory / zip archive path.
        sample_size
            Maximum number of trajectories to return.
        random_state
            If provided, a fixed-size *uniform random sample* (reservoir sampling)
            is drawn instead of the first ``sample_size`` records. This avoids the
            ordering bias of truncating a very large file (default behaviour, kept
            for backward compatibility, uses ``None``).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ToolBench path does not exist: {path}")

        trajectories: list[Trajectory] = []

        if path.is_file() and path.suffix == ".zip":
            trajectories = self._load_from_zip(
                path, sample_size=sample_size, random_state=random_state
            )
        elif path.is_file():
            trajectories = self._load_single_file(
                path, limit=sample_size, random_state=random_state
            )
        else:
            zip_files = list(path.glob("*.zip"))
            train_json = path / "data" / "toolllama_G123_dfs_train.json"
            if not train_json.exists():
                train_json = path / "toolllama_G123_dfs_train.json"

            if train_json.exists():
                trajectories = self._load_single_file(
                    train_json, limit=sample_size, random_state=random_state
                )
            elif zip_files:
                trajectories = self._load_from_zip(
                    zip_files[0], sample_size=sample_size, random_state=random_state
                )
            else:
                json_files = list(path.glob("*.json")) + list(path.glob("*.jsonl"))
                for f in json_files:
                    trajectories.extend(
                        self._load_single_file(f, limit=sample_size, random_state=random_state)
                    )
                    if sample_size and len(trajectories) >= sample_size:
                        trajectories = trajectories[:sample_size]
                        break

        log.info(f"Loaded {len(trajectories)} ToolBench trajectories from {path}")
        return trajectories

    def load_eval_tasks(
        self,
        path: str | Path,
        subsets: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Load faithful evaluation tasks from ToolBench ``test_instruction`` files.

        This is the primary, honest evaluation constructor. Each returned task is
        a dict::

            {
                "task_id": "...",            # unique, includes source prefix
                "query": "...",              # user instruction (context intent)
                "app_name": "...",           # dominant candidate category
                "source": "G1_tool",         # which subset a task comes from
                "original_query_id": "...",
                "candidates": [              # candidate pool = the query's api_list
                    {"tool_name": "<canonical full api name>",
                     "tool_description": "<api_description>",
                     "tool_label": "<api_list tool_name>",
                     "api_name": "<api_list api_name>",
                     "category_name": "..."},
                ],
                "relevant_tools": set(...),  # canonical ids that are ground truth
                "relevant_missing": int,     # relevant APIs not found in api_list
            }

        The candidate pool is the record's own ``api_list`` (the faithful ToolBench
        protocol) and NOT negative sampling, so evaluation measures ranking over
        genuinely available tools.
        """
        path = Path(path)
        subsets = subsets or list(_TEST_INSTRUCTION_SUBSETS)
        requested = {s for s in (subsets or []) if s in _TEST_INSTRUCTION_SUBSETS}
        tasks: list[dict[str, Any]] = []

        if not path.exists():
            raise FileNotFoundError(f"ToolBench path does not exist: {path}")

        if path.is_file() and path.suffix == ".zip":
            with zipfile.ZipFile(path, "r") as z:
                for subset in _TEST_INSTRUCTION_SUBSETS_ORDER:
                    if subset not in requested:
                        continue
                    entry = f"data/test_instruction/{subset}.json"
                    if entry not in z.namelist():
                        continue
                    with z.open(entry) as f:
                        files_tasks = self._parse_eval_instruction_data(
                            json.load(f), subset
                        )
                        tasks.extend(files_tasks)
        else:
            test_dir = path if path.is_dir() else path.parent
            for subset in _TEST_INSTRUCTION_SUBSETS_ORDER:
                if subset not in requested:
                    continue
                test_file = test_dir / "data" / "test_instruction" / f"{subset}.json"
                if not test_file.exists():
                    test_file = test_dir / "test_instruction" / f"{subset}.json"
                if not test_file.exists():
                    continue
                with open(test_file, "r", encoding="utf-8") as f:
                    files_tasks = self._parse_eval_instruction_data(
                        json.load(f), subset
                    )
                    tasks.extend(files_tasks)

        log.info(f"Loaded {len(tasks)} ToolBench evaluation tasks (subsets={sorted(requested)})")
        return tasks

    @staticmethod
    def _parse_eval_instruction_data(
        data: Any, source: str
    ) -> list[dict[str, Any]]:
        """Build eval-task dicts from the parsed contents of one test_instruction file."""
        records = data if isinstance(data, list) else [data]
        tasks: list[dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            task = ToolBenchAdapter._build_eval_task(rec, source)
            if task is not None:
                tasks.append(task)
        return tasks

    @staticmethod
    def _build_eval_task(record: dict[str, Any], source: str) -> dict[str, Any] | None:
        query = str(record.get("query") or record.get("instruction") or "").strip()
        query_id = str(record.get("query_id") or record.get("id") or "")
        if not query:
            return None

        api_list = record.get("api_list") or []
        candidates: list[dict[str, Any]] = []
        category_counter: dict[str, int] = {}
        for entry in api_list:
            if not isinstance(entry, dict):
                continue
            tool_label = str(entry.get("tool_name") or "")
            api_name = str(entry.get("api_name") or "")
            tool_desc = str(entry.get("api_description") or "")
            category = str(entry.get("category_name") or "")
            canonical = canonical_api_name(api_name, tool_label)
            if not canonical:
                continue
            candidates.append(
                {
                    "tool_name": canonical,
                    "tool_description": tool_desc,
                    "tool_label": tool_label,
                    "api_name": api_name,
                    "category_name": category,
                }
            )
            if category:
                category_counter[category] = category_counter.get(category, 0) + 1

        candidate_ids = {c["tool_name"] for c in candidates}
        relevant: set[str] = set()
        missing = 0
        for item in record.get("relevant APIs") or record.get("relevant_apis") or []:
            if isinstance(item, list) and len(item) >= 2:
                canonical = canonical_api_name(str(item[1]), str(item[0]))
            elif isinstance(item, str):
                canonical = normalize_tool_name(item)
            else:
                continue
            if not canonical:
                continue
            if canonical in candidate_ids:
                relevant.add(canonical)
            else:
                missing += 1

        if not candidates or not relevant:
            return None

        app_name = (
            max(category_counter, key=category_counter.get)
            if category_counter
            else ""
        )

        return {
            "task_id": f"{source}:{query_id}",
            "query": query,
            "app_name": app_name,
            "source": source,
            "original_query_id": query_id,
            "candidates": candidates,
            "relevant_tools": relevant,
            "relevant_missing": missing,
            "n_candidates": len(candidates),
            "n_relevant": len(relevant),
        }

    def load_eval_trajectories(self, path: str | Path, eval_size: int = 2000) -> list[Trajectory]:
        """Load evaluation tasks from ToolBench test instructions or answer traces."""
        path = Path(path)
        eval_trajs: list[Trajectory] = []

        if path.is_file() and path.suffix == ".zip":
            eval_trajs = self._load_eval_from_zip(path, eval_size=eval_size)
        else:
            test_dir = path / "data" / "test_instruction" if (path / "data" / "test_instruction").exists() else path / "test_instruction"
            if test_dir.exists():
                for test_file in sorted(test_dir.glob("*.json")):
                    eval_trajs.extend(self._parse_test_instruction_file(test_file))
            
            if len(eval_trajs) < eval_size:
                eval_json = path / "data" / "toolllama_G123_dfs_eval.json"
                if eval_json.exists():
                    more = self._load_single_file(eval_json, limit=eval_size - len(eval_trajs))
                    eval_trajs.extend(more)

        if eval_size and len(eval_trajs) > eval_size:
            eval_trajs = eval_trajs[:eval_size]

        log.info(f"Loaded {len(eval_trajs)} ToolBench evaluation trajectories from {path}")
        return eval_trajs

    def sanitize_splits(
        self,
        train_trajectories: list[Trajectory],
        test_trajectories: list[Trajectory],
    ) -> tuple[list[Trajectory], list[Trajectory]]:
        """Filter out any training trajectories that leak task IDs or intent queries into test set."""
        test_task_ids = {t.task_id for t in test_trajectories}
        test_intents = {t.intent.strip().lower() for t in test_trajectories if t.intent}

        clean_train = [
            t for t in train_trajectories
            if t.task_id not in test_task_ids and t.intent.strip().lower() not in test_intents
        ]
        
        removed = len(train_trajectories) - len(clean_train)
        if removed > 0:
            log.info(f"Sanitized training split: purged {removed} leaking trajectories to guarantee ShortChain group compliance.")
        
        return clean_train, test_trajectories

    @staticmethod
    def _iter_json_objects(stream) -> Iterator[dict[str, Any]]:
        """Stream JSON objects from a (possibly huge) JSON array without loading it all.

        Yields one decoded object at a time, tolerant of array delimiters and
        multi-megabyte records.
        """
        decoder = json.JSONDecoder()
        buffer = ""
        chunk_size = 16 * 1024 * 1024
        in_array = False
        idx = 0

        while True:
            chunk = stream.read(chunk_size)
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="ignore")
            if not chunk:
                break
            buffer += chunk
            if not in_array:
                start_pos = buffer.find("[")
                if start_pos == -1:
                    continue
                buffer = buffer[start_pos + 1:]
                in_array = True

            while idx < len(buffer):
                while idx < len(buffer) and buffer[idx] in " \t\r\n,":
                    idx += 1
                if idx >= len(buffer):
                    break
                if buffer[idx] == "]":
                    return
                try:
                    obj, end_pos = decoder.raw_decode(buffer[idx:])
                    yield obj
                    idx += end_pos
                except json.JSONDecodeError:
                    buffer = buffer[idx:]
                    idx = 0
                    break

    @staticmethod
    def _reservoir_sample(
        records: Iterator[Any], sample_size: int, random_state: int | None
    ) -> list[Any]:
        """Algorithm-R reservoir sampling over an iterator of raw records."""
        rng = random.Random(random_state)
        reservoir: list[Any] = []
        for idx, obj in enumerate(records):
            if len(reservoir) < sample_size:
                reservoir.append(obj)
            else:
                j = rng.randint(0, idx)
                if j < sample_size:
                    reservoir[j] = obj
        return reservoir

    def _load_from_zip(
        self,
        zip_path: Path,
        sample_size: int | None = 10000,
        random_state: int | None = None,
    ) -> list[Trajectory]:
        """Load trajectories directly from zip archive without full extraction."""
        trajectories: list[Trajectory] = []
        with zipfile.ZipFile(zip_path, "r") as z:
            train_name = "data/toolllama_G123_dfs_train.json"
            if train_name in z.namelist():
                with z.open(train_name) as f:
                    if random_state is not None and sample_size:
                        raw = self._reservoir_sample(
                            self._iter_json_objects(f), sample_size, random_state
                        )
                        for i, obj in enumerate(raw):
                            traj = self._parse_record(obj, fallback_id=f"train_{i}")
                            if traj:
                                trajectories.append(traj)
                    elif sample_size is not None:
                        for i, obj in enumerate(self._iter_json_objects(f)):
                            if len(trajectories) >= sample_size:
                                break
                            traj = self._parse_record(obj, fallback_id=f"train_{i}")
                            if traj:
                                trajectories.append(traj)
                    else:
                        for i, obj in enumerate(self._iter_json_objects(f)):
                            traj = self._parse_record(obj, fallback_id=f"train_{i}")
                            if traj:
                                trajectories.append(traj)

        return trajectories

    def _load_eval_from_zip(self, zip_path: Path, eval_size: int = 2000) -> list[Trajectory]:
        """Load test instruction files from zip archive."""
        eval_trajs: list[Trajectory] = []
        with zipfile.ZipFile(zip_path, "r") as z:
            test_files = [n for n in z.namelist() if "test_instruction/" in n and n.endswith(".json") and "__MACOSX" not in n]
            for test_file in sorted(test_files):
                with z.open(test_file) as f:
                    data = json.load(f)
                    records = data if isinstance(data, list) else [data]
                    for rec in records:
                        traj = self._parse_test_instruction_record(rec)
                        if traj:
                            eval_trajs.append(traj)

            if len(eval_trajs) < eval_size and "data/toolllama_G123_dfs_eval.json" in z.namelist():
                with z.open("data/toolllama_G123_dfs_eval.json") as f:
                    data = json.load(f)
                    for rec in data:
                        traj = self._parse_record(rec, fallback_id=f"eval_{len(eval_trajs)}")
                        if traj:
                            eval_trajs.append(traj)
                        if len(eval_trajs) >= eval_size:
                            break

        return eval_trajs[:eval_size]

    def _load_single_file(
        self,
        file_path: Path,
        limit: int | None = None,
        random_state: int | None = None,
    ) -> list[Trajectory]:
        """Parse a single ToolBench JSON or JSONL file."""
        trajectories: list[Trajectory] = []
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix == ".jsonl":
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        traj = self._parse_record(record, fallback_id=f"{file_path.stem}_{idx}")
                        if traj:
                            trajectories.append(traj)
                            if limit and len(trajectories) >= limit:
                                break
                    except Exception as err:
                        log.warning(f"Skipping line {idx} in {file_path}: {err}")
            elif random_state is not None and limit:
                raw = list(self._iter_json_objects(f))
                sampled = self._reservoir_sample(raw, limit, random_state)
                for i, record in enumerate(sampled):
                    traj = self._parse_record(record, fallback_id=f"{file_path.stem}_{i}")
                    if traj:
                        trajectories.append(traj)
            else:
                try:
                    data = json.load(f)
                    records = data if isinstance(data, list) else [data]
                    for idx, record in enumerate(records):
                        traj = self._parse_record(record, fallback_id=f"{file_path.stem}_{idx}")
                        if traj:
                            trajectories.append(traj)
                            if limit and len(trajectories) >= limit:
                                break
                except Exception as err:
                    log.warning(f"Error loading {file_path}: {err}")

        return trajectories

    def _parse_test_instruction_file(self, file_path: Path) -> list[Trajectory]:
        """Parse test_instruction files like G1_tool.json."""
        trajs: list[Trajectory] = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            records = data if isinstance(data, list) else [data]
            for rec in records:
                traj = self._parse_test_instruction_record(rec)
                if traj:
                    trajs.append(traj)
        return trajs

    def _parse_test_instruction_record(self, record: dict[str, Any]) -> Trajectory | None:
        """Parse test instruction record containing api_list and relevant APIs."""
        query_id = str(record.get("query_id") or record.get("id") or "")
        intent = str(record.get("query") or record.get("instruction") or "")
        if not intent:
            return None

        relevant_apis = record.get("relevant APIs") or record.get("relevant_apis") or []
        spans: list[Span] = []

        for item in relevant_apis:
            if isinstance(item, list) and len(item) >= 2:
                action_name = canonical_api_name(str(item[1]), str(item[0]))
                if action_name:
                    spans.append(Span(action=action_name, agent_name="ToolBenchAgent"))
            elif isinstance(item, str):
                spans.append(Span(action=item, agent_name="ToolBenchAgent"))

        app_name = "ToolBench"
        api_list = record.get("api_list") or []
        if isinstance(api_list, list) and len(api_list) > 0 and isinstance(api_list[0], dict):
            app_name = str(api_list[0].get("category_name") or "ToolBench")

        return Trajectory(
            task_id=f"test_inst_{query_id}",
            intent=intent,
            spans=spans,
            success=True,
            app_name=app_name,
            metadata={"api_list": api_list, "source": "test_instruction"},
        )

    def _parse_record(self, record: dict[str, Any], fallback_id: str) -> Trajectory | None:
        """Parse a ToolBench dictionary record into a ShortChain Trajectory."""
        task_id = str(record.get("task_id") or record.get("query_id") or record.get("id") or fallback_id)
        intent = str(record.get("query") or record.get("intent") or record.get("instruction") or "")

        raw_steps = record.get("steps") or record.get("conversations") or record.get("action_graph") or []
        spans: list[Span] = []
        app_name = str(record.get("category") or record.get("category_name") or record.get("app_name") or "ToolBench")

        if isinstance(raw_steps, list):
            for step_idx, raw_step in enumerate(raw_steps):
                if isinstance(raw_step, dict):
                    role = raw_step.get("from") or raw_step.get("role")
                    val = str(raw_step.get("value") or raw_step.get("content") or "")

                    if role == "user" and not intent:
                        intent = val

                    action = raw_step.get("action") or raw_step.get("api_name") or raw_step.get("tool_name")
                    if not action and ("Action:" in val):
                        action = val.split("Action:")[1].split("\n")[0].strip()

                    thoughts = str(raw_step.get("thoughts") or raw_step.get("thought") or "")
                    if not thoughts and "Thought:" in val:
                        thoughts = val.split("Thought:")[1].split("Action:")[0].strip()

                    observation = str(raw_step.get("observation") or raw_step.get("result") or "")

                    if action and action.lower() != "finish":
                        spans.append(Span(
                            action=str(action),
                            thoughts=thoughts,
                            observation=observation,
                            agent_name=str(raw_step.get("agent", "ToolBenchAgent")),
                        ))
                elif isinstance(raw_step, str):
                    if raw_step.lower() != "finish":
                        spans.append(Span(action=raw_step, agent_name="ToolBenchAgent"))

        if not intent:
            record_id = str(record.get("id", ""))
            if len(record_id) > 20 and (" " in record_id):
                intent = record_id.split("Step")[0].strip() if "Step" in record_id else record_id

        if not intent:
            return None

        success = bool(record.get("success", record.get("final_answer") is not None or len(spans) > 0))

        return Trajectory(
            task_id=task_id,
            intent=intent,
            spans=spans,
            success=success,
            app_name=app_name,
            metadata={"source": "toolbench", "raw_keys": list(record.keys())},
        )

    def load_catalog(self, path: str | Path) -> dict[str, str]:
        """Load ToolBench API tool catalog."""
        path = Path(path)
        catalog: dict[str, str] = {}
        if not path.exists():
            return catalog

        if path.is_file() and path.suffix == ".zip":
            with zipfile.ZipFile(path, "r") as z:
                tool_files = [n for n in z.namelist() if "toolenv/tools/" in n and n.endswith(".json") and "__MACOSX" not in n]
                for tf in tool_files:
                    try:
                        with z.open(tf) as f:
                            data = json.load(f)
                            self._extract_catalog_items(data, catalog)
                    except Exception:
                        pass
        elif path.is_dir():
            for json_file in path.glob("**/*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._extract_catalog_items(data, catalog)
                except Exception:
                    pass

        return catalog

    def _extract_catalog_items(self, data: Any, catalog: dict[str, str]) -> None:
        """Extract tool names and descriptions into catalog dict."""
        if isinstance(data, dict):
            for tool_name, info in data.items():
                if isinstance(info, str):
                    catalog[tool_name] = info
                elif isinstance(info, dict):
                    desc = info.get("description") or info.get("api_description") or json.dumps(info)
                    catalog[tool_name] = str(desc)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    name = item.get("tool_name") or item.get("api_name") or item.get("name")
                    if name:
                        desc = str(item.get("description") or item.get("api_description") or "")
                        catalog[str(name)] = desc

    def audit_eval_task_unseenness(
        self,
        train_trajectories: list[Trajectory],
        eval_tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Audit how many evaluation tasks strictly use tools never seen in training.

        For each eval task, its *relevant* tools are required to be absent from the
        training corpus (the ToolBench G1 "unseen tools" setting). The audit also
        reports per-subset breakdowns so that genuinely-unseen subsets (G1_tool,
        G1_category, G2_category) are distinguished from instruction-level subsets
        that legitimately reuse seen tools.
        """
        train_tools: set[str] = set()
        for t in train_trajectories:
            train_tools.update(t.tools_used)

        per_subset: dict[str, dict[str, int]] = {}
        all_unseen_tasks = 0
        any_unseen_tasks = 0
        valid_tasks = 0

        for task in eval_tasks:
            relevant = task.get("relevant_tools") or set()
            if not relevant:
                continue
            valid_tasks += 1
            unseen = {r for r in relevant if r not in train_tools}
            source = task.get("source", "?")
            bucket = per_subset.setdefault(
                source, {"valid": 0, "all_unseen": 0, "any_unseen": 0, "some_seen": 0}
            )
            bucket["valid"] += 1
            # A task is strictly unseen only if EVERY relevant tool is unseen.
            if len(unseen) == len(relevant):
                bucket["all_unseen"] += 1
                all_unseen_tasks += 1
            if unseen:
                bucket["any_unseen"] += 1
                any_unseen_tasks += 1
            else:
                bucket["some_seen"] += 1

        ratio = all_unseen_tasks / max(1, valid_tasks)
        report = {
            "train_unique_tools": len(train_tools),
            "valid_tasks": valid_tasks,
            "strictly_unseen_tasks": all_unseen_tasks,
            "unseen_task_ratio": ratio,
            "per_subset": per_subset,
            "verdict": (
                "PASSED: a clean unseen-tools eval subset exists."
                if all_unseen_tasks >= 400
                else "INSUFFICIENT: fewer than 400 strictly-unseen tasks found."
            ),
        }
        log.info(
            f"Unseen audit: {all_unseen_tasks}/{valid_tasks} tasks have all-relevant "
            f"tools unseen in training ({ratio:.1%})."
        )
        return report

    def audit_split_compliance(
        self,
        train_trajectories: list[Trajectory],
        test_trajectories: list[Trajectory],
    ) -> dict[str, Any]:
        """Audit train and test trajectories for compliance with ShortChain principles.

        ShortChain Principles checked:
        1. Task ID Disjointness: No task_id in train should appear in test.
        2. Query / Intent Leakage: No exact user intent in test should exist in train.
        3. Span Integrity: No spans from the same task are split across datasets.
        4. Tool Overlap Analysis: Tracks Seen tools vs. Unseen tools / categories in test.
        """
        train_task_ids = {t.task_id for t in train_trajectories}
        test_task_ids = {t.task_id for t in test_trajectories}
        task_id_leakage = train_task_ids.intersection(test_task_ids)

        train_intents = {t.intent.strip().lower() for t in train_trajectories if t.intent}
        test_intents = {t.intent.strip().lower() for t in test_trajectories if t.intent}
        intent_leakage = train_intents.intersection(test_intents)

        train_tools: set[str] = set()
        for t in train_trajectories:
            train_tools.update(t.tools_used)

        test_tools: set[str] = set()
        for t in test_trajectories:
            test_tools.update(t.tools_used)

        seen_tools_in_test = test_tools.intersection(train_tools)
        unseen_tools_in_test = test_tools - train_tools

        is_compliant = (len(task_id_leakage) == 0) and (len(intent_leakage) == 0)

        report = {
            "compliant": is_compliant,
            "train_size": len(train_trajectories),
            "test_size": len(test_trajectories),
            "train_task_count": len(train_task_ids),
            "test_task_count": len(test_task_ids),
            "task_id_leakage_count": len(task_id_leakage),
            "task_id_leakage_samples": list(task_id_leakage)[:5],
            "intent_leakage_count": len(intent_leakage),
            "intent_leakage_samples": list(intent_leakage)[:5],
            "train_unique_tools": len(train_tools),
            "test_unique_tools": len(test_tools),
            "seen_tools_in_test_count": len(seen_tools_in_test),
            "unseen_tools_in_test_count": len(unseen_tools_in_test),
            "seen_tool_ratio": len(seen_tools_in_test) / max(1, len(test_tools)),
            "shortchain_principle_verdict": (
                "PASSED: Splits are group-disjoint with zero task leakage."
                if is_compliant
                else "FAILED: Task or query leakage detected between train and test splits."
            ),
        }

        log.info(
            f"Split Audit Result: Compliant={is_compliant} | "
            f"Task Leakage={len(task_id_leakage)} | Intent Leakage={len(intent_leakage)} | "
            f"Seen Tools={len(seen_tools_in_test)} | Unseen Tools={len(unseen_tools_in_test)}"
        )
        return report
