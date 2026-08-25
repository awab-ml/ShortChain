"""Tests for the AppWorld API-spec loader (P2)."""

from __future__ import annotations

import json
from pathlib import Path


from shortchain.adapters.appworld_api import (
    ParamSpec,
    build_catalog_and_schemas,
    load_appworld_api_spec,
    coverage_report,
)


def _write_fc_dir(tmp_path: Path) -> Path:
    fc = tmp_path / "function_calling"
    fc.mkdir()
    spotify = [
        {"type": "function", "function": {
            "name": "spotify__login",
            "description": "Log into Spotify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {"type": "string", "description": "email"},
                    "scope": {"type": "string", "enum": ["user", "premium"]},
                },
            },
        }},
        {"type": "function", "function": {
            "name": "spotify__show_song",
            "description": "Show a song by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_id": {"type": "integer", "description": "song id"},
                    "include_recommendations": {"type": "boolean"},
                    "artists": {"type": "array", "items": {"type": "string"}},
                },
            },
        }},
        {"type": "function", "function": {
            "name": "spotify__logout",
            "description": "Log out.",
            "parameters": {"type": "object", "properties": {}},
        }},
    ]
    (fc / "spotify.json").write_text(json.dumps(spotify))
    # helper/control apps must be excluded
    (fc / "supervisor.json").write_text(json.dumps([
        {"type": "function", "function": {"name": "supervisor__complete_task",
                                           "description": "", "parameters": {"type": "object"}}}
    ]))
    (fc / "api_docs.json").write_text(json.dumps([
        {"type": "function", "function": {"name": "api_docs__show_doc",
                                           "description": "", "parameters": {"type": "object"}}}
    ]))
    return fc


class TestAppWorldApiLoader:
    def test_parse_and_exclude_helpers(self, tmp_path):
        fc = _write_fc_dir(tmp_path)
        specs = load_appworld_api_spec(fc)
        assert set(specs) == {"spotify__login", "spotify__show_song", "spotify__logout"}
        login = specs["spotify__login"]
        assert login.description == "Log into Spotify."
        assert login.n_params == 2
        scope = [p for p in login.parameters if p.name == "scope"][0]
        assert scope.enum == ("user", "premium")

    def test_candidate_text(self, tmp_path):
        fc = _write_fc_dir(tmp_path)
        specs = load_appworld_api_spec(fc)
        txt = specs["spotify__show_song"].candidate_text()
        assert "spotify__show_song" in txt
        assert "song_id (integer)" in txt  # argument hint
        assert "include_recommendations (boolean)" in txt
        assert "artists (array)" in txt

    def test_catalog_and_schemas_coverage(self, tmp_path):
        fc = _write_fc_dir(tmp_path)
        names = {"spotify__login", "spotify__show_song", "spotify__logout", "unknown__tool"}
        catalog, schemas = build_catalog_and_schemas(fc, names)
        assert set(schemas) == {"spotify__login", "spotify__show_song", "spotify__logout"}
        # unknown tool: empty description, no schema
        assert catalog["unknown__tool"] == ""
        assert "unknown__tool" not in schemas
        # catalog entries carry the enriched candidate text
        assert "Log into Spotify." in catalog["spotify__login"]
        # no fc_dir -> name-only catalog, empty schemas
        c2, s2 = build_catalog_and_schemas(None, names)
        assert s2 == {}
        assert c2["spotify__login"] == ""

    def test_coverage_report(self, tmp_path):
        fc = _write_fc_dir(tmp_path)
        specs = load_appworld_api_spec(fc)
        rep = coverage_report(specs)
        assert rep["n_tools"] == 3
        assert rep["n_params_total"] == 5  # 2 + 3 + 0

    def test_param_items_and_enum(self):
        p_arr = ParamSpec(name="x", type="array", items_type="string")
        assert p_arr.items_type == "string"
        p_enum = ParamSpec(name="y", type="string", enum=("a", "b"))
        assert p_enum.enum == ("a", "b")
