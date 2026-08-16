"""Tests for the runtime CLI (Section 4.2): workers=1 enforcement."""

from __future__ import annotations

import pytest

from shortchain.runtime.cli import (
    _validate_env_workers,
    _warn_if_workers_gt_1,
)


class TestWorkersGuard:
    def test_workers_gt_1_refused(self):
        with pytest.raises(SystemExit) as exc:
            _warn_if_workers_gt_1(2)
        assert exc.value.code == 2

    def test_workers_one_ok(self):
        _warn_if_workers_gt_1(1)  # no exception

    def test_env_workers_gt_1_refused(self, monkeypatch):
        monkeypatch.setenv("WEB_CONCURRENCY", "3")
        with pytest.raises(SystemExit) as exc:
            _validate_env_workers()
        assert exc.value.code == 2

    def test_env_workers_one_allowed(self, monkeypatch):
        monkeypatch.setenv("WEB_CONCURRENCY", "1")
        _validate_env_workers()  # no exception

    def test_env_unset_allowed(self, monkeypatch):
        monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
        _validate_env_workers()  # no exception