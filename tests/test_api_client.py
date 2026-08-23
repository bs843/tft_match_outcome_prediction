from __future__ import annotations

from pathlib import Path

import pytest

from src.api_client import RiotApiClient, RiotAuthenticationError


class FakeResponse:
    def __init__(self, status: int, payload, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.headers = {}

    def get(self, *args, **kwargs):
        return next(self.responses)

    def close(self):
        return None


def api_config(log_path: Path) -> dict:
    return {
        "api": {
            "key_env": "RIOT_API_KEY", "timeout_seconds": 1, "max_retries": 1,
            "base_backoff_seconds": 0, "fallback_requests_per_window": 100,
            "fallback_window_seconds": 1, "minimum_interval_seconds": 0,
        },
        "paths": {"request_log": str(log_path)},
    }


def test_request_log_redacts_puuid(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RIOT_API_KEY", "test-key-never-logged")
    secret_puuid = "private-puuid-value"
    session = FakeSession([FakeResponse(200, ["ME1_1"])])
    client = RiotApiClient(api_config(tmp_path / "requests.csv"), session=session, sleeper=lambda _: None)
    assert client.get_match_ids("EUROPE", secret_puuid) == ["ME1_1"]
    logged = (tmp_path / "requests.csv").read_text(encoding="utf-8")
    assert secret_puuid not in logged
    assert "test-key-never-logged" not in logged
    assert "{puuid}" in logged


def test_429_is_retried_after_global_block(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RIOT_API_KEY", "test-key")
    session = FakeSession([
        FakeResponse(429, {}, {"Retry-After": "1", "X-Rate-Limit-Type": "application"}),
        FakeResponse(200, {"metadata": {}, "info": {}}),
    ])
    client = RiotApiClient(api_config(tmp_path / "requests.csv"), session=session, sleeper=lambda _: None)
    assert client.get_match("EUROPE", "ME1_1")["info"] == {}
    assert "rate_limit" in (tmp_path / "requests.csv").read_text(encoding="utf-8")


def test_expired_key_pauses_loads_changed_key_and_retries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RIOT_API_KEY", "expired-key")
    config = api_config(tmp_path / "requests.csv")
    config["api"].update({
        "max_retries": 2,
        "authentication_recovery_enabled": True,
        "authentication_poll_seconds": 1,
        "authentication_max_wait_seconds": 2,
    })
    session = FakeSession([
        FakeResponse(403, {}),
        FakeResponse(200, ["EUW1_1"]),
    ])
    client = RiotApiClient(
        config, session=session, sleeper=lambda _: None,
        credential_loader=lambda: "replacement-key",
    )
    assert client.get_match_ids("EUROPE", "private-puuid") == ["EUW1_1"]
    assert session.headers["X-Riot-Token"] == "replacement-key"
    logged = (tmp_path / "requests.csv").read_text(encoding="utf-8")
    assert "expired-key" not in logged
    assert "replacement-key" not in logged


def test_expired_key_still_fails_when_recovery_wait_expires(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RIOT_API_KEY", "unchanged-key")
    config = api_config(tmp_path / "requests.csv")
    config["api"].update({
        "authentication_recovery_enabled": True,
        "authentication_poll_seconds": 1,
        "authentication_max_wait_seconds": 1,
    })
    client = RiotApiClient(
        config, session=FakeSession([FakeResponse(403, {})]), sleeper=lambda _: None,
        credential_loader=lambda: "unchanged-key",
    )
    with pytest.raises(RiotAuthenticationError):
        client.get_match_ids("EUROPE", "private-puuid")
