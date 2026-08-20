from __future__ import annotations

from pathlib import Path

from src.api_client import RiotApiClient


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
