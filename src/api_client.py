from __future__ import annotations

import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

from .config import normalise_route, project_path, required_environment_value
from .utils import append_csv_row, stable_json_hash, utc_now_iso

REQUEST_LOG_FIELDS = [
    "request_datetime", "endpoint_type", "route", "path_template", "http_status",
    "attempt_number", "successful", "response_time_seconds", "retry_after_seconds",
    "rate_limit_type", "response_sha256", "error_class",
]


class RiotAPIError(RuntimeError):
    pass


class RiotNotFoundError(RiotAPIError):
    pass


class RiotAuthenticationError(RiotAPIError):
    pass


class SlidingWindowLimiter:
    """Conservative process-local limiter with response-header adaptation."""

    def __init__(
        self,
        requests_per_window: int,
        window_seconds: float,
        minimum_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.requests_per_window = max(1, requests_per_window)
        self.window_seconds = max(0.1, window_seconds)
        self.minimum_interval = max(0.0, minimum_interval)
        self.clock = clock
        self.sleeper = sleeper
        self.timestamps: deque[float] = deque()
        self.last_request: float | None = None
        self.blocked_until = 0.0

    def acquire(self) -> None:
        now = self.clock()
        if now < self.blocked_until:
            self.sleeper(self.blocked_until - now)
            now = self.clock()
        while self.timestamps and now - self.timestamps[0] >= self.window_seconds:
            self.timestamps.popleft()
        waits = []
        if len(self.timestamps) >= self.requests_per_window:
            waits.append(self.window_seconds - (now - self.timestamps[0]) + 0.05)
        if self.last_request is not None:
            waits.append(self.minimum_interval - (now - self.last_request))
        wait = max([0.0, *waits])
        if wait > 0:
            self.sleeper(wait)
            now = self.clock()
            while self.timestamps and now - self.timestamps[0] >= self.window_seconds:
                self.timestamps.popleft()
        self.timestamps.append(now)
        self.last_request = now

    def block(self, seconds: float) -> None:
        self.blocked_until = max(self.blocked_until, self.clock() + max(0.0, seconds))

    def observe_headers(self, headers: requests.structures.CaseInsensitiveDict[str]) -> None:
        """Block before a known application/method window is exhausted."""
        waits: list[float] = []
        for prefix in ("X-App-Rate-Limit", "X-Method-Rate-Limit"):
            limits = _parse_rate_header(headers.get(prefix, ""))
            counts = dict(_parse_rate_header(headers.get(f"{prefix}-Count", "")))
            for limit, window in limits:
                if counts.get(window, 0) >= max(1, limit - 1):
                    waits.append(float(window) + 0.05)
        if waits:
            self.block(max(waits))


def _parse_rate_header(value: str) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    for component in value.split(","):
        try:
            count, window = component.strip().split(":", 1)
            parsed.append((int(count), int(window)))
        except (TypeError, ValueError):
            continue
    return parsed


class RiotApiClient:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.api_key = required_environment_value(config, "api.key_env")
        api = config["api"]
        self.timeout = int(api.get("timeout_seconds", 30))
        self.max_retries = int(api.get("max_retries", 5))
        self.base_backoff = float(api.get("base_backoff_seconds", 2))
        self.request_log_path: Path = project_path(config, "request_log")
        self.sleeper = sleeper
        self.limiter = SlidingWindowLimiter(
            int(api.get("fallback_requests_per_window", 95)),
            float(api.get("fallback_window_seconds", 120)),
            float(api.get("minimum_interval_seconds", 0.06)),
            sleeper=sleeper,
        )
        self.session = session or requests.Session()
        self.session.headers.update({
            "X-Riot-Token": self.api_key,
            "Accept": "application/json",
            "User-Agent": str(api.get("user_agent", "tft-msc-collector/2.0")),
        })

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "RiotApiClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _request_json(
        self,
        route: str,
        path: str,
        endpoint_type: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        route_value = normalise_route(route)
        url = f"https://{route_value}.api.riotgames.com{path}"
        last_error = ""
        for attempt in range(1, self.max_retries + 2):
            self.limiter.acquire()
            started = time.monotonic()
            response: requests.Response | None = None
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                elapsed = time.monotonic() - started
                self.limiter.observe_headers(response.headers)
                status = response.status_code
                if status == 200:
                    payload = response.json()
                    self._log(endpoint_type, route_value, status, attempt, True, elapsed,
                              response_hash=stable_json_hash(payload))
                    return payload
                if status in {401, 403}:
                    self._log(endpoint_type, route_value, status, attempt, False, elapsed,
                              error_class="authentication")
                    raise RiotAuthenticationError("Riot rejected the API credential")
                if status == 404:
                    self._log(endpoint_type, route_value, status, attempt, False, elapsed,
                              error_class="not_found")
                    raise RiotNotFoundError("Riot resource not found")
                if status == 429:
                    retry_after = self._retry_after_seconds(response, attempt)
                    self.limiter.block(retry_after)
                    self._log(endpoint_type, route_value, status, attempt, False, elapsed,
                              retry_after=retry_after,
                              rate_limit_type=response.headers.get("X-Rate-Limit-Type", "unknown"),
                              error_class="rate_limit")
                    if attempt <= self.max_retries:
                        continue
                    raise RiotAPIError(f"Rate-limit retries exhausted for {endpoint_type}")
                if 500 <= status <= 599:
                    delay = self._backoff_seconds(attempt)
                    self._log(endpoint_type, route_value, status, attempt, False, elapsed,
                              retry_after=delay, error_class="server_error")
                    if attempt <= self.max_retries:
                        self.sleeper(delay)
                        continue
                    raise RiotAPIError(f"Server-error retries exhausted for {endpoint_type}")
                self._log(endpoint_type, route_value, status, attempt, False, elapsed,
                          error_class="http_error")
                raise RiotAPIError(f"Riot API returned HTTP {status} for {endpoint_type}")
            except (RiotNotFoundError, RiotAuthenticationError, RiotAPIError):
                raise
            except (requests.RequestException, ValueError) as exc:
                elapsed = time.monotonic() - started
                last_error = type(exc).__name__
                delay = self._backoff_seconds(attempt)
                self._log(endpoint_type, route_value,
                          response.status_code if response is not None else None,
                          attempt, False, elapsed, retry_after=delay, error_class=last_error)
                if attempt <= self.max_retries:
                    self.sleeper(delay)
                    continue
                raise RiotAPIError(f"Request retries exhausted for {endpoint_type}: {last_error}") from exc
        raise RiotAPIError(f"Request failed for {endpoint_type}: {last_error}")

    def get_account_by_riot_id(self, regional_route: str, game_name: str, tag_line: str) -> dict[str, Any]:
        path = f"/riot/account/v1/accounts/by-riot-id/{quote(game_name, safe='')}/{quote(tag_line, safe='')}"
        payload = self._request_json(regional_route, path, "account_by_riot_id")
        if not isinstance(payload, dict):
            raise RiotAPIError("Account endpoint returned a non-object response")
        return payload

    def get_match_ids(
        self, regional_route: str, puuid: str, *, start: int = 0, count: int = 100,
        start_time: int | None = None, end_time: int | None = None,
    ) -> list[str]:
        params: dict[str, Any] = {"start": max(0, int(start)), "count": max(1, min(int(count), 100))}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        path = f"/tft/match/v1/matches/by-puuid/{quote(puuid, safe='')}/ids"
        payload = self._request_json(regional_route, path, "match_ids_by_puuid", params)
        if not isinstance(payload, list):
            raise RiotAPIError("Match-ID endpoint returned a non-list response")
        return [str(value) for value in payload]

    def get_match(self, regional_route: str, match_id: str) -> dict[str, Any]:
        path = f"/tft/match/v1/matches/{quote(match_id, safe='')}"
        payload = self._request_json(regional_route, path, "match_by_id")
        if not isinstance(payload, dict):
            raise RiotAPIError("Match endpoint returned a non-object response")
        return payload

    def get_league_entries_by_puuid(self, platform_route: str, puuid: str) -> list[dict[str, Any]]:
        path = f"/tft/league/v1/by-puuid/{quote(puuid, safe='')}"
        payload = self._request_json(platform_route, path, "league_entries_by_puuid")
        if not isinstance(payload, list):
            raise RiotAPIError("League endpoint returned a non-list response")
        return [entry for entry in payload if isinstance(entry, dict)]

    def get_league_entries_by_division(
        self, platform_route: str, tier: str, division: str, *, page: int = 1,
        queue: str = "RANKED_TFT",
    ) -> list[dict[str, Any]]:
        path = f"/tft/league/v1/entries/{quote(tier.upper(), safe='')}/{quote(division.upper(), safe='')}"
        payload = self._request_json(platform_route, path, "league_entries_by_division",
                                     {"page": max(1, int(page)), "queue": queue})
        if not isinstance(payload, list):
            raise RiotAPIError("League-division endpoint returned a non-list response")
        return [entry for entry in payload if isinstance(entry, dict)]

    def get_apex_league(
        self, platform_route: str, tier: str, *, queue: str = "RANKED_TFT"
    ) -> dict[str, Any]:
        tier_value = tier.strip().lower()
        if tier_value not in {"master", "grandmaster", "challenger"}:
            raise ValueError("Unsupported apex tier")
        payload = self._request_json(platform_route, f"/tft/league/v1/{tier_value}",
                                     f"{tier_value}_league", {"queue": queue})
        if not isinstance(payload, dict):
            raise RiotAPIError("Apex league endpoint returned a non-object response")
        return payload

    def _retry_after_seconds(self, response: requests.Response, attempt: int) -> float:
        try:
            return max(1.0, float(response.headers.get("Retry-After", "")))
        except ValueError:
            return self._backoff_seconds(attempt)

    def _backoff_seconds(self, attempt: int) -> float:
        return min(60.0, self.base_backoff * (2 ** max(0, attempt - 1)) + random.uniform(0, 0.5))

    def _log(
        self, endpoint_type: str, route: str, status: int | None, attempt: int,
        successful: bool, elapsed: float, *, retry_after: float = 0,
        rate_limit_type: str = "", response_hash: str = "", error_class: str = "",
    ) -> None:
        append_csv_row(self.request_log_path, {
            "request_datetime": utc_now_iso(),
            "endpoint_type": endpoint_type,
            "route": route,
            "path_template": _path_template(endpoint_type),
            "http_status": status,
            "attempt_number": attempt,
            "successful": successful,
            "response_time_seconds": round(elapsed, 3),
            "retry_after_seconds": round(retry_after, 3) if retry_after else "",
            "rate_limit_type": rate_limit_type,
            "response_sha256": response_hash,
            "error_class": error_class,
        }, REQUEST_LOG_FIELDS)


def _path_template(endpoint_type: str) -> str:
    return {
        "account_by_riot_id": "/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}",
        "match_ids_by_puuid": "/tft/match/v1/matches/by-puuid/{puuid}/ids",
        "match_by_id": "/tft/match/v1/matches/{match_id}",
        "league_entries_by_puuid": "/tft/league/v1/by-puuid/{puuid}",
        "league_entries_by_division": "/tft/league/v1/entries/{tier}/{division}",
        "master_league": "/tft/league/v1/master",
        "grandmaster_league": "/tft/league/v1/grandmaster",
        "challenger_league": "/tft/league/v1/challenger",
    }.get(endpoint_type, "/redacted")
