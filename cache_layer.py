"""Task 16: in-memory normalized-query response cache."""

import re
import time
from copy import deepcopy

from pydantic import BaseModel

from crew_service import CrewResponse, run_support_query


class CacheResult(BaseModel):
    cache_hit: bool
    normalized_query: str
    backend_call_count: int
    duration_ms: float
    response: CrewResponse


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


class CachedSupportService:
    def __init__(self) -> None:
        self._cache: dict[str, CrewResponse] = {}
        self.backend_call_count = 0

    def clear(self) -> None:
        self._cache.clear()
        self.backend_call_count = 0

    def ask(self, query: str) -> CacheResult:
        started = time.perf_counter()
        key = normalize_query(query)

        if key in self._cache:
            response = deepcopy(self._cache[key])
            return CacheResult(
                cache_hit=True,
                normalized_query=key,
                backend_call_count=self.backend_call_count,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                response=response,
            )

        self.backend_call_count += 1
        response = run_support_query(query)
        self._cache[key] = deepcopy(response)

        return CacheResult(
            cache_hit=False,
            normalized_query=key,
            backend_call_count=self.backend_call_count,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            response=response,
        )
