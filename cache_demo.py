"""Task 16 cache-hit demonstration."""

from cache_layer import CachedSupportService


if __name__ == "__main__":
    service = CachedSupportService()

    first = service.ask("What documents are needed for KYC?")
    second = service.ask("  WHAT documents   are needed for KYC?  ")

    print("FIRST CALL")
    print(
        {
            "cache_hit": first.cache_hit,
            "normalized_query": first.normalized_query,
            "backend_call_count": first.backend_call_count,
            "duration_ms": first.duration_ms,
        }
    )

    print("\nREPEATED NORMALIZED QUERY")
    print(
        {
            "cache_hit": second.cache_hit,
            "normalized_query": second.normalized_query,
            "backend_call_count": second.backend_call_count,
            "duration_ms": second.duration_ms,
        }
    )

    print("\nSame answer:", first.response.answer == second.response.answer)
