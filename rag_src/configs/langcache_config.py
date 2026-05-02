import os

from pydantic import BaseModel


class RedisConfig(BaseModel):
    server_url: str | None = None
    cache_id: str
    api_key: str
    llm_string: str
    ttl_seconds: int = 900
    failure_cooldown_seconds: int = 300

    @classmethod
    def from_env(cls, llm_string: str) -> "RedisConfig | None":
        cache_id = os.getenv("LANGCACHE_CACHE_ID")
        api_key = os.getenv("LANGCACHE_API_KEY")
        if not cache_id or not api_key:
            return None

        ttl_seconds = int(os.getenv("LANGCACHE_TTL_SECONDS", "900"))
        failure_cooldown_seconds = int(
            os.getenv("LANGCACHE_FAILURE_COOLDOWN_SECONDS", "300")
        )
        return cls(
            server_url=os.getenv("LANGCACHE_SERVER_URL"),
            cache_id=cache_id,
            api_key=api_key,
            llm_string=llm_string,
            ttl_seconds=ttl_seconds,
            failure_cooldown_seconds=failure_cooldown_seconds,
        )
