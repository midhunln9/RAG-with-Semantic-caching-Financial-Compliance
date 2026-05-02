from time import monotonic

from loguru import logger
from redisvl.extensions.cache.llm import LangCacheSemanticCache

from rag_src.configs.langcache_config import RedisConfig
from rag_src.protocols.cache import CacheProtocol


class RedisLangCache(CacheProtocol):
    def __init__(self, redis_config: RedisConfig):
        self.redis_config = redis_config
        self._disabled_until = 0.0
        self.cache = LangCacheSemanticCache(
            name="rag-cache",
            server_url=self.redis_config.server_url,
            cache_id=self.redis_config.cache_id,
            api_key=self.redis_config.api_key,
            ttl=self.redis_config.ttl_seconds,
        )

    def get(self, key: str) -> str | None:
        if self._is_in_cooldown():
            return None

        cache_hits = self.cache.check(
            prompt=self._namespaced_prompt(key), num_results=1
        )
        if not cache_hits:
            return None
        return self._first_cached_response(cache_hits)

    def set(self, key: str, value: str) -> None:
        if self._is_in_cooldown():
            return

        self.cache.store(
            prompt=self._namespaced_prompt(key), response=value, ttl=self.redis_config.ttl_seconds
        )

    def _namespaced_prompt(self, key: str) -> str:
        return f"{self.redis_config.llm_string}\n{key}"

    def _is_in_cooldown(self) -> bool:
        return monotonic() < self._disabled_until

    def mark_backend_unavailable(self, operation: str, error: Exception) -> None:
        self._disabled_until = monotonic() + self.redis_config.failure_cooldown_seconds
        logger.warning(
            "[lang_cache] backend unavailable during {}. "
            "Disabling cache for {} seconds. Reason: {}",
            operation,
            self.redis_config.failure_cooldown_seconds,
            error,
        )

    @staticmethod
    def _first_cached_response(cache_hits: list[dict]) -> str | None:
        first_hit = cache_hits[0] if cache_hits else None
        if first_hit is None:
            return None
        response = first_hit.get("response")
        return response if isinstance(response, str) else None
