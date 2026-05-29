from __future__ import annotations

from conformance.strategy.cache.abstract_cache_strategy import CacheStrategy


class InfiniteCacheStrategy(CacheStrategy):
    def on_event(self, log_index):
        pass

    def after_insertion(self, cache, key):
        pass

    def max_predecessor_depth(self, entrypoint, log_index):
        return 0


class DepthLimitedCacheStrategy(CacheStrategy):
    def __init__(self, inner: CacheStrategy, max_depth: int = 5):
        self.inner = inner
        self._max_depth = max_depth

    def on_event(self, log_index):
        self.inner.on_event(log_index)

    def after_insertion(self, cache, key):
        self.inner.after_insertion(cache, key)

    def max_predecessor_depth(self, entrypoint, log_index):
        return self._max_depth
