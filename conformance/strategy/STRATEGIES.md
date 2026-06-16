# Description of strategies



## Caching strategies

- `CacheAll`: Caches all states

## Query strategies

- `FullQuery`: All entrypoints are requested to the end of the compute chain
- `RankPredecessors`: The predecessors are ranked, only a certain threshold is asked to be computed
- `ContextDepthQuery`: The request only considers the context of the last n numbers
- `Batching`: A query request is only produced when a certain threshold is reached