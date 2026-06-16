# Description of strategies



## Caching strategies

- ``

## Query strategies

- `FullQuery`: All entrypoints are requested to the end of the compute chain
- `ContextDepthQuery`: The request only considers the context of the last n numbers
- `Batching`: A query request is only produced when a certain threshold is reached
- `RankPredecessors`: The predecessors are ranked, only a certain threshold is asked to be computed