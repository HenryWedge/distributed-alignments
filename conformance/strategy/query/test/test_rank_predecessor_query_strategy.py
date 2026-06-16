from conformance.strategy.query.rank_predecessors_query_strategy import RankPredecessorsQueryStrategy

if __name__ == '__main__':
    print("=== ohne Limit ===")
    strategy = RankPredecessorsQueryStrategy(initial_participants=["A", "B", "C", "D"])
    strategy.update_predecessor("A")
    strategy.update_predecessor("B")
    strategy.update_predecessor("B")
    strategy.update_predecessor("C")

    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())

    strategy.reset()
    print("---")

    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())

    strategy.reset()
    print("---")

    strategy.update_predecessor("A")
    strategy.update_predecessor("A")

    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())

    print()
    print("=== Limit 2 ===")
    strategy = RankPredecessorsQueryStrategy(
        initial_participants=["A", "B", "C", "D"], query_limit=1
    )
    strategy.update_predecessor("A")
    strategy.update_predecessor("B")
    strategy.update_predecessor("B")
    strategy.update_predecessor("C")

    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())

    strategy.reset()
    print("---")

    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())
    print(strategy.get_next_predecessor())