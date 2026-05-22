import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conformance import build_network, load_training_data, load_validation_trace
from algo.utility.event_log_splitter import EventLogSplitter


if __name__ == "__main__":
    splitter = EventLogSplitter("../gt/test/datasets/Sepsis.xes", location_key="org:group")
    total_cases = 100

    training_sequences, mapping = load_training_data(splitter, 10, c=False)
    training_sequences_c, mapping_c = load_training_data(splitter, 10, c=True)
    executor = build_network(training_sequences, mapping)
    executor_c = build_network(training_sequences_c, mapping_c)

    print(f"Training traces: {len(training_sequences)}")
    print(f"Decentral participants: {len(executor.network.participants)}")
    print(f"Central participants: {len(executor_c.network.participants)}")
    print(f"Total cases: {total_cases}")
    print()

    first_fail = None

    total_route = 0
    total_remote = 0
    total_states = 0
    total_route_c = 0
    total_states_c = 0
    n_processed = 0

    for idx in range(total_cases):
        trace_events = load_validation_trace(splitter, idx, c=False)
        trace_events_c = load_validation_trace(splitter, idx, c=True)
        if len(trace_events) < 2:
            continue

        n_processed += 1
        case_id = f"case_{idx}"
        network = executor.network
        network_c = executor_c.network

        network.reset_stats()
        network_c.reset_stats()

        case_fail = None
        for ei in range(len(trace_events)):
            trace_events[ei].case_id = case_id
            trace_events_c[ei].case_id = case_id
            dec_align = executor.record_event(trace_events[ei])
            cen_align = executor_c.record_event(trace_events_c[ei])
            print("-" * 40)
            print(dec_align)
            print("-" * 40)
            print(cen_align)
            print("-" * 40)
            if dec_align.cost != cen_align.cost:
                case_fail = (ei, dec_align, cen_align)
                break

        total_route += network.route_calls
        total_remote += network.remote_calls
        total_states += network.get_total_states()
        total_route_c += network_c.route_calls
        total_states_c += network_c.get_total_states()

        final_cost = dec_align.cost
        status = "OK" if case_fail is None else "FAIL"
        sys.stdout.write(f"\r  [{idx:4d}/{total_cases}] cost={final_cost:3d}  events={len(trace_events):3d}  {status}")
        sys.stdout.flush()

        if case_fail is not None:
            ei, dec_align, cen_align = case_fail
            first_fail = (idx, trace_events[:ei + 1], trace_events_c[:ei + 1], dec_align, cen_align)
            print("\n\nMISMATCH FOUND!")
            break

    print()

    if first_fail:
        idx, trace, trace_c, dec_align, cen_align = first_fail
        print(f"\nIndex: {idx}")
        print(f"Trace prefix ({len(trace)} events, up to failing event):")
        for i, a in enumerate(trace):
            print(f"  {i}: {a}")
        print(f"\nCentralized trace prefix ({len(trace_c)} events):")
        for i, a in enumerate(trace_c):
            print(f"  {i}: {a}")
        print(f"\nDecentralized cost: {dec_align.cost}")
        print(f"Centralized cost: {cen_align.cost}")
        print(f"\nDecentralized alignment:")
        for i, (m, l) in enumerate(dec_align.moves):
            print(f"  {i}: {m or '>>':40s} | {l or '>>'}")
        print(f"\nCentralized alignment:")
        for i, (m, l) in enumerate(cen_align.moves):
            print(f"  {i}: {m or '>>':40s} | {l or '>>'}")
    else:
        avg_route = total_route / n_processed
        avg_remote = total_remote / n_processed
        avg_states = total_states / n_processed
        avg_route_c = total_route_c / n_processed
        avg_states_c = total_states_c / n_processed
        sep = "=" * 70
        dash = "-" * 40
        dash2 = "-" * 12
        print(f"\n{sep}")
        print(f"  Results: {n_processed} traces, all matched!")
        print(f"{sep}")
        print(f"  {'Metric':<40s} {'Decentral':>12s} {'Central':>12s}")
        print(f"  {dash} {dash2} {dash2}")
        print(f"  {'Avg network lookups (route_calls)':<40s} {avg_route:>12.1f} {avg_route_c:>12.1f}")
        print(f"  {'Avg cross-participant calls (remote)':<40s} {avg_remote:>12.1f} {'N/A':>12s}")
        print(f"  {'Avg states explored (memo entries)':<40s} {avg_states:>12.1f} {avg_states_c:>12.1f}")
        print(f"  {'Total network lookups':<40s} {total_route:>12d} {total_route_c:>12d}")
        print(f"  {'Total cross-participant calls':<40s} {total_remote:>12d} {'N/A':>12s}")
        print(f"  {'Total states explored':<40s} {total_states:>12d} {total_states_c:>12d}")
        print(f"{sep}")
