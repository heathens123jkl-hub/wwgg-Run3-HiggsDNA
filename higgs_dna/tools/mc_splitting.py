"""
MC event splitting utilities for handling MC sample splits based on event ID.
"""

import awkward as ak


def split_mc_events(events: ak.Array, year: str, split_mc: bool, data_kind: str) -> ak.Array:

    if not split_mc or data_kind != "mc":
        return events

    if year == "2024":
        # Keep only even event IDs for 2024
        return events[events.event % 2 == 0]
    elif year == "2025":
        # Keep only odd event IDs for 2025
        return events[events.event % 2 != 0]

    # For other years, return all events
    return events
