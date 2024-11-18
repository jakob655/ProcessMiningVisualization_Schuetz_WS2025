def filter_events(log, events_to_remove: set[str]):
    """Filter out events from the log.

    Parameters
    ----------
    log : dict[tuple[str, ...], int]
        A dictionary containing the traces and their frequencies in the log.
    events_to_remove : set[str]
        A set containing the events that should be removed from the log.

    Returns
    -------
    dict[tuple[str, ...], int]
        A dictionary containing the traces and their frequencies in the log without the events to remove.
    """
    filtered_log = {}
    for trace, frequency in log.items():
        filtered_trace = [event for event in trace if event not in events_to_remove]
        if len(filtered_trace) > 0:
            filtered_log[tuple(filtered_trace)] = (
                filtered_log.get(tuple(filtered_trace), 0) + frequency
            )

    return filtered_log


def filter_traces(log, min_frequency: int):
    """Filter out traces from the log that have a frequency below the minimum frequency.

    Parameters
    ----------
    log : dict[tuple[str, ...], int]
        A dictionary containing the traces and their frequencies in the log.
    min_frequency : int
        The minimum frequency a trace should have to be kept in the log.

    Returns
    -------
    dict[tuple[str, ...], int]
        A dictionary containing the traces and their frequencies in the log without the traces with a frequency below the minimum frequency.
    """
    filtered_log = {}
    for trace, frequency in log.items():
        if frequency >= min_frequency:
            filtered_log[trace] = frequency

    return filtered_log
