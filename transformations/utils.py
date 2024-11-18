# This function is not implemented directly in the model, to make it reusable in other contexts. Right now it is used in the dfg.
def cases_list_to_dict(cases_list: list[list[str, ...]]) -> dict[tuple[str, ...], int]:
    """transforms a list of cases into a dictionary where the key is a tuple of events and the value is the number of occurrences of that case

    Parameters
    ----------
    cases_list : list[list[str, ...]]
        A list of cases, where each case is a list of events.

    Returns
    -------
    dict[tuple[str, ...], int]
        A dictionary of cases, where each case is a tuple of events and the value is the number of occurrences.
    """
    cases = {}
    for trace in cases_list:
        trace_tuple = tuple(trace)
        cases[trace_tuple] = cases.get(trace_tuple, 0) + 1
    return cases
