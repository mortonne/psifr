"""Module to analyze transitions during free recall."""


def transitions_masker(outputs, n_recalls, from_mask, to_mask,
                       test_values=None, test=None):
    """Iterate over transitions with masking.

    Parameters
    ----------
    outputs : array
        Values to output for each transition.

    n_recalls : int
        Number of recall attempts.

    from_mask : array
        Boolean array indicating valid positions to transition from.

    to_mask : array
        Boolean array indicating valid positions to transition to.

    test_values : array
        Same shape and order as `outputs`. Used to test whether
        individual transitions should be included.

    test : callable
        Used to test whether individual transitions should be included.
            test(prev, curr) - test for included transition
            test(prev, poss) - test for included possible transition

    Yields
    ------
    prev : object
        Output value for the "from" item on this transition.

    curr : object
        Output value for the "to" item.

    poss : array
        Output values for all possible valid "to" items.
    """

    # list of all valid outputs
    valid_outputs = outputs[to_mask]
    if test_values is not None:
        valid_values = test_values[to_mask]

    # counter for recall and counter for valid recall
    n = 0
    m = 0
    while n < n_recalls - 1:
        # check if the positions involved in this transition are valid
        if not from_mask[n] or not to_mask[n + 1]:
            n += 1
            continue

        # transition outputs
        prev = outputs[n]
        curr = outputs[n + 1]

        # valid next items at this output position
        step_outputs = valid_outputs[m + 1:]

        if test_values is not None:
            # check if this transition is included
            if not test(test_values[n], test_values[n + 1]):
                n += 1
                m += 1
                continue

            # get valid possible recalls that are included
            step_values = valid_values[m + 1:]
            poss = step_outputs[test(test_values[n], step_values)]
        else:
            poss = step_outputs

        n += 1
        m += 1
        yield prev, curr, poss
