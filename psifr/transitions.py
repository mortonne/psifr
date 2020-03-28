"""Module to analyze transitions during free recall."""

import numpy as np
import pandas as pd


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


def count_lags(recalls, list_length, n_recall, from_mask, to_mask,
               test_values=None, test=None):
    """Count actual and possible serial position lags.

    Parameters
    ----------
    recalls : list
        List of arrays. Each array should have one element for each
        recall attempt, in output order, indicating the serial position
        of that recall (NaN for intrusions).

    list_length : int
         Number of serial positions in each list.

    n_recall : array
        Number of recall attempts in each list.

    from_mask : list
        List of boolean arrays. Each array should be the same order as
        `recalls` and indicates valid recall attempts to transition
        from.

    to_mask : list
        List of boolean arrays. Indicates valid recall attempts to
        transition to.

    test_values : list
        List of arrays. Gives values to use for a transition inclusion
        test.

    test : callable
        Callable that evaluates each transition between items n and
        n+1. Must take test_values[n] and test_values[n + 1] and return
        True if a given transition should be included.
    """

    list_actual = []
    list_possible = []
    for i, output in enumerate(recalls):
        # set up masker to filter transitions
        values = None if test_values is None else test_values[i]
        masker = transitions_masker(output, n_recall[i], from_mask[i], to_mask[i],
                                    test_values=values, test=test)

        for prev, curr, poss in masker:
            # for this step, calculate actual lag and all possible lags
            list_actual.append(curr - prev)
            list_possible.extend(poss - prev)

    # count the actual and possible transitions for each lag
    lags = np.arange(-list_length + 1, list_length + 1)
    actual = pd.Series(np.histogram(list_actual, lags)[0], index=lags[:-1])
    possible = pd.Series(np.histogram(list_possible, lags)[0], index=lags[:-1])
    return actual, possible
