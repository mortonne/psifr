"""Module to analyze transitions during free recall."""

import numpy as np
import pandas as pd


def transitions_masker(pool, output, pool_test=None, output_test=None,
                       test=None):
    """Iterate over transitions with masking.

    Parameters
    ----------
    pool : list
        Items available for recall. Order does not matter. May contain
        repeated values.

    output : list
        Recalled items in output position order.

    pool_test : list
        Test values for items available for recall. Must be the same
        order as pool.

    output_test : list
        Test values for items in output position order.

    test : callable
        Used to test whether individual transitions should be included,
        based on test values.
            test(prev, curr) - test for included transition
            test(prev, poss) - test for included possible transition

    Yields
    ------
    prev : object
        Output value for the "from" item on this transition.

    curr : object
        Output value for the "to" item.

    poss : numpy.array
        Output values for all possible valid "to" items.
    """

    n = 0
    possible = pool.copy()
    if test is not None:
        possible_test = pool_test.copy()

    while n < len(output) - 1:
        # test if the previous item is in the pool
        if output[n] not in possible:
            n += 1
            continue

        # remove the item from the pool
        ind = possible.index(output[n])
        del possible[ind]
        if test is not None:
            del possible_test[ind]

        # test if the current item is in the pool
        if output[n + 1] not in possible:
            n += 1
            continue

        prev = output[n]
        curr = output[n + 1]
        poss = np.array(possible)
        if test is not None:
            # test if this transition is included
            if not test(output_test[n], output_test[n + 1]):
                n += 1
                continue

            # get included possible items
            poss = poss[test(output_test[n], np.array(possible_test))]
        n += 1
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

    n_recall : numpy.array
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
