"""Module to analyze transitions during free recall."""

import numpy as np
import pandas as pd


def transitions_masker(pool_items, recall_items, pool_output, recall_output,
                       pool_test=None, recall_test=None, test=None):
    """Iterate over transitions with masking.

    Iterate over transitions. Transitions are between a "previous" item
    and a "current" item. A transition is yielded if it matches the
    following conditions:
        Each item involved in the transition is in the pool.

        Items are removed from the pool after they appear as the
        previous item.

        Optionally, an additional check is run based on test values
        associated with the items in the transition. For example, this
        could be used to only include transitions where the category of
        the previous and current items is the same.

    Non-included transitions will be skipped.

    The masker will yield "output" values, which may be distinct from
    the item identifiers used to determine item repeats.

    Parameters
    ----------
    pool_items : list
        Items available for recall. Order does not matter. May contain
        repeated values. Item identifiers must be unique within pool.

    recall_items : list
        Recalled items in output position order.

    pool_output : list
        Output values for pool items. Must be the same order as pool.

    recall_output : list
        Output values in output position order.

    pool_test : list, optional
        Test values for items available for recall. Must be the same
        order as pool.

    recall_test : list, optional
        Test values for items in output position order.

    test : callable, optional
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
    pool_items = pool_items.copy()
    pool_output = pool_output.copy()
    if test is not None:
        pool_test = pool_test.copy()

    while n < len(recall_items) - 1:
        # test if the previous item is in the pool
        if recall_items[n] not in pool_items:
            n += 1
            continue

        # remove the item from the pool
        ind = pool_items.index(recall_items[n])
        del pool_items[ind]
        del pool_output[ind]
        if test is not None:
            del pool_test[ind]

        # test if the current item is in the pool
        if recall_items[n + 1] not in pool_items:
            n += 1
            continue

        prev = recall_output[n]
        curr = recall_output[n + 1]
        poss = np.array(pool_output)
        if test is not None:
            # test if this transition is included
            if not test(recall_test[n], recall_test[n + 1]):
                n += 1
                continue

            # get included possible items
            poss = poss[test(recall_test[n], np.array(pool_test))]
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
