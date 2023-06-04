"""Analyze recalls as a function of output position."""

import numpy as np

from psifr import maskers


def count_outputs(
    list_length,
    pool_items,
    recall_items,
    pool_label,
    recall_label,
    pool_test=None,
    recall_test=None,
    test=None,
    count_unique=False,
):
    """
    Count actual and possible recalls for each output position.

    Parameters
    ----------
    list_length : int
        Number of items in each list.

    pool_items : list
        List of the serial positions available for recall in each list.
        Must match the serial position codes used in `recall_items`.

    recall_items : list
        List indicating the serial position of each recall in output
        order (NaN for intrusions).

    pool_label : list
        List of the positions to use for calculating lag. Default is to
        use `pool_items`.

    recall_label : list
        List of position labels in recall order. Default is to use
        `recall_items`.

    pool_test : list, optional
         List of some test value for each item in the pool.

    recall_test : list, optional
        List of some test value for each recall attempt by output
        position.

    test : callable
        Callable that evaluates each transition between items n and
        n+1. Must take test values for items n and n+1 and return True
        if a given transition should be included.

    count_unique : bool
        If true, possible recalls with the same label will only be
        counted once.

    Returns
    -------
    actual : numpy.ndarray
        [outputs x inputs] array of actual recall counts.

    possible : numpy.ndarray
        [outputs x inputs] array of possible recall counts.

    Examples
    --------
    >>> from psifr import outputs
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> actual, possible = outputs.count_outputs(
    ...     4, pool_items, recall_items, pool_items, recall_items
    ... )
    >>> actual
    array([[0, 0, 0, 1],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [1, 0, 0, 0]])
    >>> possible
    array([[1, 1, 1, 1],
           [1, 1, 1, 0],
           [1, 0, 1, 0],
           [1, 0, 0, 0]])
    """
    if pool_label is None:
        pool_label = pool_items

    if recall_label is None:
        recall_label = recall_items

    count_actual = np.zeros((list_length, list_length), dtype=int)
    count_possible = np.zeros((list_length, list_length), dtype=int)
    for i, recall_items_list in enumerate(recall_items):
        # set up masker to filter outputs
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = maskers.outputs_masker(
            pool_items[i],
            recall_items_list,
            pool_label[i],
            recall_label[i],
            pool_test_list,
            recall_test_list,
            test,
        )

        for curr, poss, op in masker:
            curr = int(curr)
            poss = poss.astype(int)

            # for this step, calculate actual input position and
            # possible input positions
            count_actual[op - 1, curr - 1] += 1
            if count_unique:
                for j in poss:
                    count_possible[op - 1, j - 1] += 1
            else:
                count_possible[op - 1, poss - 1] += 1
    return count_actual, count_possible
