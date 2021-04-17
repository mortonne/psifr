"""Analyze recalls as a function of output position."""

import numpy as np


def outputs_masker(
    pool_items,
    recall_items,
    pool_output,
    recall_output,
    pool_test=None,
    recall_test=None,
    test=None,
):
    """
    Iterate over valid outputs.

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
        Used to test whether output recalls and possible recalls should
        be included, based on their test values.

    Yields
    ------
    curr : object
        Output value for the item at this valid output position.

    poss : numpy.array
        Output values for all possible items that could be recalled at
        this output position.

    output : int
        Current output position.
    """
    pool_items = pool_items.copy()
    pool_output = pool_output.copy()
    if test is not None:
        pool_test = pool_test.copy()

    n = 0
    output = 0
    while n < len(recall_items):
        # test if the current item is in the pool
        if recall_items[n] not in pool_items:
            n += 1
            continue
        output += 1

        curr = recall_output[n]
        poss = np.array(pool_output)

        # remove the item from the pool
        ind = pool_items.index(recall_items[n])
        del pool_items[ind]
        del pool_output[ind]
        if test is not None:
            del pool_test[ind]

        if test is not None:
            # test if this recall is included
            if not test(recall_test[n]):
                n += 1
                continue

            # get included possible items
            poss = poss[test(np.array(pool_test))]
        n += 1
        yield curr, poss, output


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

    pool_label : list, optional
        List of the positions to use for calculating lag. Default is to
        use `pool_items`.

    recall_label : list, optional
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
        masker = outputs_masker(
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

            # for this step, calculate actual lag and all possible lags
            count_actual[op - 1, curr - 1] += 1
            if count_unique:
                for j in poss:
                    count_possible[op - 1, j - 1] += 1
            else:
                count_possible[op - 1, poss - 1] += 1
    return count_actual, count_possible
