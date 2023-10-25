"""Utilities to iterate over masked lists of recalls."""

import numpy as np
import pandas as pd


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

    Examples
    --------
    >>> from psifr import maskers
    >>> pool_items = [1, 2, 3, 4]
    >>> recall_items = [4, 2, 3, 1]
    >>> masker = maskers.outputs_masker(
    ...     pool_items, recall_items, pool_items, recall_items
    ... )
    >>> for curr, poss, output in masker:
    ...     print(curr, poss, output)
    4 [1 2 3 4] 1
    2 [1 2 3] 2
    3 [1 3] 3
    1 [1] 4
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


def transitions_masker(
    pool_items,
    recall_items,
    pool_output,
    recall_output,
    pool_test=None,
    recall_test=None,
    test=None,
):
    """
    Iterate over transitions with masking.

    Transitions are between a "previous" item and a "current" item.
    Non-included transitions will be skipped. A transition is yielded
    only if it matches the following conditions:

    (1) Each item involved in the transition is in the pool. Items are
    removed from the pool after they appear as the previous item.

    (2) Optionally, an additional check is run based on test values
    associated with the items in the transition. For example, this
    could be used to only include transitions where the category of
    the previous and current items is the same.

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
    output : int
        Output position of this transition. The first transition is 1.

    prev : object
        Output value for the "from" item on this transition.

    curr : object
        Output value for the "to" item.

    poss : numpy.array
        Output values for all possible valid "to" items.

    Examples
    --------
    >>> from psifr import maskers
    >>> pool = [1, 2, 3, 4, 5, 6]
    >>> recs = [6, 2, 3, 6, 1, 4]
    >>> masker = maskers.transitions_masker(
    ...     pool_items=pool, recall_items=recs, pool_output=pool, recall_output=recs
    ... )
    >>> for output, prev, curr, poss in masker:
    ...     print(output, prev, curr, poss)
    1 6 2 [1 2 3 4 5]
    2 2 3 [1 3 4 5]
    5 1 4 [4 5]
    """
    pool_items = pool_items.copy()
    pool_output = pool_output.copy()
    if test is not None:
        pool_test = pool_test.copy()

    for n in range(len(recall_items) - 1):
        # test if the previous item is in the pool
        if pd.isnull(recall_items[n]) or (recall_items[n] not in pool_items):
            continue

        # remove the item from the pool
        ind = pool_items.index(recall_items[n])
        del pool_items[ind]
        del pool_output[ind]
        if test is not None:
            del pool_test[ind]

        # test if the current item is in the pool
        if pd.isnull(recall_items[n + 1]) or (recall_items[n + 1] not in pool_items):
            continue

        prev = recall_output[n]
        curr = recall_output[n + 1]
        poss = np.array(pool_output)
        if test is not None:
            # test if this transition is included
            if not test(recall_test[n], recall_test[n + 1]):
                continue

            # get included possible items
            ind = test(recall_test[n], np.array(pool_test))
            if not isinstance(ind, np.ndarray):
                ind = np.repeat(ind, poss.shape)
            poss = poss[ind]
        yield n + 1, prev, curr, poss


def sequences_masker(
    n_transitions,
    pool_items,
    recall_items,
    pool_output,
    recall_output,
    pool_test=None,
    recall_test=None,
    test=None,
):
    """
    Yield sequences of adjacent included transitions.

    Parameters
    ----------
    n_transitions : int
        Number of transitions to include in yielded sequences.

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
    output : int
        Output positions of included transitions. The first transition
        is 1.

    prev : list
        Output values for the "from" item in included transitions.

    curr : list
        Output values for the "to" item in included transitions.

    poss : list of numpy.ndarray
        Output values for all possible valid "to" items in included
        transitions.

    See Also
    --------
    transitions_masker : Yield included transitions.

    Examples
    --------
    >>> from psifr import maskers
    >>> pool = [1, 2, 3, 4, 5, 6]
    >>> recs = [6, 2, 3, 6, 1, 4, 5]
    >>> masker = maskers.sequences_masker(
    ...     2, pool_items=pool, recall_items=recs, pool_output=pool, recall_output=recs
    ... )
    >>> for output, prev, curr, poss in masker:
    ...     print(output, prev, curr, poss)
    [1, 2] [6, 2] [2, 3] [array([1, 2, 3, 4, 5]), array([1, 3, 4, 5])]
    [5, 6] [1, 4] [4, 5] [array([4, 5]), array([5])]

    >>> pool = [1, 2, 3, 4]
    >>> recs = [4, 3, 1, 2]
    >>> masker = maskers.sequences_masker(
    ...     3, pool_items=pool, recall_items=recs, pool_output=pool, recall_output=recs
    ... )
    >>> for output, prev, curr, poss in masker:
    ...     print(output, prev, curr, poss)
    [1, 2, 3] [4, 3, 1] [3, 1, 2] [array([1, 2, 3]), array([1, 2]), array([2])]
    """
    masker = transitions_masker(
        pool_items,
        recall_items,
        pool_output,
        recall_output,
        pool_test=pool_test,
        recall_test=recall_test,
        test=test,
    )
    s_output = []
    s_prev = []
    s_curr = []
    s_poss = []
    prev_output = 0
    sequence_len = 0
    for output, prev, curr, poss in masker:
        if (output - prev_output) == 1:
            s_output.append(output)
            s_prev.append(prev)
            s_curr.append(curr)
            s_poss.append(poss)
            sequence_len += 1
        else:
            # a break in the chain
            s_output = [output]
            s_prev = [prev]
            s_curr = [curr]
            s_poss = [poss]
            sequence_len = 1
        prev_output = output

        if sequence_len >= n_transitions:
            ind = slice(sequence_len - n_transitions, None)
            yield s_output[ind], s_prev[ind], s_curr[ind], s_poss[ind]


def windows_masker(
    list_length,
    window_lags,
    pool_items,
    recall_items,
    pool_output,
    recall_output,
    pool_test=None,
    recall_test=None,
    test=None,
    exclude_prev_window=False,
):
    """
    Yield windows around previous items in the input list.

    Parameters
    ----------
    list_length : int
        Number of items in each list.

    window_lags : array_like
        Serial position lags to include in the window.

    pool_items : list
        Input position of items available for recall.

    recall_items : list
        Input position of recalled items, in output position order.

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

    exclude_prev_window : bool, optional
        If true, transitions preceded by items in the window around the
        previous item in the list will be excluded.

    Yields
    ------
    output : int
        Output positions of included transitions. The first transition
        is 1.

    prev : list
        Output values for the "from" item in included transitions.

    curr : list
        Output values for the "to" item in included transitions.

    poss : list of numpy.ndarray
        Output values for all possible valid "to" items in included
        transitions.
    """
    # pool items include all items in presentation order; poss items
    # include only items that have not been recalled yet
    poss_items = pool_items.copy()
    poss_output = pool_output.copy()
    pool_output = np.asarray(pool_output)
    if test is not None:
        poss_test = pool_test.copy()
        pool_test = np.asarray(pool_test)
    window_lags = np.asarray(window_lags)

    for n in range(len(recall_items) - 1):
        # test if the previous item is in the pool
        if pd.isnull(recall_items[n]) or (recall_items[n] not in poss_items):
            continue

        # remove the item from the pool
        ind = poss_items.index(recall_items[n])
        del poss_items[ind]
        del poss_output[ind]
        if test is not None:
            del poss_test[ind]

        # test if the current item is in the pool
        if pd.isnull(recall_items[n + 1]) or (recall_items[n + 1] not in poss_items):
            continue

        # get windowed items in the input list
        prev = int(recall_items[n]) + window_lags

        # exclude if any windowed items do not exist or fail test
        if np.any(prev < 1) or np.any(prev > list_length):
            continue

        # exclude current transition if prior transition was from the window
        if exclude_prev_window and n > 0:
            if recall_items[n - 1] in prev:
                continue

        # exclude current/possible items in the window
        curr = int(recall_items[n + 1])
        if curr in prev:
            continue
        poss = np.asarray(poss_items, int)
        include_poss = ~np.isin(poss, prev)
        poss = poss[include_poss]

        if test is not None:
            # test if this transition is included
            if np.any(~test(pool_test[prev - 1], recall_test[n + 1])):
                continue

            # get included possible items
            include = test(pool_test[prev - 1][:, np.newaxis], pool_test[poss - 1])
            poss = poss[np.all(include, axis=0)]
        yield n + 1, pool_output[prev - 1], pool_output[curr - 1], pool_output[poss - 1]
