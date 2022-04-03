"""Module to analyze transitions during free recall."""

import itertools
import numpy as np
from scipy import stats
import pandas as pd


def percentile_rank(actual, possible):
    """
    Get percentile rank of a score compared to possible scores.

    Parameters
    ----------
    actual : float
        Score to be ranked. Generally a distance score.

    possible : numpy.ndarray or list
        Possible scores to be compared to.

    Returns
    -------
    rank : float
        Rank scaled to range from 0 (low score) to 1 (high score).

    Examples
    --------
    >>> from psifr import transitions
    >>> actual = 3
    >>> possible = [1, 2, 2, 2, 3]
    >>> transitions.percentile_rank(actual, possible)
    1.0
    """
    possible_rank = stats.rankdata(possible)
    actual_rank = possible_rank[actual == np.asarray(possible)]
    possible_count = np.count_nonzero(~np.isnan(possible))
    if possible_count == 1:
        return np.nan
    rank = (actual_rank - 1) / (possible_count - 1)
    return rank[0]


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
    >>> from psifr import transitions
    >>> pool = [1, 2, 3, 4, 5, 6]
    >>> recs = [6, 2, 3, 6, 1, 4]
    >>> masker = transitions.transitions_masker(
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
    >>> from psifr import transitions
    >>> pool = [1, 2, 3, 4, 5, 6]
    >>> recs = [6, 2, 3, 6, 1, 4, 5]
    >>> masker = transitions.sequences_masker(
    ...     2, pool_items=pool, recall_items=recs, pool_output=pool, recall_output=recs
    ... )
    >>> for output, prev, curr, poss in masker:
    ...     print(output, prev, curr, poss)
    [1, 2] [6, 2] [2, 3] [array([1, 2, 3, 4, 5]), array([1, 3, 4, 5])]
    [5, 6] [1, 4] [4, 5] [array([4, 5]), array([5])]

    >>> pool = [1, 2, 3, 4]
    >>> recs = [4, 3, 1, 2]
    >>> masker = transitions.sequences_masker(
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


def count_lags(
    list_length,
    pool_items,
    recall_items,
    pool_label=None,
    recall_label=None,
    pool_test=None,
    recall_test=None,
    test=None,
    count_unique=False,
):
    """
    Count actual and possible serial position lags.

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

    count_unique : bool, optional
        If true, only unique values will be counted toward the possible
        transitions. If multiple items are avilable for recall for a
        given transition and a given bin, that bin will only be
        incremented once. If false, all possible transitions will add
        to the count.

    Returns
    -------
    actual : pandas.Series
        Count of actual lags that occurred in the recall sequence.

    possible : pandas.Series
        Count of possible lags.

    See Also
    --------
    rank_lags : Rank of serial position lags.

    Examples
    --------
    >>> from psifr import transitions
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> actual, possible = transitions.count_lags(4, pool_items, recall_items)
    >>> actual
    lag
    -3    0
    -2    2
    -1    0
     0    0
     1    1
     2    0
     3    0
    dtype: int64
    >>> possible
    lag
    -3    1
    -2    2
    -1    2
     0    0
     1    1
     2    0
     3    0
    dtype: int64
    """
    if pool_label is None:
        pool_label = pool_items

    if recall_label is None:
        recall_label = recall_items

    list_actual = []
    list_possible = []
    for i, recall_items_list in enumerate(recall_items):
        # set up masker to filter transitions
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(
            pool_items[i],
            recall_items_list,
            pool_label[i],
            recall_label[i],
            pool_test_list,
            recall_test_list,
            test,
        )

        for output, prev, curr, poss in masker:
            # for this step, calculate actual lag and all possible lags
            list_actual.append(curr - prev)
            tran_poss = poss - prev
            if count_unique:
                tran_poss = np.unique(tran_poss)
            list_possible.extend(tran_poss)

    # count the actual and possible transitions for each lag
    max_lag = list_length - 1
    lags = np.arange(-max_lag, max_lag + 2)
    index = pd.Index(lags[:-1], name='lag')
    actual = pd.Series(np.histogram(list_actual, lags)[0], index=index)
    possible = pd.Series(np.histogram(list_possible, lags)[0], index=index)
    return actual, possible


def count_lags_compound(
    list_length,
    pool_items,
    recall_items,
    pool_label=None,
    recall_label=None,
    pool_test=None,
    recall_test=None,
    test=None,
    count_unique=False,
):
    """
    Count lags conditional on the lag of the previous transition.

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

    count_unique : bool, optional
        If true, only unique values will be counted toward the possible
        transitions. If multiple items are avilable for recall for a
        given transition and a given bin, that bin will only be
        incremented once. If false, all possible transitions will add
        to the count.

    Returns
    -------
    actual : pandas.Series
        Count of actual lags that occurred in the recall sequence.

    possible : pandas.Series
        Count of possible lags.

    See Also
    --------
    count_lags : Count of individual transitions.

    Examples
    --------
    >>> from psifr import transitions
    >>> pool_items = [[1, 2, 3]]
    >>> recall_items = [[3, 1, 2]]
    >>> actual, possible = transitions.count_lags_compound(3, pool_items, recall_items)
    >>> (actual == possible).all()
    True
    >>> actual
    previous  current
    -2        -2         0
              -1         0
               0         0
               1         1
               2         0
    -1        -2         0
              -1         0
               0         0
               1         0
               2         0
     0        -2         0
              -1         0
               0         0
               1         0
               2         0
     1        -2         0
              -1         0
               0         0
               1         0
               2         0
     2        -2         0
              -1         0
               0         0
               1         0
               2         0
    dtype: int64
    """
    if pool_label is None:
        pool_label = pool_items

    if recall_label is None:
        recall_label = recall_items

    prev_actual = []
    prev_possible = []
    curr_actual = []
    curr_possible = []
    for i, recall_items_list in enumerate(recall_items):
        # set up masker to filter pairs of transitions
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = sequences_masker(
            2,
            pool_items[i],
            recall_items_list,
            pool_label[i],
            recall_label[i],
            pool_test_list,
            recall_test_list,
            test,
        )

        for output, prev, curr, poss in masker:
            # calculate lag of prior transition
            prev_lag = curr[-2] - prev[-2]

            # actual and possible lags of current transition
            curr_lag = curr[-1] - prev[-1]
            poss_lag = poss[-1] - prev[-1]

            # add lags to count
            prev_actual.append(prev_lag)
            curr_actual.append(curr_lag)
            if count_unique:
                poss_lag = np.unique(poss_lag)
            prev_possible.extend([prev_lag] * len(poss_lag))
            curr_possible.extend(poss_lag)

    # define possible combinations of previous and current lags
    max_lag = list_length - 1
    bins = np.arange(-max_lag, max_lag + 2)
    lags = bins[:-1]
    compound_lags = list(itertools.product(lags, lags))
    index = pd.MultiIndex.from_tuples(compound_lags, names=['previous', 'current'])

    # count the actual and possible transitions for each (lag, lag)
    # combination
    count_actual = np.histogram2d(prev_actual, curr_actual, bins)[0].astype(int)
    actual = pd.Series(count_actual.ravel(), index=index)
    count_possible = np.histogram2d(prev_possible, curr_possible, bins)[0].astype(int)
    possible = pd.Series(count_possible.ravel(), index=index)
    return actual, possible


def rank_lags(
    pool_items,
    recall_items,
    pool_label=None,
    recall_label=None,
    pool_test=None,
    recall_test=None,
    test=None,
):
    """
    Calculate rank of absolute lag for free recall lists.

    Parameters
    ----------
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

    Returns
    -------
    rank : list
        Absolute lag percentile rank for each included transition. The
        rank is 0 if the lag was the most distant of the available
        transitions, and 1 if the lag was the closest. Ties are
        assigned to the average percentile rank.

    See Also
    --------
    count_lags : Count actual and possible serial position lags.

    Examples
    --------
    >>> from psifr import transitions
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> transitions.rank_lags(pool_items, recall_items)
    [0.5, 0.5, nan]
    """
    if pool_label is None:
        pool_label = pool_items

    if recall_label is None:
        recall_label = recall_items

    rank = []
    for i, recall_items_list in enumerate(recall_items):
        # set up masker to filter transitions
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(
            pool_items[i],
            recall_items_list,
            pool_label[i],
            recall_label[i],
            pool_test_list,
            recall_test_list,
            test,
        )

        for output, prev, curr, poss in masker:
            actual = np.abs(curr - prev)
            possible = np.abs(poss - prev)
            rank.append(1 - percentile_rank(actual, possible))
    return rank


def count_distance(
    distances,
    edges,
    pool_items,
    recall_items,
    pool_index,
    recall_index,
    pool_test=None,
    recall_test=None,
    test=None,
    count_unique=False,
):
    """
    Count transitions within distance bins.

    Parameters
    ----------
    distances : numpy.array
        Items x items matrix of pairwise distances or similarities.

    edges : array-like
        Edges of bins to apply to distances.

    pool_items : list of list
        Unique item codes for each item in the pool available for recall.

    recall_items : list of list
        Unique item codes of recalled items.

    pool_index : list of list
        Index of each item in the distances matrix.

    recall_index : list of list
        Index of each recalled item.

    pool_test : list of list, optional
        Test value for each item in the pool.

    recall_test : list of list, optional
        Test value for each recalled item.

    test : callable
        Called as test(prev, curr) or test(prev, poss) to screen
        actual and possible transitions, respectively.

    count_unique : bool, optional
        If true, only unique values will be counted toward the possible
        transitions. If multiple items are avilable for recall for a
        given transition and a given bin, that bin will only be
        incremented once. If false, all possible transitions will add
        to the count.

    Returns
    -------
    actual : pandas.Series
        Count of actual transitions made for each bin.

    possible : pandas.Series
        Count of possible transitions for each bin.

    See Also
    --------
    rank_distance : Calculate percentile rank of transition distances.

    Examples
    --------
    >>> import numpy as np
    >>> from psifr import transitions
    >>> distances = np.array([[0, 1, 2, 2], [1, 0, 2, 2], [2, 2, 0, 3], [2, 2, 3, 0]])
    >>> edges = np.array([0.5, 1.5, 2.5, 3.5])
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> pool_index = [[0, 1, 2, 3]]
    >>> recall_index = [[3, 1, 2, 0]]
    >>> actual, possible = transitions.count_distance(
    ...     distances, edges, pool_items, recall_items, pool_index, recall_index
    ... )
    >>> actual
    (0.5, 1.5]    0
    (1.5, 2.5]    3
    (2.5, 3.5]    0
    dtype: int64
    >>> possible
    (0.5, 1.5]    1
    (1.5, 2.5]    4
    (2.5, 3.5]    1
    dtype: int64
    """
    list_actual = []
    list_possible = []
    centers = edges[:-1] + np.diff(edges) / 2
    for i in range(len(recall_items)):
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(
            pool_items[i],
            recall_items[i],
            pool_index[i],
            recall_index[i],
            pool_test_list,
            recall_test_list,
            test,
        )
        for output, prev, curr, poss in masker:
            prev = int(prev)
            curr = int(curr)
            poss = poss.astype(int)

            list_actual.append(distances[prev, curr])
            tran_poss = distances[prev, poss]
            if count_unique:
                # get count of each possible bin
                bin_count_poss = np.histogram(tran_poss, edges)[0]

                # for each bin that was possible, add the center as a
                # possible transition
                tran_poss = centers[np.nonzero(bin_count_poss)[0]]
            list_possible.extend(tran_poss)

    actual = pd.cut(list_actual, edges).value_counts()
    possible = pd.cut(list_possible, edges).value_counts()
    return actual, possible


def rank_distance(
    distances,
    pool_items,
    recall_items,
    pool_index,
    recall_index,
    pool_test=None,
    recall_test=None,
    test=None,
):
    """
    Calculate percentile rank of transition distances.

    Parameters
    ----------
    distances : numpy.array
        Items x items matrix of pairwise distances or similarities.

    pool_items : list of list
        Unique item codes for each item in the pool available for recall.

    recall_items : list of list
        Unique item codes of recalled items.

    pool_index : list of list
        Index of each item in the distances matrix.

    recall_index : list of list
        Index of each recalled item.

    pool_test : list of list, optional
        Test value for each item in the pool.

    recall_test : list of list, optional
        Test value for each recalled item.

    test : callable
        Called as test(prev, curr) or test(prev, poss) to screen
        actual and possible transitions, respectively.

    Returns
    -------
    rank : list
        Distance percentile rank for each included transition. The
        rank is 0 if the distance was the largest of the available
        transitions, and 1 if the distance was the smallest. Ties are
        assigned to the average percentile rank.

    See Also
    --------
    count_distance : Count transitions within distance bins.

    Examples
    --------
    >>> import numpy as np
    >>> from psifr import transitions
    >>> distances = np.array([[0, 1, 2, 2], [1, 0, 2, 2], [2, 2, 0, 3], [2, 2, 3, 0]])
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> pool_index = [[0, 1, 2, 3]]
    >>> recall_index = [[3, 1, 2, 0]]
    >>> transitions.rank_distance(
    ...     distances, pool_items, recall_items, pool_index, recall_index
    ... )
    [0.75, 0.0, nan]
    """
    rank = []
    for i in range(len(recall_items)):
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(
            pool_items[i],
            recall_items[i],
            pool_index[i],
            recall_index[i],
            pool_test_list,
            recall_test_list,
            test,
        )
        for output, prev, curr, poss in masker:
            prev = int(prev)
            curr = int(curr)
            poss = poss.astype(int)
            actual = distances[prev, curr]
            possible = distances[prev, poss]
            rank.append(1 - percentile_rank(actual, possible))
    return rank


def rank_distance_shifted(
    distances,
    max_shift,
    pool_items,
    recall_items,
    pool_index,
    recall_index,
    pool_test=None,
    recall_test=None,
    test=None,
):
    """
    Calculate percentile rank of shifted distances.

    Parameters
    ----------
    distances : numpy.array
        Items x items matrix of pairwise distances or similarities.

    max_shift : int
        Maximum number of items back for which to rank distances.

    pool_items : list of list
        Unique item codes for each item in the pool available for recall.

    recall_items : list of list
        Unique item codes of recalled items.

    pool_index : list of list
        Index of each item in the distances matrix.

    recall_index : list of list
        Index of each recalled item.

    pool_test : list of list, optional
        Test value for each item in the pool.

    recall_test : list of list, optional
        Test value for each recalled item.

    test : callable
        Called as test(prev, curr) or test(prev, poss) to screen
        actual and possible transitions, respectively.

    Returns
    -------
    rank : numpy.ndarray
        [transitions x max_shift] array with distance percentile ranks.
        The rank is 0 if the distance was the largest of the available
        transitions, and 1 if the distance was the smallest. Ties are
        assigned to the average percentile rank.

    See Also
    --------
    rank_distance - Percentile rank of transition distances relative
        to the immediately preceding item only.

    Examples
    --------
    >>> import numpy as np
    >>> from psifr import transitions
    >>> distances = np.array(
    ...     [
    ...         [0, 1, 2, 2, 2],
    ...         [1, 0, 2, 2, 2],
    ...         [2, 2, 0, 3, 3],
    ...         [2, 2, 3, 0, 2],
    ...         [2, 2, 3, 2, 0],
    ...     ]
    ... )
    >>> pool_items = [[1, 2, 3, 4, 5]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> pool_index = [[0, 1, 2, 3, 4]]
    >>> recall_index = [[3, 1, 2, 0]]
    >>> transitions.rank_distance_shifted(
    ...     distances, 2, pool_items, recall_items, pool_index, recall_index
    ... )
    array([[0.  , 0.25],
           [1.  , 1.  ]])
    """
    rank = []
    for i in range(len(recall_items)):
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = sequences_masker(
            max_shift,
            pool_items[i],
            recall_items[i],
            pool_index[i],
            recall_index[i],
            pool_test_list,
            recall_test_list,
            test,
        )
        for s_output, s_prev, s_curr, s_poss in masker:
            curr = int(s_curr[-1])
            poss = s_poss[-1].astype(int)
            rank_shift = []
            for j in range(max_shift, 0, -1):
                # rank previous to current distance for this shift
                prev = int(s_prev[-j])
                actual = distances[prev, curr]
                possible = distances[prev, poss]
                rank_shift.append(1 - percentile_rank(actual, possible))
            rank.append(rank_shift)
    rank = np.array(rank)
    return rank


def rank_distance_window(
    distances,
    list_length,
    window_lags,
    pool_items,
    recall_items,
    pool_index,
    recall_index,
    pool_test=None,
    recall_test=None,
    test=None,
):
    """
    Calculate percentile rank of distances relative to items in a window.

    Parameters
    ----------
    distances : numpy.array
        Items x items matrix of pairwise distances or similarities.

    list_length : int
        Number of items in each list.

    window_lags : array_like
        Serial position lags to include in the window.

    pool_items : list of list
        Unique item codes for each item in the pool available for recall.

    recall_items : list of list
        Unique item codes of recalled items.

    pool_index : list of list
        Index of each item in the distances matrix.

    recall_index : list of list
        Index of each recalled item.

    pool_test : list of list, optional
        Test value for each item in the pool.

    recall_test : list of list, optional
        Test value for each recalled item.

    test : callable
        Called as test(prev, curr) or test(prev, poss) to screen
        actual and possible transitions, respectively.

    Returns
    -------
    rank : numpy.ndarray
        [transitions x window lags] array with distance percentile ranks.
        The rank is 0 if the distance was the largest of the available
        transitions, and 1 if the distance was the smallest. Ties are
        assigned to the average percentile rank.
    """
    rank = []
    for i in range(len(recall_items)):
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = windows_masker(
            list_length,
            window_lags,
            pool_items[i],
            recall_items[i],
            pool_index[i],
            recall_index[i],
            pool_test_list,
            recall_test_list,
            test,
        )
        for output, w_prev, curr, poss in masker:
            rank_lag = []
            for prev in w_prev:
                actual = distances[prev, curr]
                possible = distances[prev, poss]
                rank_lag.append(1 - percentile_rank(actual, possible))
            rank.append(rank_lag)
    rank = np.array(rank)
    return rank


def count_category(
    pool_items,
    recall_items,
    pool_category,
    recall_category,
    pool_test=None,
    recall_test=None,
    test=None,
):
    """
    Count within-category transitions.

    Parameters
    ----------
    pool_items : list
        List of the serial positions available for recall in each list.
        Must match the serial position codes used in `recall_items`.

    recall_items : list
        List indicating the serial position of each recall in output
        order (NaN for intrusions).

    pool_category : list
        List of the category of each item in the pool for each list.

    recall_category : list
        List of item category in recall order.

    pool_test : list, optional
         List of some test value for each item in the pool.

    recall_test : list, optional
        List of some test value for each recall attempt by output
        position.

    test : callable
        Callable that evaluates each transition between items n and
        n+1. Must take test values for items n and n+1 and return True
        if a given transition should be included.

    Returns
    -------
    actual : int
        Count of actual within-category transitions.

    possible : int
        Count of possible within-category transitions.

    Examples
    --------
    >>> from psifr import transitions
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 3, 1, 2]]
    >>> pool_category = [[1, 1, 2, 2]]
    >>> recall_category = [[2, 2, 1, 1]]
    >>> transitions.count_category(
    ...     pool_items, recall_items, pool_category, recall_category
    ... )
    (2, 2)
    """
    actual = 0
    possible = 0
    for i in range(len(recall_items)):
        # set up masker to filter transitions
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(
            pool_items[i],
            recall_items[i],
            pool_category[i],
            recall_category[i],
            pool_test_list,
            recall_test_list,
            test,
        )

        for output, prev, curr, poss in masker:
            if prev == curr:
                actual += 1
            if np.any(prev == poss):
                possible += 1
    return actual, possible


def count_pairs(
    n_item, pool_items, recall_items, pool_test=None, recall_test=None, test=None
):
    """Count transitions between pairs of specific items."""
    actual = np.zeros((n_item, n_item), dtype=int)
    possible = np.zeros((n_item, n_item), dtype=int)
    for i, recall_items_list in enumerate(recall_items):
        # set up masker to filter transitions
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(
            pool_items[i],
            recall_items_list,
            pool_items[i],
            recall_items_list,
            pool_test_list,
            recall_test_list,
            test,
        )

        for output, prev, curr, poss in masker:
            actual[prev, curr] += 1
            possible[prev, poss] += 1
    return actual, possible
