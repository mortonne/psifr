"""Calculate statistics of list recall."""
import itertools

import numpy
import numpy as np
import pandas as pd
import scipy.stats

from psifr import maskers


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
    >>> from psifr import stats
    >>> actual = 3
    >>> possible = [1, 2, 2, 2, 3]
    >>> stats.percentile_rank(actual, possible)
    1.0
    """
    possible_rank = scipy.stats.rankdata(possible)
    actual_rank = possible_rank[actual == np.asarray(possible)]
    possible_count = np.count_nonzero(~np.isnan(possible))
    if possible_count == 1:
        return np.nan
    rank = (actual_rank - 1) / (possible_count - 1)
    return rank[0]


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
    >>> from psifr import stats
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> actual, possible = stats.count_outputs(
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
    >>> from psifr import stats
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> actual, possible = stats.count_lags(4, pool_items, recall_items)
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
        masker = maskers.transitions_masker(
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
    >>> from psifr import stats
    >>> pool_items = [[1, 2, 3]]
    >>> recall_items = [[3, 1, 2]]
    >>> actual, possible = stats.count_lags_compound(3, pool_items, recall_items)
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
        masker = maskers.sequences_masker(
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
    >>> from psifr import stats
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> stats.rank_lags(pool_items, recall_items)
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
        masker = maskers.transitions_masker(
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
    >>> from psifr import stats
    >>> distances = np.array([[0, 1, 2, 2], [1, 0, 2, 2], [2, 2, 0, 3], [2, 2, 3, 0]])
    >>> edges = np.array([0.5, 1.5, 2.5, 3.5])
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> pool_index = [[0, 1, 2, 3]]
    >>> recall_index = [[3, 1, 2, 0]]
    >>> actual, possible = stats.count_distance(
    ...     distances, edges, pool_items, recall_items, pool_index, recall_index
    ... )
    >>> actual
    (0.5, 1.5]    0
    (1.5, 2.5]    3
    (2.5, 3.5]    0
    Name: count, dtype: int64
    >>> possible
    (0.5, 1.5]    1
    (1.5, 2.5]    4
    (2.5, 3.5]    1
    Name: count, dtype: int64
    """
    list_actual = []
    list_possible = []
    centers = edges[:-1] + np.diff(edges) / 2
    for i in range(len(recall_items)):
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = maskers.transitions_masker(
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
    >>> from psifr import stats
    >>> distances = np.array([[0, 1, 2, 2], [1, 0, 2, 2], [2, 2, 0, 3], [2, 2, 3, 0]])
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 2, 3, 1]]
    >>> pool_index = [[0, 1, 2, 3]]
    >>> recall_index = [[3, 1, 2, 0]]
    >>> stats.rank_distance(
    ...     distances, pool_items, recall_items, pool_index, recall_index
    ... )
    [0.75, 0.0, nan]
    """
    rank = []
    for i in range(len(recall_items)):
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = maskers.transitions_masker(
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
    >>> from psifr import stats
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
    >>> stats.rank_distance_shifted(
    ...     distances, 2, pool_items, recall_items, pool_index, recall_index
    ... )
    array([[0.  , 0.25],
           [1.  , 1.  ]])
    """
    rank = []
    for i in range(len(recall_items)):
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = maskers.sequences_masker(
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
    exclude_prev_window=False,
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

    exclude_prev_window : bool, optional
        If true, transitions preceded by items in the window around the
        previous item in the list will be excluded.

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
        masker = maskers.windows_masker(
            list_length,
            window_lags,
            pool_items[i],
            recall_items[i],
            pool_index[i],
            recall_index[i],
            pool_test_list,
            recall_test_list,
            test,
            exclude_prev_window,
        )
        for output, w_prev, curr, poss in masker:
            curr = int(curr)
            rank_lag = []
            for prev in w_prev:
                prev = int(prev)
                poss = poss.astype(int)
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
    >>> from psifr import stats
    >>> pool_items = [[1, 2, 3, 4]]
    >>> recall_items = [[4, 3, 1, 2]]
    >>> pool_category = [[1, 1, 2, 2]]
    >>> recall_category = [[2, 2, 1, 1]]
    >>> stats.count_category(
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
        masker = maskers.transitions_masker(
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
        masker = maskers.transitions_masker(
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


def lbc(study_category, recall_category):
    """Calculate list-based clustering (LBC) for a set of lists."""
    lbc_scores = np.zeros(len(study_category))
    for i, (study, recall) in enumerate(zip(study_category, recall_category)):
        study = np.asarray(study)
        recall = np.asarray(recall)
        if np.any(pd.isna(study)):
            raise ValueError('Study category contains N/A values.')
        if np.any(pd.isna(recall)):
            raise ValueError('Recall category contains N/A values.')

        # number of correct recalls
        r = len(recall)

        # number of items per category
        m = len(study) / len(np.unique(study))

        # list length
        nl = len(study)

        # observed and expected clustering
        observed = np.count_nonzero(recall[:-1] == recall[1:])
        expected = ((r - 1) * (m - 1)) / (nl - 1)
        lbc_scores[i] = observed - expected
    return lbc_scores


def arc(recall_category):
    """Calculate adjusted ratio of clustering for a set of lists."""
    arc_scores = np.zeros(len(recall_category))
    for i, recall in enumerate(recall_category):
        recall = np.asarray(recall)
        if np.any(pd.isna(recall)):
            raise ValueError('Recall category contains N/A values.')

        # number of categories and correct recalls from each category
        categories = np.unique(recall)
        n = np.array([np.count_nonzero(recall == c) for c in categories])
        c = len(categories)

        # number of correct recalls
        r = len(recall)

        # observed, expected, and maximum clustering
        expected = np.sum((n * (n - 1)) / r)
        observed = np.count_nonzero(recall[:-1] == recall[1:])
        maximum = r - c
        if maximum == expected:
            # when maximum is the same as expected, ARC is undefined
            arc_scores[i] = np.nan
            continue
        arc_scores[i] = (observed - expected) / (maximum - expected)
    return arc_scores
