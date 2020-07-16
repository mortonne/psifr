"""Module to analyze transitions during free recall."""

import abc
import numpy as np
from scipy import stats
import pandas as pd
from psifr import fr


def percentile_rank(actual, possible):
    """Get percentile rank of a score compared to possible scores."""
    possible_rank = stats.rankdata(possible)
    actual_rank = possible_rank[actual == np.asarray(possible)]
    possible_count = np.count_nonzero(~np.isnan(possible))
    if possible_count == 1:
        return np.nan
    rank = (actual_rank - 1) / (possible_count - 1)
    return rank[0]


def outputs_masker(pool_items, recall_items, pool_output, recall_output,
                   pool_test=None, recall_test=None, test=None):
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


def count_outputs(list_length, pool_items, recall_items, pool_label,
                  recall_label, pool_test=None, recall_test=None,
                  test=None, count_unique=False):
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
        masker = outputs_masker(pool_items[i], recall_items_list,
                                pool_label[i], recall_label[i],
                                pool_test_list, recall_test_list, test)

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


def transitions_masker(pool_items, recall_items, pool_output, recall_output,
                       pool_test=None, recall_test=None, test=None):
    """Iterate over transitions with masking.

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
            ind = test(recall_test[n], np.array(pool_test))
            if not isinstance(ind, np.ndarray):
                ind = np.repeat(ind, poss.shape)
            poss = poss[ind]
        n += 1
        yield prev, curr, poss


def count_lags(list_length, pool_items, recall_items,
               pool_label=None, recall_label=None,
               pool_test=None, recall_test=None, test=None):
    """Count actual and possible serial position lags.

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
        masker = transitions_masker(pool_items[i], recall_items_list,
                                    pool_label[i], recall_label[i],
                                    pool_test_list, recall_test_list, test)

        for prev, curr, poss in masker:
            # for this step, calculate actual lag and all possible lags
            list_actual.append(curr - prev)
            list_possible.extend(poss - prev)

    # count the actual and possible transitions for each lag
    max_lag = list_length - 1
    lags = np.arange(-max_lag, max_lag + 2)
    actual = pd.Series(np.histogram(list_actual, lags)[0],
                       index=lags[:-1])
    possible = pd.Series(np.histogram(list_possible, lags)[0],
                         index=lags[:-1])
    return actual, possible


def rank_lags(pool_items, recall_items, pool_label=None, recall_label=None,
              pool_test=None, recall_test=None, test=None):
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
        masker = transitions_masker(pool_items[i], recall_items_list,
                                    pool_label[i], recall_label[i],
                                    pool_test_list, recall_test_list, test)

        for prev, curr, poss in masker:
            actual = np.abs(curr - prev)
            possible = np.abs(poss - prev)
            rank.append(1 - percentile_rank(actual, possible))
    return rank


def count_distance(distances, edges, pool_items, recall_items,
                   pool_index, recall_index,
                   pool_test=None, recall_test=None, test=None,
                   count_unique=False):
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
    """

    list_actual = []
    list_possible = []
    for i in range(len(recall_items)):
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(pool_items[i], recall_items[i],
                                    pool_index[i], recall_index[i],
                                    pool_test_list, recall_test_list, test)
        for prev, curr, poss in masker:
            prev = int(prev)
            curr = int(curr)
            poss = poss.astype(int)

            list_actual.append(distances[prev, curr])
            if count_unique:
                list_possible.extend(np.unique(distances[prev, poss]).tolist())
            else:
                list_possible.extend(distances[prev, poss])

    actual = pd.cut(list_actual, edges).value_counts()
    possible = pd.cut(list_possible, edges).value_counts()
    return actual, possible


def rank_distance(distances, pool_items, recall_items,
                  pool_index, recall_index,
                  pool_test=None, recall_test=None, test=None):
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
    """
    rank = []
    for i in range(len(recall_items)):
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(pool_items[i], recall_items[i],
                                    pool_index[i], recall_index[i],
                                    pool_test_list, recall_test_list, test)
        for prev, curr, poss in masker:
            prev = int(prev)
            curr = int(curr)
            poss = poss.astype(int)
            actual = distances[prev, curr]
            possible = distances[prev, poss]
            rank.append(1 - percentile_rank(actual, possible))
    return rank


def count_category(pool_items, recall_items, pool_category,
                   recall_category, pool_test=None,
                   recall_test=None, test=None):
    """Count within-category transitions."""

    actual = 0
    possible = 0
    for i in range(len(recall_items)):
        # set up masker to filter transitions
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(pool_items[i], recall_items[i],
                                    pool_category[i], recall_category[i],
                                    pool_test_list, recall_test_list, test)

        for prev, curr, poss in masker:
            if prev == curr:
                actual += 1
            if np.any(prev == poss):
                possible += 1
    return actual, possible


def count_pairs(n_item, pool_items, recall_items,
                pool_test=None, recall_test=None, test=None):
    """Count transitions between pairs of specific items."""

    actual = np.zeros((n_item, n_item), dtype=int)
    possible = np.zeros((n_item, n_item), dtype=int)
    for i, recall_items_list in enumerate(recall_items):
        # set up masker to filter transitions
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(pool_items[i], recall_items_list,
                                    pool_items[i], recall_items_list,
                                    pool_test_list, recall_test_list, test)

        for prev, curr, poss in masker:
            actual[prev, curr] += 1
            possible[prev, poss] += 1
    return actual, possible


class TransitionMeasure(object):

    def __init__(self, items_key, label_key, item_query=None, test_key=None,
                 test=None):

        self.keys = {'items': items_key, 'label': label_key, 'test': test_key}
        self.item_query = item_query
        self.test = test

    def split_lists(self, data, phase):
        """Get relevant fields and split by list."""
        names = list(self.keys.keys())
        keys = list(self.keys.values())
        split = fr.split_lists(data, phase, keys, names, self.item_query,
                               as_list=True)
        return split

    @abc.abstractmethod
    def analyze_subject(self, subject, pool_lists, recall_lists):
        pass

    def analyze(self, data):
        subj_results = []
        for subject, subject_data in data.groupby('subject'):
            pool_lists = self.split_lists(subject_data, 'study')
            recall_lists = self.split_lists(subject_data, 'recall')
            results = self.analyze_subject(subject, pool_lists, recall_lists)
            subj_results.append(results)
        stat = pd.concat(subj_results, axis=0)
        return stat


class TransitionOutputs(TransitionMeasure):

    def __init__(self, list_length, item_query=None, test_key=None, test=None):
        super().__init__('input', 'input', item_query=item_query,
                         test_key=test_key, test=test)
        self.list_length = list_length

    def analyze_subject(self, subject, pool, recall):
        actual, possible = count_outputs(
            self.list_length, pool['items'], recall['items'],
            pool['label'], recall['label'],
            pool['test'], recall['test'], self.test
        )
        inputs = np.tile(np.arange(1, actual.shape[1] + 1), actual.shape[0])
        outputs = np.repeat(np.arange(1, actual.shape[0] + 1), actual.shape[1])
        with np.errstate(divide='ignore', invalid='ignore'):
            prob = actual.flatten() / possible.flatten()
        pnr = pd.DataFrame(
            {'subject': subject, 'input': inputs, 'output': outputs,
             'prob': prob, 'actual': actual.flatten(),
             'possible': possible.flatten()}
        )
        pnr = pnr.set_index(['subject', 'output', 'input'])
        return pnr


class TransitionLag(TransitionMeasure):

    def __init__(self, list_length, item_query=None, test_key=None, test=None):
        super().__init__('input', 'input', item_query=item_query,
                         test_key=test_key, test=test)
        self.list_length = list_length

    def analyze_subject(self, subject, pool, recall):

        actual, possible = count_lags(self.list_length,
                                      pool['items'], recall['items'],
                                      pool['label'], recall['label'],
                                      pool['test'], recall['test'], self.test)
        crp = pd.DataFrame({'subject': subject, 'lag': actual.index,
                            'prob': actual / possible,
                            'actual': actual, 'possible': possible})
        crp = crp.set_index(['subject', 'lag'])
        return crp


class TransitionLagRank(TransitionMeasure):

    def __init__(self, item_query=None, test_key=None, test=None):
        super().__init__('input', 'input', item_query=item_query,
                         test_key=test_key, test=test)

    def analyze_subject(self, subject, pool, recall):
        ranks = rank_lags(
            pool['items'], recall['items'], pool['label'], recall['label'],
            pool['test'], recall['test'], self.test
        )
        stat = pd.DataFrame({'subject': subject, 'rank': np.nanmean(ranks)},
                            index=[subject])
        stat = stat.set_index('subject')
        return stat


class TransitionDistance(TransitionMeasure):

    def __init__(self, index_key, distances, edges, count_unique=False,
                 centers=None, item_query=None, test_key=None, test=None):
        super().__init__('input', index_key, item_query=item_query,
                         test_key=test_key, test=test)
        self.distances = distances
        self.edges = edges
        if centers is None:
            # if no explicit centers, use halfway between edges
            centers = edges[:-1] + (np.diff(edges) / 2)
        self.centers = centers
        self.count_unique = count_unique

    def analyze_subject(self, subject, pool, recall):

        actual, possible = count_distance(
            self.distances, self.edges, pool['items'], recall['items'],
            pool['label'], recall['label'], pool['test'], recall['test'],
            self.test, count_unique=self.count_unique)
        crp = pd.DataFrame({'subject': subject, 'center': self.centers,
                            'bin': actual.index,
                            'prob': actual / possible,
                            'actual': actual, 'possible': possible})
        crp = crp.set_index(['subject', 'center'])
        return crp


class TransitionDistanceRank(TransitionMeasure):

    def __init__(self, index_key, distances, item_query=None,
                 test_key=None, test=None):
        super().__init__(index_key, index_key, item_query=item_query,
                         test_key=test_key, test=test)
        self.distances = distances

    def analyze_subject(self, subject, pool, recall):
        ranks = rank_distance(
            self.distances, pool['items'], recall['items'],
            pool['label'], recall['label'],
            pool['test'], recall['test'], self.test
        )
        stat = pd.DataFrame({'subject': subject, 'rank': np.nanmean(ranks)},
                            index=[subject])
        stat = stat.set_index('subject')
        return stat


class TransitionCategory(TransitionMeasure):

    def __init__(self, category_key, item_query=None, test_key=None, test=None):
        super().__init__('input', category_key, item_query=item_query,
                         test_key=test_key, test=test)

    def analyze_subject(self, subject, pool, recall):
        actual, possible = count_category(
            pool['items'], recall['items'],
            pool['label'], recall['label'], pool['test'],
            recall['test'], self.test)
        crp = pd.DataFrame({'subject': subject, 'prob': actual / possible,
                           'actual': actual, 'possible': possible},
                           index=[subject])
        crp = crp.set_index('subject')
        return crp
