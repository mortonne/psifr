"""Module to analyze transitions during free recall."""

import abc
import numpy as np
import pandas as pd


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
            poss = poss[test(recall_test[n], np.array(pool_test))]
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
        masker = transitions_masker(pool_items, recall_items[i],
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

        if phase == 'input':
            phase_data = data.query('study').sort_values('list')
        elif phase == 'output':
            phase_data = data.query('recall').sort_values(['list', 'output'])
        else:
            raise ValueError(f'Invalid phase: {phase}')

        if phase == 'input' and self.item_query is not None:
            # get the subset of the pool that is of interest
            mask = phase_data.eval(self.item_query).to_numpy()
        else:
            mask = np.ones(data.shape[0], dtype=bool)

        indices = phase_data.reset_index().groupby('list').indices

        split = {}
        for name, val in self.keys.items():
            if val == 'position':
                key = phase
            else:
                if val is None:
                    split[name] = None
                    continue
                key = val

            all_values = phase_data[key].to_numpy()
            split[name] = [all_values[ind][mask[ind]].tolist()
                           for name, ind in indices.items()]
        return split

    @abc.abstractmethod
    def analyze_subject(self, subject, pool_lists, recall_lists):
        pass

    def analyze(self, data):
        subj_results = []
        for subject, subject_data in data.groupby('subject'):
            pool_lists = self.split_lists(subject_data, 'input')
            recall_lists = self.split_lists(subject_data, 'output')
            results = self.analyze_subject(subject, pool_lists, recall_lists)
            subj_results.append(results)
        stat = pd.concat(subj_results, axis=0)
        return stat


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