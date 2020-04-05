"""Module to analyze transitions during free recall."""

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


def count_lags(pool_items, recall_items, pool_test=None, recall_test=None,
               test=None):
    """Count actual and possible serial position lags.

    Parameters
    ----------
    pool_items : list
        List of the serial positions available for recall in each list.
        Must match the serial position codes used in `recall_items`.

    recall_items : list
        List indicating the serial position of each recall in output
        order (NaN for intrusions).

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

    list_actual = []
    list_possible = []
    for i, recall_items_list in enumerate(recall_items):
        # set up masker to filter transitions
        pool_test_list = None if pool_test is None else pool_test[i]
        recall_test_list = None if recall_test is None else recall_test[i]
        masker = transitions_masker(pool_items, recall_items_list,
                                    pool_items, recall_items_list,
                                    pool_test_list, recall_test_list, test)

        for prev, curr, poss in masker:
            # for this step, calculate actual lag and all possible lags
            list_actual.append(curr - prev)
            list_possible.extend(poss - prev)

    # count the actual and possible transitions for each lag
    max_lag = np.max(pool_items) - np.min(pool_items)
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

    def __init__(self, data, items_key, label_key, test_key=None, test=None):

        self.data = data
        self.keys = {'items': items_key, 'label': label_key, 'test': test_key}
        self.items_key = items_key
        self.label_key = label_key
        self.test_key = test_key
        self.test = test

    def split_lists(self, data, phase):
        """Get relevant fields and split by list."""

        if phase == 'input':
            phase_data = data.query('repeat == 0 and ~intrusion')
            phase_data = phase_data.sort_values('list')
        elif phase == 'output':
            phase_data = data.query('recalled')
            phase_data = phase_data.sort_values(['list', 'input'])
        else:
            raise ValueError(f'Invalid phase: {phase}')

        indices = phase_data.reset_index().groupby('list').indices

        split = {}
        for name, val in self.keys.items():
            if val == 'position':
                key = phase
            else:
                key = val

            if key is None:
                continue
            all_values = phase_data[val].to_numpy()
            split[name] = [all_values[ind] for name, ind in indices.items()]
        return split

    def get_subject_lists(self, subject):
        """Get pool and recall information by list for one subject."""
        # filter for this subject
        subject = self.data.query(f'subject == {subject}')

        # get pool information by excluding invalid recalls
        pool = subject.query('repeat == 0 and ~intrusion').sort_values('list')
        pool_lists = self.split_lists(pool)

        # get recall information in sorted order
        recall = subject.query('recalled').sort_values(['list', 'output'])
        recall_lists = self.split_lists(recall)

        return pool_lists, recall_lists
