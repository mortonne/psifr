"""Alternate module for analyzing transitions."""

import numpy as np
import pandas as pd


def _transition_masker(df, test=None):
    """Iterate over transitions with masking and exclusion of repeats.

    Parameters
    ----------
    seq : sequence
        Sequence of item identifiers. IDs must be unique within list.

    possible : sequence
        List of all possible items that may be transitioned to next.
        After an item has been iterated through, it will be removed
        from the `possible` list to exclude repeats.

    test_values : sequence, optional
        Array of values to use to test whether a transition is valid.

    test : callable, optional
        Callable to test whether a given transition is valid. Will be
        passed the previous and current item IDs or other test values
        (if `test_values` are specified; see below).

    Yields
    ------
    current : hashable
        ID for the current item in the sequence.

    actual : hashable
        ID for the next item in the sequence.

    possible : sequence
        IDs for all remaining possible items.
    """

    n = 0
    while n < (df['output'].max() - 1):
        prev = df.iloc[n]
        curr = df.iloc[n + 1]
        poss = df.iloc[(n + 1):]
        n += 1

        if not prev['_from_mask']:
            continue

        if not curr['_to_mask']:
            continue

        if test is not None:
            if not test(prev, curr):
                continue
            valid = poss.loc[test(prev, poss) & poss['_to_mask']]
        else:
            valid = poss.loc[poss['_to_mask']]

        # return the current item, actual next item,
        # and possible next items
        yield prev, curr, valid


def _subj_lag_crp(df, test=None):

    list_length = df['input'].max()
    list_actual = []
    list_possible = []
    for i, rec in df.groupby('list'):
        for prev, curr, poss in _transition_masker(rec, test):
            list_actual.append(curr['input'] - prev['input'])
            list_possible.extend(poss.loc[:, 'input'] - prev['input'])

    lags = np.arange(-list_length + 1, list_length + 1)
    actual = np.histogram(list_actual, lags)[0]
    possible = np.histogram(list_possible, lags)[0]
    results = pd.DataFrame({'actual': actual, 'possible': possible,
                            'prob': actual / possible}, index=lags[:-1])
    return results
