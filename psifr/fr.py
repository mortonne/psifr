"""Utilities for working with free recall data."""

import numpy as np
import pandas as pd
from . import transitions


def check_data(df):
    """Run checks on free recall data.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains one row for each trial (study and recall). Must have fields:
            subject : number or str
                Subject identifier.
            list : number
                List identifier. This applies to both study and recall trials.
            trial_type : str
                Type of trial; may be 'study' or 'recall'.
            position : number
                Position within the study list or recall sequence.
            item : str
                Item that was either presented or recalled on this trial.
    """

    # check that all fields are accounted for
    columns = ['subject', 'list', 'trial_type', 'position', 'item']
    for col in columns:
        assert col in df.columns, f'Required column {col} is missing.'

    # only one column has a hard constraint on its exact content
    assert df['trial_type'].isin(['study', 'recall']).all(), (
        'trial_type for all trials must be "study" or "recall".')


def block_index(list_labels):
    """Get index of each block in a list."""

    prev_label = ''
    curr_block = 0
    block = np.zeros(len(list_labels), dtype=int)
    for i, label in enumerate(list_labels):
        if prev_label != label:
            curr_block += 1
        block[i] = curr_block
        prev_label = label
    return block


def merge_lists(study, recall, merge_keys=None, list_keys=None, study_keys=None,
                recall_keys=None, position_key='position'):
    """Merge study and recall events together for each list.

    Parameters
    ----------
    study : pandas.DataFrame
        Information about all study events. Should have one row for
        each study event.

    recall : pandas.DataFrame
        Information about all recall events. Should have one row for
        each recall attempt.

    merge_keys : list, optional
        Columns to use to designate events to merge. Default is
        ['subject', 'list', 'item'], which will merge events related to
        the same item, but only within list.

    list_keys : list, optional
        Columns that apply to both study and recall events.

    study_keys : list, optional
        Columns that only apply to study events.

    recall_keys : list, optional
        Columns that only apply to recall events.

    position_key : str, optional
        Column indicating the position of each item in either the study
        list or the recall sequence.

    Returns
    -------
    merged : pandas.DataFrame
        Merged information about study and recall events. Each row
        corresponds to one unique input/output pair.

        The following columns will be added:
        input : int
            Position of each item in the input list (i.e., serial
            position).
        output : int
            Position of each item in the recall sequence.
        recalled : bool
            True for rows with an associated recall event.
        repeat : int
            Number of times this recall event has been repeated (0 for
            the first recall of an item).
        intrusion : bool
            True for recalls that do not correspond to any study event.
    """

    if merge_keys is None:
        merge_keys = ['subject', 'list', 'item']

    if list_keys is None:
        list_keys = []

    if study_keys is None:
        study_keys = []

    if recall_keys is None:
        recall_keys = []

    # get running count of number of times each item is recalled in each list
    recall.loc[:, 'repeat'] = recall.groupby(merge_keys).cumcount()

    # get just the fields to use in the merge
    study = study[merge_keys + ['position'] + list_keys + study_keys]
    recall = recall[merge_keys + ['repeat', 'position'] + list_keys +
                    recall_keys]

    # merge information from study and recall trials
    merged = pd.merge(study, recall, left_on=merge_keys + list_keys,
                      right_on=merge_keys + list_keys, how='outer')

    # position from study events indicates input position;
    # position from recall events indicates output position
    merged = merged.rename(columns={position_key + '_x': 'input',
                                    position_key + '_y': 'output'})

    # field to indicate whether a given item was recalled
    merged.loc[:, 'recalled'] = merged['output'].notna().astype('bool')

    # field to indicate whether a given recall was an intrusion
    merged.loc[:, 'intrusion'] = merged['input'].isna().astype('bool')

    # fix repeats field to define for non-recalled items
    merged.loc[merged['repeat'].isna(), 'repeat'] = 0
    merged = merged.astype({'repeat': 'int'})

    # reorder columns
    columns = (merge_keys + ['input', 'output'] +
               ['recalled', 'repeat', 'intrusion'] +
               list_keys + study_keys + recall_keys)
    merged = merged.reindex(columns=columns)

    # sort rows in standard order
    sort_keys = merge_keys.copy() + ['input']
    sort_keys.remove('item')
    merged = merged.sort_values(by=sort_keys, ignore_index=True)

    return merged


def get_study_value(df, column, list_cols=None):
    """Get study column value by list."""

    if list_cols is None:
        list_cols = ['list']

    pres_df = df.query('repeat == 0 and ~intrusion').sort_values('input')
    values = [pres[column].to_numpy()
              for name, pres in pres_df.groupby(list_cols)]
    return values


def get_recall_index(df, list_cols=None):
    """Get recall input position index by list."""

    if list_cols is None:
        list_cols = ['list']

    # get just recall trials and sort by output position
    rec_df = df.query('recalled').sort_values('output')

    # compile recalls for each list
    recalls = []
    for name, rec in rec_df.groupby(list_cols):
        assert (rec['output'].diff()[1:] == 1).all(), (
            'There are gaps in the recall sequence.')

        # get recalls as a list of recall input indices
        input_ind = rec['input'] - 1
        recalls.append(input_ind.to_numpy())
    return recalls


def get_recall_mask(df, mask_spec, list_cols=None):
    """Create a recall mask from a data frame."""

    if list_cols is None:
        list_cols = ['list']

    # get just recall trials and sort by output position
    rec_df = df.query('recalled').sort_values('output')

    # unpack spec
    if not callable(mask_spec) or isinstance(mask_spec, str):
        raise ValueError('Invalid mask specification.')

    # get mask for each list
    mask = []
    for name, rec in rec_df.groupby(list_cols):
        if callable(mask_spec):
            list_mask = mask_spec(rec)
        else:
            list_mask = rec[mask_spec].to_numpy()
        mask.append(list_mask)
    return mask


def _transition_masker(seq, possible, test_values=None, test=None):
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

    if test is not None and test_values is None:
        raise ValueError('If test is specified, must specify test values.')

    n = 0
    possible = possible.copy()
    while n < (len(seq) - 1):
        if np.isnan(seq[n]) or np.isnan(seq[n + 1]):
            n += 1
            continue

        if seq[n] in possible:
            # remove item from consideration on future transitions
            possible.remove(seq[n])
        else:
            # item recalled was not in the set of possible items
            n += 1
            continue

        if seq[n + 1] not in possible:
            n += 1
            continue

        if test is not None:
            # check if this transition is included
            if not test(test_values[n], test_values[n + 1]):
                n += 1
                continue

        prev = int(seq[n])
        curr = int(seq[n + 1])
        n += 1

        # run dynamic check if applicable
        valid = np.array(possible)
        if test is not None:
            # filter possible recalls to get only included ones
            valid = valid[test(test_values[n], test_values[possible])]

        # return the current item, actual next item,
        # and possible next items
        yield prev, curr, valid


def _masker_opt(df, input_key=None, input_test=None,
                output_key=None, output_test=None,
                from_mask=None, to_mask=None,
                list_cols=None):
    """Define masker settings for a data frame."""

    if list_cols is None:
        list_cols = ['list']

    opt = {}
    if input_key is not None:
        opt['input_values'] = get_study_value(df, input_key, list_cols)
        opt['input_test'] = input_test

    if output_key is not None:
        rec_df = df.query('recalled').sort_values('output')
        values = []
        for name, rec in rec_df.groupby(list_cols):
            values.append(rec[output_key])
        opt['output_values'] = values
        opt['output_test'] = output_test

    if from_mask is not None:
        opt['from_mask'] = get_recall_mask(df, from_mask, list_cols)

    if to_mask is not None:
        opt['to_mask'] = get_recall_mask(df, to_mask, list_cols)
    return opt


def _subject_lag_crp(list_recalls, list_length, test_values=None, test=None):
    """Conditional response probability by lag for one subject.

    Parameters
    ----------
    list_recalls : sequence
        Recall sequence for each list. Should have one element for each
        list, which should contain a list of serial position indices.

    list_length : int
        Length of each list. All lists must have the same length.

    masker_kws : dict
        Options to pass to _transition_masker.

    Returns
    -------
    actual : pandas.Series
        Count of times each lag transition was actually made, among
        included transitions.

    possible : pandas.Series
        Count of times each lag transition could have been made, given
        items that had not yet been recalled at each output position.
    """

    list_actual = []
    list_possible = []
    for i, recalls in enumerate(list_recalls):
        # calculate actual and possible lags for each included transition
        possible_recalls = list(range(list_length))

        # set masker options
        opt = {}
        if masker_kws is not None:
            for key, val in masker_kws.items():
                opt[key] = val if callable(val) else np.array(val[i])

        # iterate over valid transitions
        for prev, curr, poss in _transition_masker(recalls, possible_recalls,
                                                   **opt):
            list_actual.append(curr - prev)
            list_possible.extend(np.subtract(poss, prev))

    # tally all transitions
    lags = np.arange(-list_length + 1, list_length + 1)
    actual = pd.Series(np.histogram(list_actual, lags)[0], index=lags[:-1])
    possible = pd.Series(np.histogram(list_possible, lags)[0], index=lags[:-1])
    return actual, possible


def lag_crp(df, from_mask_def=None, to_mask_def=None, test_key=None,
            test=None):
    """Lag-CRP for multiple subjects.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists. List length is
        assumed to be the same for all lists within each subject.
        Must have fields: subject, list, input, output, recalled.
        Input position must be defined such that the first serial
        position is 1, not 0.

    from_mask_def : str or callable, optional
        Specification for boolean mask to exclude output positions
        being transitioned from. If str, will select a column from
        `df`. If callable, will run with `df` as input and must return
        a boolean Series. Default is to exclude repeats and intrusions.

    to_mask_def : str or callable, optional
        Specification for a boolean mask to exclude output positions
        being transitioned to. Default is to exclude repeats and
        intrusions.

    test_key : str, optional
        Column with labels to use when testing transitions for
        inclusion.

    test : callable, optional
        Callable that takes in previous and current item values and
        returns True for transitions that should be included.

    Returns
    -------
    results : pandas.DataFrame
        Has fields:
        subject : hashable
            Results are separated by each subject.
        lag : int
            Lag of input position between two adjacent recalls.
        prob : float
            Probability of each lag transition.
        actual : int
            Total of actual made transitions at each lag.
        possible : int
            Total of times each lag was possible, given the prior
            input position and the remaining items to be recalled.
    """

    # define masks
    df.loc[:, '_from_mask'] = (df['repeat'] == 0) & ~df['intrusion']
    df.loc[:, '_to_mask'] = (df['repeat'] == 0) & ~df['intrusion']

    subj_results = []
    df = df.sort_values(['subject', 'list', 'output'])
    for subject, subj_df in df.groupby('subject'):
        # get recall events for each list
        list_length = int(subj_df['input'].max())

        # compile recalls for each list
        recalls = []
        from_mask = []
        to_mask = []
        if test_key is not None:
            test_values = []
        else:
            test_values = None
        n_recall = subj_df.groupby('list')['output'].max().to_numpy()
        for name, rec in subj_df.groupby(['list']):
            assert (rec['output'].diff().dropna() == 1).all(), (
                'There are gaps in the recall sequence.')

            # get recalls as a list of recall input indices
            recalls.append(rec['input'].to_numpy())

            # static masks defining valid recalls
            from_mask.append(rec['_from_mask'].to_numpy())
            to_mask.append(rec['_to_mask'].to_numpy())

            # dynamic mask defining included transitions
            if test_key is not None:
                test_values.append(rec[test_key])

        # calculate frequency of each lag
        actual, possible = transitions.count_lags(recalls, list_length, n_recall,
                                                  from_mask, to_mask, test_values, test)
        results = pd.DataFrame({'subject': subject, 'lag': actual.index,
                                'prob': actual / possible, 'actual': actual,
                                'possible': possible})
        results = results.set_index(['subject', 'lag'])
        subj_results.append(results)
    return pd.concat(subj_results, axis=0)
