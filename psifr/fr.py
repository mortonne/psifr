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


def _prep_column(df, col_spec, default_key, default_mask=None):
    """Set up a column in a data frame."""

    if isinstance(col_spec, str):
        col_key = col_spec
    else:
        col_key = default_key
        if isinstance(col_spec, pd.Series):
            df[col_key] = col_spec
        elif callable(col_spec):
            df[col_key] = col_spec(df)
        elif default_mask is not None:
            if isinstance(default_mask, str):
                col_key = default_mask
            elif callable(default_mask):
                df[col_key] = default_mask(df)
            else:
                raise ValueError('Invalid default mask specification.')
        else:
            raise ValueError('Invalid mask specification.')
    return df, col_key


def _subject_lag_crp(df, test_values=None, test=None, first_output=None):
    """Calculate lag-CRP for one subject."""

    # sort by output so we can extract recall sequences
    df_output = df.query('recalled').sort_values(['list', 'output'])
    if first_output is not None:
        df_output = df_output.query(f'output >= {first_output}')

    # recall pool
    list_length = int(df['input'].max())
    pool_items = list(range(1, list_length + 1))

    # recall input positions
    all_recalls = df_output['input'].to_numpy()
    indices = df_output.reset_index().groupby(['list']).indices
    recall_items = [all_recalls[ind] for name, ind in indices.items()]

    if test_values is not None:
        # create column if necessary
        df, test_key = _prep_column(df, test_values, '_test_values')

        # get pool values
        study_df = (df.query('repeat == 0 and ~intrusion')
                    .sort_values('input'))
        study_values = study_df[test_key].to_numpy()
        study_indices = study_df.reset_index().groupby('list').indices
        pool_test = [study_values[ind].tolist()
                     for name, ind in study_indices.items()]

        # get recall values
        all_values = df_output[test_key].to_numpy()
        recall_test = [all_values[ind] for name, ind in indices.items()]
    else:
        pool_test = None
        recall_test = None

    # count lags
    actual, possible = transitions.count_lags(pool_items, recall_items,
                                              pool_test, recall_test, test)

    # calculate and annotate crp
    subject = df['subject'].iloc[0]
    crp = pd.DataFrame({'subject': subject, 'lag': actual.index,
                        'prob': actual / possible,
                        'actual': actual, 'possible': possible})
    crp = crp.set_index(['subject', 'lag'])
    return crp


def lag_crp(df, test_values=None, test=None, first_output=None):
    """Lag-CRP for multiple subjects.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists. List length is
        assumed to be the same for all lists within each subject.
        Must have fields: subject, list, input, output, recalled.
        Input position must be defined such that the first serial
        position is 1, not 0.

    test_values : pandas.Series or column name, optional
        Column with labels to use when testing transitions for
        inclusion.

    test : callable, optional
        Callable that takes in previous and current item values and
        returns True for transitions that should be included.

    first_output : int, optional
        First output position to include when calculating transition
        probabilities. Used to exclude initial outputs. Default is
        to start at the first recall on each list.

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

    subj_results = []
    for subject, subj_df in df.groupby('subject'):
        results = _subject_lag_crp(subj_df, test_values, test, first_output)
        subj_results.append(results)
    crp = pd.concat(subj_results, axis=0)
    return crp
