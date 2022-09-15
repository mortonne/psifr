"""Utilities for working with free recall data."""

from pkg_resources import resource_filename
import warnings
import numpy as np
import pandas as pd
import seaborn as sns

from psifr import measures
from psifr import clustering


def sample_data(study):
    """Read sample data."""
    data_file = resource_filename('psifr', f'data/{study}.csv')
    df = pd.read_csv(data_file)
    return df


def sample_distances(study):
    """Read sample distances."""
    distance_file = resource_filename('psifr', f'distances/{study}.npz')
    f = np.load(distance_file)
    return f['items'], f['distances']


def table_from_lists(subjects, study, recall, lists=None, **kwargs):
    """
    Create table format data from list format data.

    Parameters
    ----------
    subjects : list of hashable
        Subject identifier for each list.

    study : list of list of hashable
        List of items for each study list.

    recall : list of list of hashable
        List of recalled items for each study list.

    lists : list of hashable, optional
        List of list numbers. If not specified, lists for each subject
        will be numbered sequentially starting from one.

    Returns
    -------
    data : pandas.DataFrame
        Data in table format.

    See Also
    --------
    split_lists : Split a table into list format.

    Examples
    --------
    >>> from psifr import fr
    >>> subjects_list = [1, 1, 2, 2]
    >>> study_lists = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
    >>> recall_lists = [['b'], ['d', 'c'], ['f', 'e'], []]
    >>> fr.table_from_lists(subjects_list, study_lists, recall_lists)
        subject  list trial_type  position item
    0         1     1      study         1    a
    1         1     1      study         2    b
    2         1     1     recall         1    b
    3         1     2      study         1    c
    4         1     2      study         2    d
    5         1     2     recall         1    d
    6         1     2     recall         2    c
    7         2     1      study         1    e
    8         2     1      study         2    f
    9         2     1     recall         1    f
    10        2     1     recall         2    e
    11        2     2      study         1    g
    12        2     2      study         2    h

    >>> subjects_list = [1, 1]
    >>> study_lists = [['a', 'b'], ['c', 'd']]
    >>> recall_lists = [['b'], ['d', 'c']]
    >>> col1 = ([[1, 2], [1, 2]], [[2], [2, 1]])
    >>> col2 = ([[1, 1], [2, 2]], None)
    >>> fr.table_from_lists(subjects_list, study_lists, recall_lists, col1=col1, col2=col2)
       subject  list trial_type  position item  col1  col2
    0        1     1      study         1    a     1   1.0
    1        1     1      study         2    b     2   1.0
    2        1     1     recall         1    b     2   NaN
    3        1     2      study         1    c     1   2.0
    4        1     2      study         2    d     2   2.0
    5        1     2     recall         1    d     2   NaN
    6        1     2     recall         2    c     1   NaN
    """
    assert len(subjects) == len(study) == len(recall), 'Input lengths must match.'
    d = {'subject': [], 'list': [], 'trial_type': [], 'position': [], 'item': []}
    for key in kwargs.keys():
        d[key] = []
    prev_subject = None
    current_list = 1
    if lists is None:
        lists = [None] * len(subjects)
    else:
        assert len(subjects) == len(lists), 'Length of lists must match subjects.'
    labels = zip(subjects, study, recall, lists)
    for i, (subject, study_list, recall_list, n) in enumerate(labels):
        # set list number
        if n is not None:
            current_list = n
        elif subject != prev_subject:
            current_list = 1

        # add study events
        for j, study_item in enumerate(study_list):
            d['subject'].append(subject)
            d['list'].append(current_list)
            d['trial_type'].append('study')
            d['position'].append(j + 1)
            d['item'].append(study_item)
            for key, val in kwargs.items():
                if val[0] is not None:
                    d[key].append(val[0][i][j])
                else:
                    d[key].append(np.nan)

        # add recall events
        for j, recall_item in enumerate(recall_list):
            d['subject'].append(subject)
            d['list'].append(current_list)
            d['trial_type'].append('recall')
            d['position'].append(j + 1)
            d['item'].append(recall_item)
            for key, val in kwargs.items():
                if val[1] is not None:
                    d[key].append(val[1][i][j])
                else:
                    d[key].append(np.nan)
        current_list += 1
        prev_subject = subject
    data = pd.DataFrame(d)
    return data


def _match_values(series, values):
    """Get matches for a data column."""
    if not hasattr(values, '__iter__') or isinstance(values, str):
        values = [values]
    include = series.isin(values)
    return include


def filter_data(
    data,
    subjects=None,
    lists=None,
    trial_type=None,
    positions=None,
    inputs=None,
    outputs=None,
):
    """
    Filter data to get a subset of trials.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw or merged data to filter.

    subjects : hashable or list of hashable
        Subject or subjects to include.

    lists : hashable or list of hashable
        List or lists to include.

    trial_type : {'study', 'recall'}
        Trial type to include.

    positions : int or list of int
        Position or positions to include.

    inputs : int or list of int
        Input position or positions to include.

    outputs : int or list of int
        Output position or positions to include.

    Returns
    -------
    filtered : pandas.DataFrame
        The filtered subset of data.

    Examples
    --------
    >>> from psifr import fr
    >>> subjects_list = [1, 1, 2, 2]
    >>> study_lists = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
    >>> recall_lists = [['b'], ['d', 'c'], ['f', 'e'], []]
    >>> raw = fr.table_from_lists(subjects_list, study_lists, recall_lists)
    >>> fr.filter_data(raw, subjects=1, trial_type='study')
       subject  list trial_type  position item
    0        1     1      study         1    a
    1        1     1      study         2    b
    3        1     2      study         1    c
    4        1     2      study         2    d

    >>> data = fr.merge_free_recall(raw)
    >>> fr.filter_data(data, subjects=2)
       subject  list item  input  output  study  recall  repeat  intrusion  prior_list  prior_input
    4        2     1    e      1     2.0   True    True       0      False         NaN          NaN
    5        2     1    f      2     1.0   True    True       0      False         NaN          NaN
    6        2     2    g      1     NaN   True   False       0      False         NaN          NaN
    7        2     2    h      2     NaN   True   False       0      False         NaN          NaN
    """
    include = data['subject'].notna()
    if subjects is not None:
        include &= _match_values(data['subject'], subjects)

    if lists is not None:
        include &= _match_values(data['list'], lists)

    if trial_type is not None:
        include &= data['trial_type'] == trial_type

    if positions is not None:
        include &= _match_values(data['position'], positions)

    if inputs is not None:
        include &= _match_values(data['input'], inputs)

    if outputs is not None:
        include &= _match_values(data['output'], outputs)

    filtered = data.loc[include].copy()
    return filtered


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

    Examples
    --------
    >>> from psifr import fr
    >>> import pandas as pd
    >>> raw = pd.DataFrame(
    ...     {'subject': [1, 1], 'list': [1, 1], 'position': [1, 2], 'item': ['a', 'b']}
    ... )
    >>> fr.check_data(raw)
    Traceback (most recent call last):
      File "psifr/fr.py", line 253, in check_data
        assert col in df.columns, f'Required column {col} is missing.'
    AssertionError: Required column trial_type is missing.
    """
    # check that all fields are accounted for
    columns = ['subject', 'list', 'trial_type', 'position', 'item']
    for col in columns:
        assert col in df.columns, f'Required column {col} is missing.'

    # only one column has a hard constraint on its exact content
    assert (
        df['trial_type'].isin(['study', 'recall']).all()
    ), 'trial_type for all trials must be "study" or "recall".'


def block_index(list_labels):
    """
    Get index of each block in a list.

    Parameters
    ----------
    list_labels : list or numpy.ndarray
        Position labels that define the blocks.

    Returns
    -------
    block : numpy.ndarray
        Block index of each position.

    Examples
    --------
    >>> from psifr import fr
    >>> list_labels = [2, 2, 3, 3, 3, 1, 1]
    >>> fr.block_index(list_labels)
    array([1, 1, 2, 2, 2, 3, 3])
    """
    prev_label = ''
    curr_block = 0
    block = np.zeros(len(list_labels), dtype=int)
    for i, label in enumerate(list_labels):
        if prev_label != label:
            curr_block += 1
        block[i] = curr_block
        prev_label = label
    return block


def pool_index(trial_items, pool_items_list):
    """
    Get the index of each item in the full pool.

    Parameters
    ----------
    trial_items : pandas.Series
        The item presented on each trial.

    pool_items_list : list or numpy.ndarray
        List of items in the full pool.

    Returns
    -------
    item_index : pandas.Series
        Index of each item in the pool. Trials with items not in the
        pool will be <NA>.

    Examples
    --------
    >>> import pandas as pd
    >>> from psifr import fr
    >>> trial_items = pd.Series(['b', 'a', 'z', 'c', 'd'])
    >>> pool_items_list = ['a', 'b', 'c', 'd', 'e', 'f']
    >>> fr.pool_index(trial_items, pool_items_list)
    0       1
    1       0
    2    <NA>
    3       2
    4       3
    dtype: Int64
    """
    pool_map = dict(zip(pool_items_list, np.arange(len(pool_items_list))))
    item_index = trial_items.map(pool_map).astype('Int64')
    return item_index


def reset_list(df):
    """
    Reset list index in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw or merged data. Must have subject and list fields.

    Returns
    -------
    pandas.DataFrame
        Data with a renumbered list field, starting from 1.

    Examples
    --------
    >>> from psifr import fr
    >>> subjects_list = [1, 1]
    >>> study_lists = [['a', 'b'], ['c', 'd']]
    >>> recall_lists = [['b'], ['c', 'd']]
    >>> list_nos = [3, 4]
    >>> raw = fr.table_from_lists(subjects_list, study_lists, recall_lists, lists=list_nos)
    >>> raw
       subject  list trial_type  position item
    0        1     3      study         1    a
    1        1     3      study         2    b
    2        1     3     recall         1    b
    3        1     4      study         1    c
    4        1     4      study         2    d
    5        1     4     recall         1    c
    6        1     4     recall         2    d

    >>> fr.reset_list(raw)
       subject  list trial_type  position item
    0        1     1      study         1    a
    1        1     1      study         2    b
    2        1     1     recall         1    b
    3        1     2      study         1    c
    4        1     2      study         2    d
    5        1     2     recall         1    c
    6        1     2     recall         2    d
    """
    df = df.copy()
    for subject in df['subject'].unique():
        subject_lists = df.loc[df['subject'] == subject, 'list'].unique()
        for i, listno in enumerate(subject_lists):
            df.loc[(df['subject'] == subject) & (df['list'] == listno), 'list'] = i + 1
    return df


def split_lists(frame, phase, keys=None, names=None, item_query=None, as_list=False):
    """
    Convert free recall data from one phase to split format.

    Parameters
    ----------
    frame : pandas.DataFrame
        Free recall data with separate study and recall events.

    phase : {'study', 'recall', 'raw'}
        Phase of recall to split. If 'raw', all trials will be included.

    keys : list of str, optional
        Data columns to include in the split data. If not specified,
        all columns will be included.

    names : list of str, optional
        Name for each column in the returned split data. Default is to
        use the same names as the input columns.

    item_query : str, optional
        Query string to select study trials to include. See
        `pandas.DataFrame.query` for allowed format.

    as_list : bool, optional
        If true, each column will be output as a list; otherwise,
        outputs will be numpy.ndarray.

    Returns
    -------
    split : dict of str: list
        Data in split format. Each included column will be a key in the
        dictionary, with a list of either numpy.ndarray (default) or
        lists, containing the values for that column.

    See Also
    --------
    table_from_lists : Convert list-format data to a table.

    Examples
    --------
    >>> from psifr import fr
    >>> study = [['absence', 'hollow'], ['fountain', 'piano']]
    >>> recall = [['absence'], ['piano', 'fountain']]
    >>> raw = fr.table_from_lists([1, 1], study, recall)
    >>> data = fr.merge_free_recall(raw)
    >>> data
       subject  list      item  input  output  study  recall  repeat  intrusion  prior_list  prior_input
    0        1     1   absence      1     1.0   True    True       0      False         NaN          NaN
    1        1     1    hollow      2     NaN   True   False       0      False         NaN          NaN
    2        1     2  fountain      1     2.0   True    True       0      False         NaN          NaN
    3        1     2     piano      2     1.0   True    True       0      False         NaN          NaN

    Get study events split by list, just including the list and item fields.

    >>> fr.split_lists(data, 'study', keys=['list', 'item'], as_list=True)
    {'list': [[1, 1], [2, 2]], 'item': [['absence', 'hollow'], ['fountain', 'piano']]}

    Export recall events, split by list.

    >>> fr.split_lists(data, 'recall', keys=['item'], as_list=True)
    {'item': [['absence'], ['piano', 'fountain']]}

    Raw events (i.e., events that haven't been scored) can also be
    exported to list format.

    >>> fr.split_lists(raw, 'raw', keys=['position'])
    {'position': [array([1, 2, 1]), array([1, 2, 1, 2])]}
    """
    split = {}
    if keys is None:
        keys = frame.columns.tolist()
    if not keys:
        return split
    if names is None:
        names = keys

    unique_lists = frame['list'].unique()
    if phase == 'study':
        phase_data = frame.loc[frame['study']]
    elif phase == 'recall':
        phase_data = frame.loc[frame['recall']].sort_values(['list', 'output'])
    elif phase == 'raw':
        phase_data = frame
    else:
        raise ValueError(f'Invalid phase: {phase}')

    if item_query is not None:
        # get the subset of the pool that is of interest
        mask = phase_data.eval(item_query).to_numpy()
    else:
        mask = np.ones(phase_data.shape[0], dtype=bool)

    frame_idx = phase_data.reset_index().groupby('list').indices
    for key, name in zip(keys, names):
        if key is None or key not in frame.columns:
            split[name] = None
            continue
        all_values = phase_data[key].to_numpy()
        split[name] = []
        for n in unique_lists:
            if n in frame_idx:
                list_idx = frame_idx[n]
                x = all_values[list_idx][mask[list_idx]]
            else:
                x = np.array([])

            if as_list:
                split[name].append(x.tolist())
            else:
                split[name].append(x)
    return split


def merge_free_recall(data, **kwargs):
    """
    Score free recall data by matching up study and recall events.

    Parameters
    ----------
    data : pandas.DataFrame
        Free recall data in Psifr format. Must have subject, list,
        trial_type, position, and item columns.

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

        study : bool
            True for rows corresponding to a unique study event.

        recall : bool
            True for rows corresponding to a unique recall event.

        repeat : int
            Number of times this recall event has been repeated (0 for
            the first recall of an item).

        intrusion : bool
            True for recalls that do not correspond to any study event.

        prior_list : int
            For prior-list intrusions, the list the item was presented.

        prior_position : int
            For prior-list intrusions, the position the item was presented.

    See Also
    --------
    merge_lists : Flexibly merge study events with recall events.
        Useful for recall phases that don't match the typical free
        recall setup, like final free recall of all lists.

    Examples
    --------
    >>> import numpy as np
    >>> from psifr import fr
    >>> study = [['absence', 'hollow'], ['fountain', 'piano']]
    >>> recall = [['absence'], ['piano', 'hollow']]
    >>> raw = fr.table_from_lists([1, 1], study, recall)
    >>> raw
       subject  list trial_type  position      item
    0        1     1      study         1   absence
    1        1     1      study         2    hollow
    2        1     1     recall         1   absence
    3        1     2      study         1  fountain
    4        1     2      study         2     piano
    5        1     2     recall         1     piano
    6        1     2     recall         2    hollow

    Score the data to create a table with matched study and recall events.

    >>> data = fr.merge_free_recall(raw)
    >>> data
       subject  list      item  input  output  study  recall  repeat  intrusion  prior_list  prior_input
    0        1     1   absence    1.0     1.0   True    True       0      False         NaN          NaN
    1        1     1    hollow    2.0     NaN   True   False       0      False         NaN          NaN
    2        1     2  fountain    1.0     NaN   True   False       0      False         NaN          NaN
    3        1     2     piano    2.0     1.0   True    True       0      False         NaN          NaN
    4        1     2    hollow    NaN     2.0  False    True       0       True         1.0          2.0

    You can also include non-standard columns. Information that only
    applies to study events (here, the encoding task used) can be
    indicated using the :code:`study_keys` input.

    >>> raw['task'] = np.array([1, 2, np.nan, 2, 1, np.nan, np.nan])
    >>> fr.merge_free_recall(raw, study_keys=['task'])
       subject  list      item  input  output  study  recall  repeat  intrusion  task  prior_list  prior_input
    0        1     1   absence    1.0     1.0   True    True       0      False   1.0         NaN          NaN
    1        1     1    hollow    2.0     NaN   True   False       0      False   2.0         NaN          NaN
    2        1     2  fountain    1.0     NaN   True   False       0      False   2.0         NaN          NaN
    3        1     2     piano    2.0     1.0   True    True       0      False   1.0         NaN          NaN
    4        1     2    hollow    NaN     2.0  False    True       0       True   NaN         1.0          2.0

    Information that only applies to recall onsets (here, the time in
    seconds after the start of the recall phase that a recall attempt
    was made), can be indicated using the :code:`recall_keys` input.

    >>> raw['onset'] = np.array([np.nan, np.nan, 1.1, np.nan, np.nan, 1.4, 3.8])
    >>> fr.merge_free_recall(raw, recall_keys=['onset'])
       subject  list      item  input  output  study  recall  repeat  intrusion  onset  prior_list  prior_input
    0        1     1   absence    1.0     1.0   True    True       0      False    1.1         NaN          NaN
    1        1     1    hollow    2.0     NaN   True   False       0      False    NaN         NaN          NaN
    2        1     2  fountain    1.0     NaN   True   False       0      False    NaN         NaN          NaN
    3        1     2     piano    2.0     1.0   True    True       0      False    1.4         NaN          NaN
    4        1     2    hollow    NaN     2.0  False    True       0       True    3.8         1.0          2.0

    Use :code:`list_keys` to indicate columns that apply to both study
    and recall events. If :code:`list_keys` do not match for a pair of
    study and recall events, they will not be matched in the output.

    >>> raw['condition'] = np.array([1, 1, 1, 2, 2, 2, 2])
    >>> fr.merge_free_recall(raw, list_keys=['condition'])
       subject  list      item  input  output  study  recall  repeat  intrusion  condition  prior_list  prior_input
    0        1     1   absence    1.0     1.0   True    True       0      False          1         NaN          NaN
    1        1     1    hollow    2.0     NaN   True   False       0      False          1         NaN          NaN
    2        1     2  fountain    1.0     NaN   True   False       0      False          2         NaN          NaN
    3        1     2     piano    2.0     1.0   True    True       0      False          2         NaN          NaN
    4        1     2    hollow    NaN     2.0  False    True       0       True          2         1.0          2.0
    """
    study = data.loc[data['trial_type'] == 'study'].copy()
    recall = data.loc[data['trial_type'] == 'recall'].copy()
    merged = merge_lists(study, recall, **kwargs)

    # to identify prior-list intrusions, merge study events and intrusions
    intrusions = merged.query('intrusion')
    plis = pd.merge(intrusions, study, on=['subject', 'item'], how='inner')

    # add prior list and prior input information
    plis['list'] = plis['list_x']
    plis['prior_list'] = plis['list_y']
    plis['prior_input'] = plis['position']
    include = ['subject', 'list', 'item', 'output', 'prior_list', 'prior_input']
    merged = pd.merge(
        merged, plis[include], on=['subject', 'list', 'item', 'output'], how='outer'
    )

    # reset concidental "future list intrusions"
    isfli = merged['list'] < merged['prior_list']
    merged.loc[isfli, 'prior_list'] = np.nan
    merged.loc[isfli, 'prior_input'] = np.nan
    return merged


def merge_lists(
    study,
    recall,
    merge_keys=None,
    list_keys=None,
    study_keys=None,
    recall_keys=None,
    position_key='position',
):
    """
    Merge study and recall events together for each list.

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

        study : bool
            True for rows corresponding to a unique study event.

        recall : bool
            True for rows corresponding to a unique recall event.

        repeat : int
            Number of times this recall event has been repeated (0 for
            the first recall of an item).

        intrusion : bool
            True for recalls that do not correspond to any study event.

    See Also
    --------
    merge_free_recall : Score standard free recall data.

    Examples
    --------
    >>> import pandas as pd
    >>> from psifr import fr
    >>> study = pd.DataFrame(
    ...    {'subject': [1, 1], 'list': [1, 1], 'position': [1, 2], 'item': ['a', 'b']}
    ... )
    >>> recall = pd.DataFrame(
    ...    {'subject': [1], 'list': [1], 'position': [1], 'item': ['b']}
    ... )
    >>> fr.merge_lists(study, recall)
       subject  list item  input  output  study  recall  repeat  intrusion
    0        1     1    a      1     NaN   True   False       0      False
    1        1     1    b      2     1.0   True    True       0      False
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
    recall = recall.copy()
    recall.loc[:, 'repeat'] = recall.groupby(merge_keys).cumcount()

    # get just the fields to use in the merge
    study = study.copy()
    study = study[merge_keys + ['position'] + list_keys + study_keys]
    recall = recall[merge_keys + ['repeat', 'position'] + list_keys + recall_keys]

    # merge information from study and recall trials
    merged = pd.merge(
        study,
        recall,
        left_on=merge_keys + list_keys,
        right_on=merge_keys + list_keys,
        how='outer',
    )

    # position from study events indicates input position;
    # position from recall events indicates output position
    merged = merged.rename(
        columns={position_key + '_x': 'input', position_key + '_y': 'output'}
    )

    # fix repeats field to define for non-recalled items
    merged.loc[merged['repeat'].isna(), 'repeat'] = 0
    merged = merged.astype({'repeat': 'int'})

    # field to indicate unique study events
    merged.loc[:, 'study'] = merged['input'].notna() & (merged['repeat'] == 0)

    # TODO: deal with repeats in the study list
    # field to indicate unique recall events
    merged.loc[:, 'recall'] = merged['output'].notna()

    # field to indicate whether a given recall was an intrusion
    merged.loc[:, 'intrusion'] = merged['input'].isna()

    # reorder columns
    core_keys = ['input', 'output', 'study', 'recall', 'repeat', 'intrusion']
    columns = merge_keys + core_keys + list_keys + study_keys + recall_keys
    merged = merged.reindex(columns=columns)

    # sort rows in standard order
    sort_keys = merge_keys.copy() + ['input']
    sort_keys.remove('item')
    merged = merged.sort_values(by=sort_keys, ignore_index=True)

    return merged


def spc(df):
    """
    Serial position curve.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists.

    Returns
    -------
    recall : pandas.Series
        Index includes:

        subject : hashable
            Subject identifier.

        input : int
            Serial position in the list.

        Values are:

        recall : float
            Recall probability for each serial position.

    See Also
    --------
    plot_spc : Plot serial position curve results.
    pnr : Probability of nth recall.

    Examples
    --------
    >>> from psifr import fr
    >>> raw = fr.sample_data('Morton2013')
    >>> data = fr.merge_free_recall(raw)
    >>> fr.spc(data)
                     recall
    subject input          
    1       1.0    0.541667
            2.0    0.458333
            3.0    0.625000
            4.0    0.333333
            5.0    0.437500
    ...                 ...
    47      20.0   0.500000
            21.0   0.770833
            22.0   0.729167
            23.0   0.895833
            24.0   0.958333
    <BLANKLINE>
    [960 rows x 1 columns]
    """
    clean = df.query('study')
    recall = clean.groupby(['subject', 'input'])['recall'].mean()
    return pd.DataFrame(recall)


def pnr(df, item_query=None, test_key=None, test=None):
    """
    Probability of recall by serial position and output position.

    Calculate probability of Nth recall, where N is each output
    position. Invalid recalls (repeats and intrusions) are ignored and
    not counted toward output position.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists. List length is
        assumed to be the same for all lists within each subject.
        Must have fields: subject, list, input, output, study, recall.
        Input position must be defined such that the first serial
        position is 1, not 0.

    item_query : str, optional
        Query string to select items to include in the pool of possible
        recalls to be examined. See `pandas.DataFrame.query` for
        allowed format.

    test_key : str, optional
        Name of column with labels to use when testing transitions for
        inclusion.

    test : callable, optional
        Callable that takes in previous and current item values and
        returns True for transitions that should be included.

    Returns
    -------
    prob : pandas.DataFrame
        Analysis results. Has fields: subject, output, input, prob,
        actual, possible. The prob column for output x and input y
        indicates the probability of recalling input position y at
        output position x. The actual and possible columns give the
        raw tallies for how many times an event actually occurred and
        how many times it was possible given the recall sequence.

    See Also
    --------
    plot_spc : Plot recall probability as a function of serial
        position.
    spc : Overall recall probability by serial position.

    Examples
    --------
    >>> from psifr import fr
    >>> raw = fr.sample_data('Morton2013')
    >>> data = fr.merge_free_recall(raw)
    >>> fr.pnr(data)
                              prob  actual  possible
    subject output input                            
    1       1      1      0.000000       0        48
                   2      0.020833       1        48
                   3      0.000000       0        48
                   4      0.000000       0        48
                   5      0.000000       0        48
    ...                        ...     ...       ...
    47      24     20          NaN       0         0
                   21          NaN       0         0
                   22          NaN       0         0
                   23          NaN       0         0
                   24          NaN       0         0
    <BLANKLINE>
    [23040 rows x 3 columns]
    """
    list_length = int(df['input'].max())
    measure = measures.TransitionOutputs(
        list_length, item_query=item_query, test_key=test_key, test=test
    )
    prob = measure.analyze(df)
    return prob


def _subject_pli_list_lag(df, max_lag):
    """List lag of prior-list intrusions for one subject."""
    if max_lag >= df['list'].nunique():
        warnings.warn('All lists are excluded based on max_lag.')
    results = pd.DataFrame(
        index=pd.Index(np.arange(1, max_lag + 1), name='list_lag'),
        columns=['count', 'per_list', 'prob'],
        dtype=float,
    )
    included = df[df['list'] > max_lag]
    intrusions = included.query('intrusion')
    if len(intrusions) > 0:
        list_lag = intrusions['list'] - intrusions['prior_list']
        results['count'] = list_lag.value_counts()
    results['count'] = results['count'].fillna(0).astype(int)
    results['per_list'] = results['count'] / included['list'].nunique()
    results['prob'] = results['count'] / intrusions.shape[0]
    return results


def pli_list_lag(df, max_lag):
    """
    List lag of prior-list intrusions.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_free_recall. Must have
        fields: subject, list, intrusion, prior_list. Lists must be
        numbered starting from 1 and all lists must be included.

    max_lag : int
        Maximum list lag to consider. The intial :code:`max_lag` lists
        for each subject will be excluded so that all considered lags
        are possible for all included lists.

    Returns
    -------
    results : pandas.DataFrame
        For each subject and list lag, the proportion of intrusions at
        that lag, in the :code:`results['prob']` column.

    Examples
    --------
    >>> from psifr import fr
    >>> raw = fr.sample_data('Morton2013')
    >>> data = fr.merge_free_recall(raw)
    >>> fr.pli_list_lag(data, 3)
                      count  per_list      prob
    subject list_lag                           
    1       1             7  0.155556  0.259259
            2             5  0.111111  0.185185
            3             0  0.000000  0.000000
    2       1             9  0.200000  0.191489
            2             2  0.044444  0.042553
    ...                 ...       ...       ...
    46      2             1  0.022222  0.100000
            3             0  0.000000  0.000000
    47      1             5  0.111111  0.277778
            2             1  0.022222  0.055556
            3             0  0.000000  0.000000
    <BLANKLINE>
    [120 rows x 3 columns]
    """
    result = df.groupby('subject').apply(_subject_pli_list_lag, max_lag)
    return result


def lag_crp(
    df, lag_key='input', count_unique=False, item_query=None, test_key=None, test=None
):
    """
    Lag-CRP for multiple subjects.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists. List length is
        assumed to be the same for all lists. Must have fields:
        subject, list, input, output, recalled. Input position must be
        defined such that the first serial position is 1, not 0.

    lag_key : str, optional
        Name of column to use when calculating lag between recalled
        items. Default is to calculate lag based on input position.

    count_unique : bool, optional
        If true, possible transitions of the same lag will only be
        incremented once per transition.

    item_query : str, optional
        Query string to select items to include in the pool of possible
        recalls to be examined. See `pandas.DataFrame.query` for
        allowed format.

    test_key : str, optional
        Name of column with labels to use when testing transitions for
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

    See Also
    --------
    lag_rank : Rank of the absolute lags in recall sequences.

    Examples
    --------
    >>> from psifr import fr
    >>> raw = fr.sample_data('Morton2013')
    >>> data = fr.merge_free_recall(raw)
    >>> fr.lag_crp(data)
                       prob  actual  possible
    subject lag                              
    1       -23.0  0.020833       1        48
            -22.0  0.035714       3        84
            -21.0  0.026316       3       114
            -20.0  0.024000       3       125
            -19.0  0.014388       2       139
    ...                 ...     ...       ...
    47       19.0  0.061224       3        49
             20.0  0.055556       2        36
             21.0  0.045455       1        22
             22.0  0.071429       1        14
             23.0  0.000000       0         6
    <BLANKLINE>
    [1880 rows x 3 columns]
    """
    list_length = df[lag_key].max()
    measure = measures.TransitionLag(
        list_length,
        lag_key=lag_key,
        count_unique=count_unique,
        item_query=item_query,
        test_key=test_key,
        test=test,
    )
    crp = measure.analyze(df)
    return crp


def lag_crp_compound(
    df, lag_key='input', count_unique=False, item_query=None, test_key=None, test=None
):
    """
    Conditional response probability by lag of current and prior transitions.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists. List length is
        assumed to be the same for all lists. Must have fields:
        subject, list, input, output, recalled. Input position must be
        defined such that the first serial position is 1, not 0.

    lag_key : str, optional
        Name of column to use when calculating lag between recalled
        items. Default is to calculate lag based on input position.

    count_unique : bool, optional
        If true, possible transitions of the same lag will only be
        incremented once per transition.

    item_query : str, optional
        Query string to select items to include in the pool of possible
        recalls to be examined. See `pandas.DataFrame.query` for
        allowed format.

    test_key : str, optional
        Name of column with labels to use when testing transitions for
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

        previous : int
            Lag of the previous transition.

        current : int
            Lag of the current transition.

        prob : float
            Probability of each lag transition.

        actual : int
            Total of actual made transitions at each lag.

        possible : int
            Total of times each lag was possible, given the prior
            input position and the remaining items to be recalled.

    See Also
    --------
    lag_crp : Conditional response probability by lag.

    Examples
    --------
    >>> from psifr import fr
    >>> subjects = [1]
    >>> study = [['absence', 'hollow', 'pupil', 'fountain']]
    >>> recall = [['fountain', 'hollow', 'absence']]
    >>> raw = fr.table_from_lists(subjects, study, recall)
    >>> data = fr.merge_free_recall(raw)
    >>> crp = fr.lag_crp_compound(data)
    >>> crp.head(14)
                              prob  actual  possible
    subject previous current                        
    1       -3       -3        NaN       0         0
                     -2        NaN       0         0
                     -1        NaN       0         0
                      0        NaN       0         0
                      1        NaN       0         0
                      2        NaN       0         0
                      3        NaN       0         0
            -2       -3        NaN       0         0
                     -2        NaN       0         0
                     -1        1.0       1         1
                      0        NaN       0         0
                      1        0.0       0         1
                      2        NaN       0         0
                      3        NaN       0         0
    """
    list_length = df[lag_key].max()
    measure = measures.TransitionLag(
        list_length,
        lag_key=lag_key,
        count_unique=count_unique,
        item_query=item_query,
        test_key=test_key,
        test=test,
        compound=True
    )
    crp = measure.analyze(df)
    return crp


def lag_rank(df, item_query=None, test_key=None, test=None):
    """
    Calculate rank of the absolute lags in free recall lists.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists. List length is
        assumed to be the same for all lists within each subject.
        Must have fields: subject, list, input, output, recalled.
        Input position must be defined such that the first serial
        position is 1, not 0.

    item_query : str, optional
        Query string to select items to include in the pool of possible
        recalls to be examined. See `pandas.DataFrame.query` for
        allowed format.

    test_key : str, optional
        Name of column with labels to use when testing transitions for
        inclusion.

    test : callable, optional
        Callable that takes in previous and current item values and
        returns True for transitions that should be included.

    Returns
    -------
    stat : pandas.DataFrame
        Has fields 'subject' and 'rank'.

    See Also
    --------
    lag_crp : Conditional response probability by input lag.

    Examples
    --------
    >>> from psifr import fr
    >>> raw = fr.sample_data('Morton2013')
    >>> data = fr.merge_free_recall(raw)
    >>> lag_rank = fr.lag_rank(data)
    >>> lag_rank.head()
                 rank
    subject          
    1        0.610953
    2        0.635676
    3        0.612607
    4        0.667090
    5        0.643923
    """
    measure = measures.TransitionLagRank(
        item_query=item_query, test_key=test_key, test=test
    )
    rank = measure.analyze(df)
    return rank


def distance_crp(
    df,
    index_key,
    distances,
    edges,
    centers=None,
    count_unique=False,
    item_query=None,
    test_key=None,
    test=None,
):
    """
    Conditional response probability by distance bin.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged free recall data.

    index_key : str
        Name of column containing the index of each item in the
        `distances` matrix.

    distances : numpy.array
        Items x items matrix of pairwise distances or similarities.

    edges : array-like
        Edges of bins to apply to the distances.

    centers : array-like, optional
        Centers to label each bin with. If not specified, the center
        point between edges will be used.

    count_unique : bool, optional
        If true, possible transitions to a given distance bin will only
        count once for a given transition.

    item_query : str, optional
        Query string to select items to include in the pool of possible
        recalls to be examined. See `pandas.DataFrame.query` for
        allowed format.

    test_key : str, optional
        Name of column with labels to use when testing transitions for
        inclusion.

    test : callable, optional
        Callable that takes in previous and current item values and
        returns True for transitions that should be included.

    Returns
    -------
    crp : pandas.DataFrame
        Has fields:

        subject : hashable
            Results are separated by each subject.

        bin : int
            Distance bin.

        prob : float
            Probability of each distance bin.

        actual : int
            Total of actual transitions for each distance bin.

        possible : int
            Total of times each distance bin was possible, given the
            prior input position and the remaining items to be
            recalled.

    See Also
    --------
    pool_index : Given a list of presented items and an item pool, look
        up the pool index of each item.
    distance_rank : Calculate rank of transition distances.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.distance import squareform
    >>> from psifr import fr
    >>> raw = fr.sample_data('Morton2013')
    >>> data = fr.merge_free_recall(raw)
    >>> items, distances = fr.sample_distances('Morton2013')
    >>> data['item_index'] = fr.pool_index(data['item'], items)
    >>> edges = np.percentile(squareform(distances), np.linspace(1, 99, 10))
    >>> fr.distance_crp(data, 'item_index', distances, edges)
                                 bin      prob  actual  possible
    subject center                                              
    1       0.467532  (0.352, 0.583]  0.085456     151      1767
            0.617748  (0.583, 0.653]  0.067916      87      1281
            0.673656  (0.653, 0.695]  0.062500      65      1040
            0.711075  (0.695, 0.727]  0.051836      48       926
            0.742069  (0.727, 0.757]  0.050633      44       869
    ...                          ...       ...     ...       ...
    47      0.742069  (0.727, 0.757]  0.062822      61       971
            0.770867  (0.757, 0.785]  0.030682      27       880
            0.800404  (0.785, 0.816]  0.040749      37       908
            0.834473  (0.816, 0.853]  0.046651      39       836
            0.897275  (0.853, 0.941]  0.028868      25       866
    <BLANKLINE>
    [360 rows x 4 columns]
    """
    measure = measures.TransitionDistance(
        index_key,
        distances,
        edges,
        centers=centers,
        count_unique=count_unique,
        item_query=item_query,
        test_key=test_key,
        test=test,
    )
    crp = measure.analyze(df)
    return crp


def distance_rank(df, index_key, distances, item_query=None, test_key=None, test=None):
    """
    Calculate rank of transition distances in free recall lists.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists. List length is
        assumed to be the same for all lists within each subject.
        Must have fields: subject, list, input, output, recalled.
        Input position must be defined such that the first serial
        position is 1, not 0.

    index_key : str
        Name of column containing the index of each item in the
        `distances` matrix.

    distances : numpy.array
        Items x items matrix of pairwise distances or similarities.

    item_query : str, optional
        Query string to select items to include in the pool of possible
        recalls to be examined. See `pandas.DataFrame.query` for
        allowed format.

    test_key : str, optional
        Name of column with labels to use when testing transitions for
        inclusion.

    test : callable, optional
        Callable that takes in previous and current item values and
        returns True for transitions that should be included.

    Returns
    -------
    stat : pandas.DataFrame
        Has fields 'subject' and 'rank'.

    See Also
    --------
    pool_index : Given a list of presented items and an item pool, look
        up the pool index of each item.
    distance_crp : Conditional response probability by distance bin.

    Examples
    --------
    >>> from scipy.spatial.distance import squareform
    >>> from psifr import fr
    >>> raw = fr.sample_data('Morton2013')
    >>> data = fr.merge_free_recall(raw)
    >>> items, distances = fr.sample_distances('Morton2013')
    >>> data['item_index'] = fr.pool_index(data['item'], items)
    >>> dist_rank = fr.distance_rank(data, 'item_index', distances)
    >>> dist_rank.head()
                 rank
    subject          
    1        0.635571
    2        0.571457
    3        0.627282
    4        0.637596
    5        0.646181
    """
    measure = measures.TransitionDistanceRank(
        index_key, distances, item_query=item_query, test_key=test_key, test=test
    )
    rank = measure.analyze(df)
    return rank


def distance_rank_shifted(
    df, index_key, distances, max_shift, item_query=None, test_key=None, test=None
):
    """
    Rank of transition distances relative to earlier items.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists. List length is
        assumed to be the same for all lists within each subject.
        Must have fields: subject, list, input, output, recalled.
        Input position must be defined such that the first serial
        position is 1, not 0.

    index_key : str
        Name of column containing the index of each item in the
        `distances` matrix.

    distances : numpy.array
        Items x items matrix of pairwise distances or similarities.

    max_shift : int
        Maximum number of items back for which to rank distances.

    item_query : str, optional
        Query string to select items to include in the pool of possible
        recalls to be examined. See `pandas.DataFrame.query` for
        allowed format.

    test_key : str, optional
        Name of column with labels to use when testing transitions for
        inclusion.

    test : callable, optional
        Callable that takes in previous and current item values and
        returns True for transitions that should be included.

    Returns
    -------
    stat : pandas.DataFrame
        Has fields 'subject' and 'rank'.

    See Also
    --------
    pool_index : Given a list of presented items and an item pool, look
        up the pool index of each item.
    distance_rank : Rank of transition distances relative to the
        just-previous item.

    Examples
    --------
    >>> from scipy.spatial.distance import squareform
    >>> from psifr import fr
    >>> raw = fr.sample_data('Morton2013')
    >>> data = fr.merge_free_recall(raw)
    >>> items, distances = fr.sample_distances('Morton2013')
    >>> data['item_index'] = fr.pool_index(data['item'], items)
    >>> dist_rank = fr.distance_rank_shifted(data, 'item_index', distances, 3)
    >>> dist_rank
                       rank
    subject shift          
    1       -3     0.523426
            -2     0.559199
            -1     0.634392
    2       -3     0.475931
            -2     0.507574
    ...                 ...
    46      -2     0.515332
            -1     0.603304
    47      -3     0.542951
            -2     0.565001
            -1     0.635415
    <BLANKLINE>
    [120 rows x 1 columns]
    """
    measure = measures.TransitionDistanceRankShifted(
        index_key, distances, max_shift, item_query=item_query, test_key=test_key, test=test
    )
    rank = measure.analyze(df)
    return rank


def distance_rank_window(
    df, index_key, distances, window_lags, item_query=None, test_key=None, test=None
):
    """
    Rank of transition distances relative to items in a window.

    Transitions are ranked based on their distance relative to items
    at specified lags from the previous item in the input list.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists. List length is
        assumed to be the same for all lists within each subject.
        Must have fields: subject, list, input, output, recalled.
        Input position must be defined such that the first serial
        position is 1, not 0.

    index_key : str
        Name of column containing the index of each item in the
        `distances` matrix.

    distances : numpy.array
        Items x items matrix of pairwise distances or similarities.

    window_lags : array_like
        Serial position lags to include in the window.

    item_query : str, optional
        Query string to select items to include in the pool of possible
        recalls to be examined. See `pandas.DataFrame.query` for
        allowed format.

    test_key : str, optional
        Name of column with labels to use when testing transitions for
        inclusion.

    test : callable, optional
        Callable that takes in previous and current item values and
        returns True for transitions that should be included.

    Returns
    -------
    stat : pandas.DataFrame
        Has fields 'subject', 'lag', and 'rank'.
    """
    list_length = int(df['input'].max())
    measure = measures.TransitionDistanceRankWindow(
        index_key,
        distances,
        list_length,
        window_lags,
        item_query=item_query,
        test_key=test_key,
        test=test,
    )
    rank = measure.analyze(df)
    return rank


def category_crp(df, category_key, item_query=None, test_key=None, test=None):
    """
    Conditional response probability of within-category transitions.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_lists. List length is
        assumed to be the same for all lists within each subject.
        Must have fields: subject, list, input, output, recalled.

    category_key : str
        Name of column with category labels.

    item_query : str, optional
        Query string to select items to include in the pool of possible
        recalls to be examined. See `pandas.DataFrame.query` for
        allowed format.

    test_key : str, optional
        Name of column with labels to use when testing transitions for
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

        prob : float
            Probability of each lag transition.

        actual : int
            Total of actual made transitions at each lag.

        possible : int
            Total of times each lag was possible, given the prior
            input position and the remaining items to be recalled.

    Examples
    --------
    >>> from psifr import fr
    >>> raw = fr.sample_data('Morton2013')
    >>> data = fr.merge_free_recall(raw, study_keys=['category'])
    >>> cat_crp = fr.category_crp(data, 'category')
    >>> cat_crp.head()
                 prob  actual  possible
    subject                            
    1        0.801147     419       523
    2        0.733456     399       544
    3        0.763158     377       494
    4        0.814882     449       551
    5        0.877273     579       660
    """
    measure = measures.TransitionCategory(
        category_key, item_query=item_query, test_key=test_key, test=test
    )
    crp = measure.analyze(df)
    return crp


def _subject_category_clustering(df, category_key):
    """Subject category clustering."""
    study = split_lists(df, 'study', keys=[category_key])
    recall = split_lists(df, 'recall', keys=[category_key])
    lbc = clustering.lbc(study[category_key], recall[category_key])
    arc = clustering.arc(recall[category_key])
    stats = pd.Series({'lbc': np.nanmean(lbc), 'arc': np.nanmean(arc)})
    return stats


def category_clustering(df, category_key):
    """
    Category clustering of recall sequences.

    Calculates ARC (adjusted ratio of clustering) and LBC (list-based
    clustering) statistics indexing recall clustering by category.

    The papers introducing these measures do not describe how to handle
    repeats and intrusions. Here, to maintain the assumptions of the
    measures, they are removed from the recall sequences.

    Note that ARC is undefined when only one category is recalled.
    Lists with undefined statistics will be excluded from calculation
    of mean subject-level statistics. To calculate for each list
    separately, group by list before calling the function. For example:
    :code:`df.groupby('list').apply(fr.category_clustering, 'category')`.

    Parameters
    ----------
    df : pandas.DataFrame
        Merged study and recall data. See merge_free_recall. Must have
        a field indicating the category of each study and recall event.

    category_key : str
        Column with category labels. Labels may be any hashable (e.g.,
        a str or int).

    Returns
    -------
    stats : pandas.DataFrame
        For each subject, includes columns with the mean ARC and LBC
        statistics.

    Examples
    --------
    >>> from psifr import fr
    >>> raw = fr.sample_data('Morton2013')
    >>> mixed = raw.query('list_type == "mixed"')
    >>> data = fr.merge_free_recall(mixed, list_keys=['category'])
    >>> stats = fr.category_clustering(data, 'category')
    >>> stats.head()
                  lbc       arc
    subject                    
    1        3.657971  0.614545
    2        2.953623  0.407839
    3        3.363768  0.627371
    4        4.444928  0.688761
    5        7.530435  0.873755
    """
    # these analyses are undefined when there are repeats and
    # intrusions, so strip them out
    clean = df.query('~intrusion and repeat == 0')
    stats = clean.groupby('subject').apply(_subject_category_clustering, category_key)
    return stats


def plot_spc(recall, **facet_kws):
    """
    Plot a serial position curve.

    Additional arguments are passed to seaborn.relplot.

    Parameters
    ----------
    recall : pandas.DataFrame
        Results from calling `spc`.
    """
    y = 'recall' if 'recall' in recall else 'prob'
    g = sns.FacetGrid(dropna=False, **facet_kws, data=recall.reset_index())
    g.map_dataframe(sns.lineplot, x='input', y=y)
    g.set_xlabels('Serial position')
    g.set_ylabels('Recall probability')
    g.set(ylim=(0, 1))
    return g


def plot_lag_crp(recall, max_lag=5, lag_key='lag', split=True, **facet_kws):
    """
    Plot conditional response probability by lag.

    Additional arguments are passed to seaborn.FacetGrid.

    Parameters
    ----------
    recall : pandas.DataFrame
        Results from calling `lag_crp`.

    max_lag : int, optional
        Maximum absolute lag to plot.

    lag_key : str, optional
        Name of the column indicating lag.

    split : bool, optional
        If true, will plot as two separate lines with a gap at lag 0.
    """
    if split:
        filt_neg = f'{-max_lag} <= {lag_key} < 0'
        filt_pos = f'0 < {lag_key} <= {max_lag}'
        g = sns.FacetGrid(dropna=True, **facet_kws, data=recall.reset_index())
        g.map_dataframe(
            lambda data, **kws: sns.lineplot(
                data=data.query(filt_neg), x=lag_key, y='prob', **kws
            )
        )
        g.map_dataframe(
            lambda data, **kws: sns.lineplot(
                data=data.query(filt_pos), x=lag_key, y='prob', **kws
            )
        )
    else:
        data = recall.query(f'{-max_lag} <= {lag_key} <= {max_lag}')
        g = sns.FacetGrid(dropna=False, **facet_kws, data=data.reset_index())
        g.map_dataframe(sns.lineplot, x=lag_key, y='prob')

    g.set_xlabels('Lag')
    g.set_ylabels('CRP')
    g.set(ylim=(0, 1))
    return g


def plot_distance_crp(crp, min_samples=None, **facet_kws):
    """
    Plot response probability by distance bin.

    Parameters
    ----------
    crp : pandas.DataFrame
        Results from `fr.distance_crp`.

    min_samples : int
        Minimum number of samples a bin must have per subject to
        include in the plot.

    **facet_kws
        Additional inputs to pass to `seaborn.relplot`.
    """
    crp = crp.reset_index()
    if min_samples is not None:
        min_n = crp.groupby('center')['possible'].min()
        include = min_n.loc[min_n >= min_samples].index.to_numpy()
        crp = crp.loc[crp['center'].isin(include)]
    g = sns.FacetGrid(dropna=False, **facet_kws, data=crp.reset_index())
    g.map_dataframe(sns.lineplot, x='center', y='prob')
    g.set_xlabels('Distance')
    g.set_ylabels('CRP')
    g.set(ylim=(0, 1))
    return g


def plot_swarm_error(
    data, x=None, y=None, swarm_color=None, swarm_size=5, point_color='k', **facet_kws
):
    """
    Plot points as a swarm plus mean with error bars.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with statistics to plot.

    x : str
        Name of variable to plot on x-axis.

    y : str
        Name of variable to plot on y-axis.

    swarm_color
        Color for swarm plot points. May use any specification
        supported by seaborn.

    swarm_size : float
        Size of swarm plot points.

    point_color
        Color for the point plot (error bars).

    facet_kws
        Additional keywords for the FacetGrid.
    """
    g = sns.FacetGrid(data=data.reset_index(), dropna=False, **facet_kws)
    g.map_dataframe(
        sns.swarmplot, x=x, y=y, color=swarm_color, size=swarm_size, zorder=1
    )
    g.map_dataframe(
        sns.pointplot, x=x, y=y, color=point_color, join=False, capsize=0.5
    )
    return g


def plot_raster(
    df,
    hue='input',
    palette=None,
    marker='s',
    intrusion_color=None,
    orientation='horizontal',
    length=6,
    aspect=None,
    legend='auto',
    **facet_kws,
):
    """
    Plot recalls in a raster plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Scored free recall data.

    hue : str or None, optional
        Column to use to set marker color.

    palette : optional
        Palette specification supported by Seaborn.

    marker : str, optional
         Marker code supported by Seaborn.

    intrusion_color : optional
        Color of intrusions.

    orientation : {'horizontal', 'vertical'}, optional
        Whether lists should be stacked horizontally or vertically in
        the plot.

    length : float, optional
        Size of the plot dimension along which list varies.

    aspect : float, optional
        Aspect ratio of plot for lists over items.

    legend : str, optional
        Legend setting. See seaborn.scatterplot for details.

    facet_kws : optional
        Additional key words to pass to seaborn.FacetGrid.
    """
    n_item = int(df['input'].max())
    n_list = int(df['list'].max())
    if palette is None and hue == 'input':
        palette = 'viridis'

    if intrusion_color is None:
        intrusion_color = (0.8, 0.1, 0.3)

    list_lim = (0, n_list + 1)
    item_lim = (0, n_item + 1)
    if orientation == 'horizontal':
        x_var, y_var = 'list', 'output'
        x_lim, y_lim = list_lim, item_lim
        x_label, y_label = 'List', 'Output position'
        def_aspect = n_list / n_item
    else:
        x_var, y_var = 'output', 'list'
        x_lim, y_lim = item_lim, list_lim[::-1]
        x_label, y_label = 'Output position', 'List'
        def_aspect = n_item / n_list

    if aspect is None:
        aspect = def_aspect

    if orientation == 'horizontal':
        height = length / aspect
    else:
        height = length

    g = sns.FacetGrid(
        data=df.reset_index(), dropna=False, aspect=aspect, height=height, **facet_kws
    )
    g.map_dataframe(
        sns.scatterplot,
        x=x_var,
        y=y_var,
        marker=marker,
        hue=hue,
        palette=palette,
        legend=legend,
    )
    g.map_dataframe(
        lambda data, color=None, label=None: sns.scatterplot(
            data=data.query('intrusion'),
            x=x_var,
            y=y_var,
            color=intrusion_color,
            marker=marker,
        )
    )
    g.set_xlabels(x_label)
    g.set_ylabels(y_label)
    g.set(xlim=x_lim, ylim=y_lim)
    return g
