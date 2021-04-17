"""Utilities for working with free recall data."""

from pkg_resources import resource_filename
import numpy as np
import pandas as pd
import seaborn as sns

from psifr import measures


def sample_data(study):
    """Read sample data."""
    data_file = resource_filename('psifr', f'data/{study}.csv')
    df = pd.read_csv(data_file)
    return df


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
    """Filter data to get a subset of trials."""
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


def reset_list(df):
    """Reset list index in a DataFrame."""
    df = df.copy()
    for subject in df['subject'].unique():
        subject_lists = df.loc[df['subject'] == subject, 'list'].unique()
        for i, listno in enumerate(subject_lists):
            df.loc[(df['subject'] == subject) & (df['list'] == listno), 'list'] = i + 1
    return df


def split_lists(frame, phase, keys, names=None, item_query=None, as_list=False):
    """
    Convert free recall data from one phase to split format.

    Parameters
    ----------
    frame : pandas.DataFrame
        Free recall data with separate study and recall events.

    phase : {'study', 'recall', 'raw'}
        Phase of recall to split. If 'raw', all trials will be included.

    keys : list of str
        Data columns to include in the split data.

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
    """
    split = {}
    if keys is None:
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
    Merge standard free recall events.

    Split study and recall events and then merge them.
    See `merge_lists` for details.
    """
    study = data.loc[data['trial_type'] == 'study'].copy()
    recall = data.loc[data['trial_type'] == 'recall'].copy()
    merged = merge_lists(study, recall, **kwargs)
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
    """
    list_length = int(df['input'].max())
    measure = measures.TransitionOutputs(
        list_length, item_query=item_query, test_key=test_key, test=test
    )
    prob = measure.analyze(df)
    return prob


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
    """
    list_length = df[lag_key].max()
    measure = measures.TransitionLag(
        list_length,
        lag_key=lag_key,
        count_unique=count_unique,
        item_query=item_query,
        test_key=test_key,
        test=test
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
    """
    measure = measures.TransitionDistanceRank(
        index_key, distances, item_query=item_query, test_key=test_key, test=test
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
    """
    measure = measures.TransitionCategory(
        category_key, item_query=item_query, test_key=test_key, test=test
    )
    crp = measure.analyze(df)
    return crp


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


def plot_lag_crp(recall, max_lag=5, **facet_kws):
    """
    Plot conditional response probability by lag.

    Additional arguments are passed to seaborn.FacetGrid.

    Parameters
    ----------
    recall : pandas.DataFrame
        Results from calling `lag_crp`.

    max_lag : int
        Maximum absolute lag to plot.
    """
    filt_neg = f'{-max_lag} <= lag < 0'
    filt_pos = f'0 < lag <= {max_lag}'
    g = sns.FacetGrid(dropna=False, **facet_kws, data=recall.reset_index())
    g.map_dataframe(
        lambda data, **kws: sns.lineplot(
            data=data.query(filt_neg), x='lag', y='prob', **kws
        )
    )
    g.map_dataframe(
        lambda data, **kws: sns.lineplot(
            data=data.query(filt_pos), x='lag', y='prob', **kws
        )
    )
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
    """Plot points as a swarm plus mean with error bars."""
    g = sns.FacetGrid(data=data.reset_index(), dropna=False, **facet_kws)
    g.map_dataframe(
        sns.swarmplot, x=x, y=y, color=swarm_color, size=swarm_size, zorder=1
    )
    g.map_dataframe(
        sns.pointplot, x=x, y=y, color=point_color, join=False, capsize=0.5, linewidth=1
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
    """Plot recalls in a raster plot."""
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
