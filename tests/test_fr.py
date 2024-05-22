"""Test high-level operations in the fr module."""

import numpy as np
import pandas as pd
import pytest

from psifr import measures
from psifr import fr


@pytest.fixture()
def raw():
    """Create raw free recall data."""
    raw = pd.DataFrame(
        {
            'subject': [
                1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
            ],
            'list': [
                1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2,
            ],
            'trial_type': [
                'study', 'study', 'study', 'recall', 'recall', 'recall',
                'study', 'study', 'study', 'recall', 'recall', 'recall',
            ],
            'position': [
                1, 2, 3, 1, 2, 3,
                1, 2, 3, 1, 2, 3,
            ],
            'item': [
                'absence', 'hollow', 'pupil', 'hollow', 'pupil', 'empty',
                'fountain', 'piano', 'pillow', 'pillow', 'fountain', 'pillow',
            ],
            'item_index': [
                0, 1, 2, 1, 2, np.nan,
                3, 4, 5, 5, 3, 5,
            ],
            'task': [
                1, 2, 1, 2, 1, np.nan,
                1, 2, 1, 1, 1, 1,
            ],
            'block': [
                1, 2, 2, 2, 2, np.nan,
                1, 1, 2, 2, 1, 2,
            ],
        }
    )
    return raw


@pytest.fixture()
def data(raw):
    """Create merged free recall data."""
    data = fr.merge_free_recall(
        raw, study_keys=['task', 'block'], list_keys=['item_index']
    )
    return data


@pytest.fixture()
def distances():
    """Create item distance matrix."""
    mat = np.array(
        [
            [0, 1, 2, 2, 2, 2],
            [1, 0, 1, 2, 2, 2],
            [2, 1, 0, 2, 2, 2],
            [2, 2, 2, 0, 2, 3],
            [2, 2, 2, 2, 0, 2],
            [2, 2, 2, 3, 2, 0],
        ]
    )
    return mat


@pytest.fixture()
def distances2():
    """Create another item distance matrix."""
    distances = np.array(
        [
            [0, 1, 3, 3, 3, 3],
            [1, 0, 3, 3, 3, 3],
            [3, 3, 0, 2, 1, 2],
            [3, 3, 2, 0, 2, 2],
            [3, 3, 1, 2, 0, 2],
            [3, 3, 2, 2, 2, 0],
        ]
    )
    return distances


def test_table_from_lists():
    """Test creating a data table from lists."""
    subjects = [1, 1, 2, 2]
    lists = [1, 2, 3, 4]
    study = [
        ['absence', 'hollow', 'pupil'],
        ['fountain', 'piano', 'pillow'],
        ['fountain', 'piano', 'pillow'],
        ['absence', 'hollow', 'pupil'],
    ]
    recall = [
        ['hollow', 'pupil', 'empty'],
        ['fountain', 'piano', 'fountain'],
        ['pillow', 'piano'],
        ['pupil'],
    ]
    category = (
        [[1, 1, 2], [2, 2, 2], [1, 1, 2], [2, 2, 2]],
        [[1, 2, 1], [2, 2, 2], [2, 2], [2]],
    )
    task = ([[1, 2, 1], [2, 1, 2], [2, 1, 2], [1, 2, 1]], None)

    # explicit labeling of list number
    data = fr.table_from_lists(
        subjects, study, recall, lists=lists, category=category, task=task
    )

    # subject, list, and trial_type labels
    np.testing.assert_array_equal(
        data['subject'].to_numpy(), np.repeat([1, 2], [12, 9])
    )
    np.testing.assert_array_equal(
        data['list'].to_numpy(), np.repeat([1, 2, 3, 4], [6, 6, 5, 4])
    )
    np.testing.assert_array_equal(
        data['trial_type'].to_numpy(),
        np.repeat(np.tile(['study', 'recall'], 4), [3, 3, 3, 3, 3, 2, 3, 1]),
    )

    # position
    n = [len(items) for seqs in zip(study, recall) for items in seqs]
    position = np.array([i for j in n for i in range(1, j + 1)])
    np.testing.assert_array_equal(data['position'].to_numpy(), position)

    # items
    items = [item for phase in zip(study, recall) for items in phase for item in items]
    np.testing.assert_array_equal(data['item'].to_numpy(), np.array(items))

    # category
    category = [cat for phase in zip(*category) for cats in phase for cat in cats]
    np.testing.assert_array_equal(data['category'].to_numpy(), category)

    # task
    task_recall = [np.tile(np.nan, n) for n in [3, 3, 2, 1]]
    task = np.hstack([t for t_list in zip(task[0], task_recall) for t in t_list])
    np.testing.assert_array_equal(data['task'].to_numpy(), task)

    # implicit labeling of list number
    data = fr.table_from_lists(subjects, study, recall)
    np.testing.assert_array_equal(
        data['list'].to_numpy(), np.repeat([1, 2, 1, 2], [6, 6, 5, 4])
    )


def test_merge(raw):
    """Test merging of study and recall trials."""
    study = raw.loc[raw['trial_type'] == 'study'].copy()
    recall = raw.loc[raw['trial_type'] == 'recall'].copy()
    merged = fr.merge_lists(study, recall)

    # correct recall
    correct = merged.query('item == "pupil"')
    correct = correct.reset_index().loc[0]
    assert correct['input'] == 3
    assert correct['study']
    assert correct['recall']
    assert correct['repeat'] == 0
    assert not correct['intrusion']

    # item not recalled
    forgot = merged.query('item == "piano"')
    forgot = forgot.reset_index().loc[0]
    assert forgot['input'] == 2
    assert forgot['study']
    assert not forgot['recall']
    assert not forgot['intrusion']

    # intrusion
    intrusion = merged.query('item == "empty"')
    intrusion = intrusion.reset_index().loc[0]
    assert pd.isnull(intrusion['input'])
    assert not intrusion['study']
    assert intrusion['recall']
    assert intrusion['repeat'] == 0
    assert intrusion['intrusion']

    # repeat
    repeat = merged.query('item == "pillow" and output == 3')
    repeat = repeat.reset_index().loc[0]
    assert repeat['input'] == 3
    assert not repeat['study']
    assert repeat['recall']
    assert repeat['repeat'] == 1
    assert not repeat['intrusion']


def test_pli(raw):
    """Test labeling of prior-list intrusions."""
    data = raw.copy()
    data.loc[3:5, 'item'] = ['hollow', 'pupil', 'fountain']
    data.loc[9:11, 'item'] = ['pillow', 'fountain', 'pupil']
    merged = fr.merge_free_recall(data)
    assert 'prior_list' in merged.columns
    assert 'prior_input' in merged.columns

    # check the PLI (prior-list intrusion) in the second list
    pli = merged.query('item == "pupil" and output == 3')
    pli = pli.reset_index().loc[0]
    assert pli['prior_list'] == 1
    assert pli['prior_input'] == 3

    # check the FLI (future-list intrusion) in the first list
    fli = merged.query('item == "fountain" and output == 3')
    assert pd.isnull(fli['prior_list'].to_numpy()[0])
    assert pd.isnull(fli['prior_input'].to_numpy()[0])


def test_filter_raw_data(raw):
    """Test filtering raw data."""
    filt = fr.filter_data(raw, 1, [1, 2], 'study', positions=[1, 2])
    assert filt['item'].to_list() == ['absence', 'hollow', 'fountain', 'piano']


def test_filter_merged_data(data):
    """Test filtering merged data."""
    filt = fr.filter_data(data, 1, [1, 2], inputs=[1, 2])
    assert filt['item'].to_list() == ['absence', 'hollow', 'fountain', 'piano']

    filt = fr.filter_data(data, 1, [1, 2], outputs=1.0)
    assert filt['item'].to_list() == ['hollow', 'pillow']


def test_split_lists(data):
    """Test splitting lists for study and recall data."""
    study = fr.split_lists(data, 'study', ['item', 'input', 'task'])
    np.testing.assert_allclose(study['input'][1], np.array([1.0, 2.0, 3.0]))
    recall = fr.split_lists(data, 'recall', ['input'], ['recalls'])
    np.testing.assert_allclose(recall['recalls'][0], np.array([2.0, 3.0, np.nan]))


def test_lists_input(data):
    """Test splitting lists through the transitions interface."""
    m = measures.TransitionMeasure('input', 'input')
    pool_lists = m.split_lists(data, 'study')
    pool_expected = {
        'items': [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        'label': [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        'test': None,
    }
    assert pool_lists == pool_expected

    recall_lists = m.split_lists(data, 'recall')
    recall_expected = {
        'items': [[2.0, 3.0, np.nan], [3.0, 1.0, 3.0]],
        'label': [[2.0, 3.0, np.nan], [3.0, 1.0, 3.0]],
        'test': None,
    }
    for key in recall_expected.keys():
        assert key in recall_lists
        np.testing.assert_array_equal(recall_lists[key], recall_expected[key])


def test_spc(data):
    """Test serial position curve analysis."""
    recall = fr.spc(data)
    expected = np.array([0.5, 0.5, 1])
    np.testing.assert_array_equal(recall['recall'].to_numpy(), expected)


def test_pnr(data):
    """Test probability of nth recall analysis."""
    stat = fr.pnr(data)
    expected = np.array([0, 0.5, 0.5, 0.5, 0, 1, np.nan, np.nan, np.nan])
    observed = stat['prob'].to_numpy()
    np.testing.assert_array_equal(expected, observed)


def test_pli_list_lag():
    """Test proportion of list lags for prior-list intrusions."""
    subjects = [1, 1, 1, 1, 2, 2, 2, 2]
    study = [
        ['tree', 'cat'],
        ['absence', 'hollow'],
        ['fountain', 'piano'],
        ['pillow', 'pupil'],
        ['tree', 'cat'],
        ['absence', 'hollow'],
        ['fountain', 'piano'],
        ['pillow', 'pupil'],
    ]
    recall = [
        ['tree', 'cat'],
        ['hollow', 'absence'],
        ['fountain', 'hollow'],
        ['absence', 'piano', 'cat'],
        ['tree', 'cat'],
        ['absence', 'hollow'],
        ['fountain', 'piano'],
        ['pillow', 'pupil'],
    ]
    raw = fr.table_from_lists(subjects, study, recall)
    data = fr.merge_free_recall(raw)

    # max lag 1 (exclude just the first list)
    stat = fr.pli_list_lag(data, max_lag=1)
    np.testing.assert_array_equal(stat['count'].to_numpy(), np.array([2, 0]))
    np.testing.assert_array_equal(stat['per_list'].to_numpy(), np.array([2 / 3, 0]))
    np.testing.assert_array_equal(stat['prob'].to_numpy(), np.array([0.5, np.nan]))

    # max lag 2 (exclude first two lists)
    stat = fr.pli_list_lag(data, max_lag=2)
    np.testing.assert_array_equal(stat['count'].to_numpy(), np.array([2, 1, 0, 0]))
    np.testing.assert_array_equal(stat['per_list'].to_numpy(), np.array([1, 0.5, 0, 0]))
    np.testing.assert_array_equal(
        stat['prob'].to_numpy(), np.array([0.5, 0.25, np.nan, np.nan])
    )

    # max lag 3 (exclude first three lists)
    stat = fr.pli_list_lag(data, max_lag=3)
    np.testing.assert_array_equal(
        stat['count'].to_numpy(), np.array([1, 1, 1, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        stat['per_list'].to_numpy(), np.array([1, 1, 1, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        stat['prob'].to_numpy(), np.array([1 / 3, 1 / 3, 1 / 3, np.nan, np.nan, np.nan])
    )


def test_lag_crp(data):
    """Test basic lag-CRP analysis."""
    crp = fr.lag_crp(data)
    actual = np.array([1, 0, 0, 1, 0])
    possible = np.array([1, 2, 0, 1, 0])
    prob = np.array([1.0, 0.0, np.nan, 1.0, np.nan])

    np.testing.assert_array_equal(crp['actual'], actual)
    np.testing.assert_array_equal(crp['possible'], possible)
    np.testing.assert_array_equal(crp['prob'], prob)


def test_lag_crp_query_input(data):
    """Test lag-CRP analysis with item input position filter."""
    crp = fr.lag_crp(data, item_query='input != 2')
    actual = np.array([1, 0, 0, 0, 0])
    possible = np.array([1, 0, 0, 0, 0])
    prob = np.array([1.0, np.nan, np.nan, np.nan, np.nan])

    np.testing.assert_array_equal(crp['actual'], actual)
    np.testing.assert_array_equal(crp['possible'], possible)
    np.testing.assert_array_equal(crp['prob'], prob)


def test_lag_crp_query_output(data):
    """Test lag-CRP analysis with item output position filter."""
    crp = fr.lag_crp(data, item_query='output > 1 or not recall')

    np.testing.assert_array_equal(crp['actual'], np.zeros(5))
    np.testing.assert_array_equal(crp['possible'], np.zeros(5))
    expected = np.empty(5)
    expected.fill(np.nan)
    np.testing.assert_array_equal(crp['prob'], expected)


def test_lag_crp_block(data):
    """Test block-CRP analysis."""
    crp = fr.lag_crp(data, lag_key='block', count_unique=True)
    # recalls: [[2, 3, NaN], [3, 1, 3]]
    # blocks:  [[2, 2, NaN], [2, 1, 2]]
    np.testing.assert_array_equal(crp['actual'], np.array([1, 1, 0]))
    np.testing.assert_array_equal(crp['possible'], np.array([2, 1, 0]))
    np.testing.assert_array_equal(crp['prob'], np.array([0.5, 1.0, np.nan]))


def test_lag_crp_cat(data):
    """Test lag-CRP conditional on within-category transitions."""
    crp = fr.lag_crp(data, test_key='task', test=lambda x, y: x == y)
    actual = np.array([1, 0, 0, 0, 0])
    possible = np.array([1, 0, 0, 0, 0])
    prob = np.array([1.0, np.nan, np.nan, np.nan, np.nan])

    np.testing.assert_array_equal(crp['actual'], actual)
    np.testing.assert_array_equal(crp['possible'], possible)
    np.testing.assert_array_equal(crp['prob'], prob)


def test_lag_crp_compound():
    """Test compound lag-CRP analysis."""
    subjects = [1, 1]
    study = [
        ['absence', 'hollow', 'pupil', 'fountain'],
        ['tree', 'cat', 'house', 'dog'],
    ]
    recall = [
        ['fountain', 'hollow', 'absence'],
        ['mouse', 'cat', 'tree', 'house', 'dog'],
    ]
    raw = fr.table_from_lists(subjects, study, recall)
    data = fr.merge_free_recall(raw)
    crp = fr.lag_crp_compound(data)
    # -2, -1
    # NaN, -1, +2, +1
    actual = np.hstack(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    possible = np.hstack(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(actual, crp['actual'].to_numpy())
    np.testing.assert_array_equal(possible, crp['possible'].to_numpy())


def test_distance_crp(data, distances2):
    """Test distance CRP analysis."""
    edges = [0.5, 1.5, 2.5, 3.5]
    crp = fr.distance_crp(data, 'item_index', distances2, edges, count_unique=False)

    actual = np.array([0, 1, 1])
    possible = np.array([1, 2, 1])
    prob = np.array([0, 0.5, 1])

    np.testing.assert_array_equal(crp['actual'], actual)
    np.testing.assert_array_equal(crp['possible'], possible)
    np.testing.assert_array_equal(crp['prob'], prob)


def test_distance_crp_unique(data, distances2):
    """Test distance CRP analysis with unique counts only."""
    edges = [0.5, 1.5, 2.5, 3.5]
    crp = fr.distance_crp(data, 'item_index', distances2, edges, count_unique=True)

    actual = np.array([0, 1, 1])
    possible = np.array([1, 1, 1])
    prob = np.array([0, 1, 1])

    np.testing.assert_array_equal(crp['actual'], actual)
    np.testing.assert_array_equal(crp['possible'], possible)
    np.testing.assert_array_equal(crp['prob'], prob)


def test_category_crp(data):
    """Test category CRP analysis."""
    # TODO: replace with more diagnostic test
    crp = fr.category_crp(data, 'task')

    assert crp['prob'].iloc[0] == 1
    assert crp['actual'].iloc[0] == 1
    assert crp['possible'].iloc[0] == 1


def test_category_clustering():
    """Test category clustering statistics."""
    subject = [1] * 2

    # category of study and list items (two cases from category
    # clustering tests)
    study_category = [list('abcd') * 4] * 2
    recall_str = ['aaabbbcccddd', 'aabbcdcd']
    recall_category = [list(s) for s in recall_str]

    # unique item codes (needed for merging study and recall events;
    # not actually needed for the stats)
    study_item = [[i for i in range(len(c))] for c in study_category]
    recall_item = [[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], [0, 4, 1, 5, 2, 3, 6, 7]]

    # create merged free recall data
    raw = fr.table_from_lists(
        subject, study_item, recall_item, category=(study_category, recall_category)
    )
    data = fr.merge_free_recall(raw, list_keys=['category'])

    # test ARC and LBC stats
    stats = fr.category_clustering(data, 'category')
    np.testing.assert_allclose(stats.loc[0, 'arc'], 0.667, rtol=0.011)
    np.testing.assert_allclose(stats.loc[0, 'lbc'], 3.2, rtol=0.011)


def test_lag_rank(data):
    """Test lag rank analysis."""
    stat = fr.lag_rank(data)
    expected = np.array([0.25])
    observed = stat['rank'].to_numpy()
    np.testing.assert_array_equal(expected, observed)


def test_distance_rank(data, distances):
    """Test distance rank analysis."""
    stat = fr.distance_rank(data, 'item_index', distances)
    expected = np.array([0.25])
    observed = stat['rank'].to_numpy()
    np.testing.assert_array_equal(expected, observed)


def test_distance_rank_shifted():
    """Test shifted distance rank analysis."""
    distances = np.array(
        [
            [0, 1, 1, 1, 2, 2, 2, 2],
            [1, 0, 1, 4, 2, 2, 2, 2],
            [1, 1, 0, 1, 2, 2, 2, 2],
            [1, 4, 1, 0, 2, 2, 2, 2],
            [2, 2, 2, 2, 0, 3, 3, 3],
            [2, 2, 2, 2, 3, 0, 3, 3],
            [2, 2, 2, 2, 3, 3, 0, 3],
            [2, 2, 2, 2, 3, 3, 3, 0],
        ]
    )
    subjects = [1]
    study = [
        ['absence', 'hollow', 'pupil', 'fountain', 'piano', 'pillow', 'cat', 'tree']
    ]
    recall = [
        ['piano', 'fountain', 'hollow', 'tree', 'fountain', 'absence', 'cat', 'pupil']
    ]
    item_index = ([[0, 1, 2, 3, 4, 5, 6, 7]], [[4, 3, 1, 7, 3, 0, 6, 2]])
    raw = fr.table_from_lists(subjects, study, recall, item_index=item_index)
    data = fr.merge_free_recall(raw, list_keys=['item_index'])
    stat = fr.distance_rank_shifted(data, 'item_index', distances, 2)

    expected = np.array([0.683333, 0.416667])
    np.testing.assert_allclose(expected, stat['rank'].to_numpy(), atol=0.000001)


def test_distance_rank_window():
    """Test distance rank window analysis."""
    distances = np.array(
        [
            [0, 3, 1, 1, 2, 2, 2, 2],
            [3, 0, 1, 1, 2, 2, 2, 2],
            [1, 1, 0, 4, 2, 2, 2, 2],
            [1, 1, 4, 0, 2, 2, 2, 2],
            [2, 2, 2, 2, 0, 5, 1, 1],
            [2, 2, 2, 2, 5, 0, 1, 1],
            [2, 2, 2, 2, 1, 1, 0, 6],
            [2, 2, 2, 2, 1, 1, 6, 0],
        ]
    )
    subjects = [1]
    study = [
        ['absence', 'hollow', 'pupil', 'fountain', 'piano', 'pillow', 'cat', 'tree']
    ]
    recall = [['fountain', 'hollow', 'absence', 'cat', 'piano', 'pupil', 'fountain']]
    item_index = ([[0, 1, 2, 3, 4, 5, 6, 7]], [[3, 1, 0, 6, 4, 2, 3]])
    raw = fr.table_from_lists(subjects, study, recall, item_index=item_index)
    data = fr.merge_free_recall(raw, list_keys=['item_index'])
    stat = fr.distance_rank_window(data, 'item_index', distances, [-1, 0, 1])
    expected = np.array([[0.875, 0.875, 0.375], [0, 1, 1], [0, 0, 0]])
    np.testing.assert_allclose(np.mean(expected, 0), stat['rank'].to_numpy())

    # example from stats function test
    distances = np.array(
        [
            [0, 3, 1, 1, 2, 2, 2, 2],
            [3, 0, 1, 1, 2, 2, 2, 2],
            [1, 1, 0, 4, 2, 2, 2, 2],
            [1, 1, 4, 0, 2, 2, 2, 2],
            [2, 2, 2, 2, 0, 5, 3, 3],
            [2, 2, 2, 2, 5, 0, 3, 3],
            [2, 2, 2, 2, 3, 3, 0, 6],
            [2, 2, 2, 2, 3, 3, 6, 0],
        ]
    )
    pool = ['piano', 'pillow', 'cat', 'tree', 'absence', 'hollow', 'pupil', 'fountain']
    recall = [['fountain', 'hollow', 'absence', 'cat', 'piano', 'pillow', 'tree']]
    raw = fr.table_from_lists(subjects, study, recall)
    raw['item_index'] = fr.pool_index(raw['item'], pool)
    data = fr.merge_free_recall(raw, list_keys=['item_index'])

    # including all transitions outside the window
    stat = fr.distance_rank_window(data, 'item_index', distances, [-1, 0, 1])
    expected = np.array([[0.125, 0.125, 0.375], [0, 1, 1], [1, 1, 0]])
    np.testing.assert_allclose(np.mean(expected, 0), stat['rank'].to_numpy())

    # excluding transitions when the prior transition was from the window
    stat = fr.distance_rank_window(
        data, 'item_index', distances, [-1, 0, 1], exclude_prev_window=True
    )
    expected = np.array([[0.125, 0.125, 0.375], [0, 1, 1]])
    np.testing.assert_allclose(np.mean(expected, 0), stat['rank'].to_numpy())

    # just calculating asymmetry
    stat = fr.distance_rank_window_asym(data, 'item_index', distances)
    expected = np.array([[.25, 1, -1]])
    np.testing.assert_allclose(np.mean(expected), stat['asym'].to_numpy())
