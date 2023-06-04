"""Classes for defining recall measures."""

import abc

import numpy as np
import pandas as pd

from psifr import fr
from psifr import transitions
from psifr import outputs


class TransitionMeasure(object):
    """
    Measure of free recall dataset with multiple subjects.

    Parameters
    ----------
    items_key : str
        Data column with item identifiers.

    label_key : str
        Data column with trial labels to use for the measure.

    item_query : str
        Query string to indicate trials to include in the measure.

    test_key : str
        Data column with labels to use when testing for trial inclusion.

    test : callable
        Test of trial inclusion. Takes the previous and current test
        values and return True if the transition should be included.

    Attributes
    ----------
    keys : dict of {str: str}
        List of columns to use for the measure.

    item_query : str
        Query string to indicate trials to include in the measure.

    test : callable
        Test of trial inclusion.
    """

    def __init__(self, items_key, label_key, item_query=None, test_key=None, test=None):

        self.keys = {'items': items_key, 'label': label_key, 'test': test_key}
        self.item_query = item_query
        self.test = test

    def split_lists(self, data, phase, item_query=None):
        """
        Get relevant fields and split by list.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw free recall data.

        phase : str
            Phase to split ('study' or 'recall').

        item_query : str, optional
            Query string to determine included trials.
        """
        names = list(self.keys.keys())
        keys = list(self.keys.values())
        for key in keys:
            if (key is not None) and (key not in data.columns):
                raise ValueError(f'Required column {key} is missing.')
        split = fr.split_lists(data, phase, keys, names, item_query, as_list=True)
        return split

    @abc.abstractmethod
    def analyze_subject(self, subject, pool_lists, recall_lists):
        """
        Analyze a single subject.

        Parameters
        ----------
        subject : int or str
            Identifier of the subject to analyze.

        pool_lists : dict of lists of numpy.ndarray
            Information about the item pool for each list, with keys
            for items, label, and test arrays.

        recall_lists : dict of lists of numpy.ndarray
            Information about the recall sequence for each list, with
            keys for items, label, and test arrays.

        Returns
        -------
        pandas.DataFrame
            Results of the analysis for one subject. Should include a
            'subject' column in the index.
        """
        pass

    def analyze(self, data):
        """
        Analyze a free recall dataset with multiple subjects.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw (not merged) free recall data.

        Returns
        -------
        stat : pandas.DataFrame
            Statistics calculated for each subject.
        """
        subj_results = []
        for subject, subject_data in data.groupby('subject'):
            pool_lists = self.split_lists(subject_data, 'study', self.item_query)
            recall_lists = self.split_lists(subject_data, 'recall')
            results = self.analyze_subject(subject, pool_lists, recall_lists)
            subj_results.append(results)
        stat = pd.concat(subj_results, axis=0).reset_index()
        return stat


class TransitionOutputs(TransitionMeasure):
    """Measure recall probability by input and output position."""

    def __init__(self, list_length, item_query=None, test_key=None, test=None):
        super().__init__(
            'input', 'input', item_query=item_query, test_key=test_key, test=test
        )
        self.list_length = list_length

    def analyze_subject(self, subject, pool, recall):
        actual, possible = outputs.count_outputs(
            self.list_length,
            pool['items'],
            recall['items'],
            pool['label'],
            recall['label'],
            pool['test'],
            recall['test'],
            self.test,
        )
        input_pos = np.tile(np.arange(1, actual.shape[1] + 1), actual.shape[0])
        output_pos = np.repeat(np.arange(1, actual.shape[0] + 1), actual.shape[1])
        with np.errstate(divide='ignore', invalid='ignore'):
            prob = actual.flatten() / possible.flatten()
        pnr = pd.DataFrame(
            {
                'subject': subject,
                'input': input_pos,
                'output': output_pos,
                'prob': prob,
                'actual': actual.flatten(),
                'possible': possible.flatten(),
            }
        )
        pnr = pnr.set_index(['subject', 'output', 'input'])
        return pnr


class TransitionLag(TransitionMeasure):
    """Measure conditional response probability by lag."""

    def __init__(
        self,
        list_length,
        lag_key='input',
        count_unique=False,
        item_query=None,
        test_key=None,
        test=None,
        compound=False,
    ):
        super().__init__(
            'input', lag_key, item_query=item_query, test_key=test_key, test=test
        )
        self.list_length = list_length
        self.count_unique = count_unique
        self.compound = compound

    def analyze_subject(self, subject, pool, recall):

        if self.compound:
            counter = transitions.count_lags_compound
        else:
            counter = transitions.count_lags

        actual, possible = counter(
            self.list_length,
            pool['items'],
            recall['items'],
            pool['label'],
            recall['label'],
            pool['test'],
            recall['test'],
            self.test,
            self.count_unique,
        )
        crp = pd.DataFrame(
            {
                'subject': subject,
                'prob': actual / possible,
                'actual': actual,
                'possible': possible,
            }, index=actual.index
        )
        if self.compound:
            crp = crp.set_index('subject', append=True)
            crp = crp.reorder_levels(['subject', 'previous', 'current'])
        else:
            crp = crp.set_index('subject', append=True)
            crp = crp.reorder_levels(['subject', 'lag'])
        return crp


class TransitionLagRank(TransitionMeasure):
    """Measure lag rank of transitions."""

    def __init__(self, lag_key='input', item_query=None, test_key=None, test=None):
        super().__init__(
            'input', lag_key, item_query=item_query, test_key=test_key, test=test
        )

    def analyze_subject(self, subject, pool, recall):
        ranks = transitions.rank_lags(
            pool['items'],
            recall['items'],
            pool['label'],
            recall['label'],
            pool['test'],
            recall['test'],
            self.test,
        )
        stat = pd.DataFrame(
            {'subject': subject, 'rank': np.nanmean(ranks)}, index=[subject]
        )
        stat = stat.set_index('subject')
        return stat


class TransitionDistance(TransitionMeasure):
    """Measure conditional response probability by distance."""

    def __init__(
        self,
        index_key,
        distances,
        edges,
        count_unique=False,
        centers=None,
        item_query=None,
        test_key=None,
        test=None,
    ):
        super().__init__(
            'input', index_key, item_query=item_query, test_key=test_key, test=test
        )
        self.distances = distances
        self.edges = edges
        if centers is None:
            # if no explicit centers, use halfway between edges
            centers = edges[:-1] + (np.diff(edges) / 2)
        self.centers = centers
        self.count_unique = count_unique

    def analyze_subject(self, subject, pool, recall):

        actual, possible = transitions.count_distance(
            self.distances,
            self.edges,
            pool['items'],
            recall['items'],
            pool['label'],
            recall['label'],
            pool['test'],
            recall['test'],
            self.test,
            count_unique=self.count_unique,
        )
        crp = pd.DataFrame(
            {
                'subject': subject,
                'center': self.centers,
                'bin': actual.index,
                'prob': actual / possible,
                'actual': actual,
                'possible': possible,
            }
        )
        crp = crp.set_index(['subject', 'center'])
        return crp


class TransitionDistanceRank(TransitionMeasure):
    """Measure transition rank by distance."""

    def __init__(self, index_key, distances, item_query=None, test_key=None, test=None):
        super().__init__(
            index_key, index_key, item_query=item_query, test_key=test_key, test=test
        )
        self.distances = distances

    def analyze_subject(self, subject, pool, recall):
        ranks = transitions.rank_distance(
            self.distances,
            pool['items'],
            recall['items'],
            pool['label'],
            recall['label'],
            pool['test'],
            recall['test'],
            self.test,
        )
        stat = pd.DataFrame(
            {'subject': subject, 'rank': np.nanmean(ranks)}, index=[subject]
        )
        stat = stat.set_index('subject')
        return stat


class TransitionDistanceRankShifted(TransitionMeasure):
    """Measure shifted transition rank by distance."""

    def __init__(
        self, index_key, distances, max_shift, item_query=None, test_key=None, test=None
    ):
        super().__init__(
            index_key, index_key, item_query=item_query, test_key=test_key, test=test
        )
        self.distances = distances
        self.max_shift = int(max_shift)

    def analyze_subject(self, subject, pool, recall):
        ranks = transitions.rank_distance_shifted(
            self.distances,
            self.max_shift,
            pool['items'],
            recall['items'],
            pool['label'],
            recall['label'],
            pool['test'],
            recall['test'],
            self.test,
        )
        shifts = np.arange(-self.max_shift, 0)
        index = pd.MultiIndex.from_arrays(
            [[subject] * self.max_shift, shifts], names=['subject', 'shift']
        )
        stat = pd.DataFrame({'rank': np.nanmean(ranks, 0)}, index=index)
        return stat


class TransitionDistanceRankWindow(TransitionMeasure):
    """Measure transition distance rank within a window."""

    def __init__(
        self,
        index_key,
        distances,
        list_length,
        window_lags,
        item_query=None,
        test_key=None,
        test=None,
    ):
        super().__init__(
            'input', index_key, item_query=item_query, test_key=test_key, test=test
        )
        self.distances = distances
        self.list_length = list_length
        self.window_lags = window_lags

    def analyze_subject(self, subject, pool, recall):
        ranks = transitions.rank_distance_window(
            self.distances,
            self.list_length,
            self.window_lags,
            pool['items'],
            recall['items'],
            pool['label'],
            recall['label'],
            pool['test'],
            recall['test'],
            self.test,
        )
        index = pd.MultiIndex.from_arrays(
            [[subject] * len(self.window_lags), self.window_lags],
            names=['subject', 'lag'],
        )
        stat = pd.DataFrame({'rank': np.nanmean(ranks, 0)}, index=index)
        return stat


class TransitionCategory(TransitionMeasure):
    """Measure conditional response probability by category transition."""

    def __init__(self, category_key, item_query=None, test_key=None, test=None):
        super().__init__(
            'input', category_key, item_query=item_query, test_key=test_key, test=test
        )

    def analyze_subject(self, subject, pool, recall):
        actual, possible = transitions.count_category(
            pool['items'],
            recall['items'],
            pool['label'],
            recall['label'],
            pool['test'],
            recall['test'],
            self.test,
        )
        crp = pd.DataFrame(
            {
                'subject': subject,
                'prob': actual / possible,
                'actual': actual,
                'possible': possible,
            },
            index=[subject],
        )
        crp = crp.set_index('subject')
        return crp
