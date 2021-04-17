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

    def split_lists(self, data, phase):
        """
        Get relevant fields and split by list.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw free recall data.

        phase : str
            Phase to split ('study' or 'recall').
        """
        names = list(self.keys.keys())
        keys = list(self.keys.values())
        split = fr.split_lists(data, phase, keys, names, self.item_query, as_list=True)
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
            pool_lists = self.split_lists(subject_data, 'study')
            recall_lists = self.split_lists(subject_data, 'recall')
            results = self.analyze_subject(subject, pool_lists, recall_lists)
            subj_results.append(results)
        stat = pd.concat(subj_results, axis=0)
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
    def __init__(self, list_length, item_query=None, test_key=None, test=None):
        super().__init__(
            'input', 'input', item_query=item_query, test_key=test_key, test=test
        )
        self.list_length = list_length

    def analyze_subject(self, subject, pool, recall):

        actual, possible = transitions.count_lags(
            self.list_length,
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
                'lag': actual.index,
                'prob': actual / possible,
                'actual': actual,
                'possible': possible,
            }
        )
        crp = crp.set_index(['subject', 'lag'])
        return crp


class TransitionLagRank(TransitionMeasure):
    """Measure lag rank of transitions."""
    def __init__(self, item_query=None, test_key=None, test=None):
        super().__init__(
            'input', 'input', item_query=item_query, test_key=test_key, test=test
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
