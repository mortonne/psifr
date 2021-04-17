"""Test plotting functions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytest

from psifr import fr


@pytest.fixture()
def data():
    """Create merged free recall data."""
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
                'absence', 'hollow', 'pupil', 'pupil', 'absence', 'empty',
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
        }
    )
    data = fr.merge_free_recall(raw, study_keys=['task'], list_keys=['item_index'])
    return data


@pytest.fixture()
def distances():
    """Create a pairwise distance matrix."""
    mat = np.array(
        [
            [0, 1, 2, 2, 2, 2],
            [1, 0, 1, 2, 2, 2],
            [2, 1, 0, 2, 2, 2],
            [2, 2, 2, 0, 2, 3],
            [2, 2, 2, 2, 0, 2],
            [2, 2, 2, 3, 2, 0],
        ],
    )
    return mat


def test_plot_spc(data):
    """Test plotting a serial position curve."""
    recall = fr.spc(data)
    g = fr.plot_spc(recall)
    plt.close()
    assert isinstance(g, sns.FacetGrid)


def test_plot_lag_crp(data):
    """Test plotting a lag-CRP curve."""
    stat = fr.lag_crp(data)
    g = fr.plot_lag_crp(stat)
    plt.close()
    assert isinstance(g, sns.FacetGrid)


def test_plot_distance_crp(data, distances):
    """Test plotting a distance CRP curve."""
    edges = [0.5, 1.5, 2.5, 3.5]
    stat = fr.distance_crp(data, 'item_index', distances, edges)
    g = fr.plot_distance_crp(stat, min_samples=2)
    plt.close()
    assert isinstance(g, sns.FacetGrid)


def test_plot_swarm_error():
    """Test creating swarm plot with error bars."""
    stat = pd.DataFrame(
        {
            'category': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'value': [1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9],
        }
    )
    g = fr.plot_swarm_error(stat, x='category', y='value')
    plt.close()
    assert isinstance(g, sns.FacetGrid)


def test_plot_raster(data):
    """Test creating a raster plot."""
    g = fr.plot_raster(data)
    plt.close()
    assert isinstance(g, sns.FacetGrid)

    g = fr.plot_raster(data, orientation='vertical')
    plt.close()
    assert isinstance(g, sns.FacetGrid)
