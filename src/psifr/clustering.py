"""Measures of clustering."""

import numpy as np
import pandas as pd


def lbc(study_category, recall_category):
    """Calculate list-based clustering (LBC) for a set of lists."""
    lbc_scores = np.zeros(len(study_category))
    for i, (study, recall) in enumerate(zip(study_category, recall_category)):
        study = np.asarray(study)
        recall = np.asarray(recall)
        if np.any(pd.isna(study)):
            raise ValueError('Study category contains N/A values.')
        if np.any(pd.isna(recall)):
            raise ValueError('Recall category contains N/A values.')

        # number of correct recalls
        r = len(recall)

        # number of items per category
        m = len(study) / len(np.unique(study))

        # list length
        nl = len(study)

        # observed and expected clustering
        observed = np.count_nonzero(recall[:-1] == recall[1:])
        expected = ((r - 1) * (m - 1)) / (nl - 1)
        lbc_scores[i] = observed - expected
    return lbc_scores


def arc(recall_category):
    """Calculate adjusted ratio of clustering for a set of lists."""
    arc_scores = np.zeros(len(recall_category))
    for i, recall in enumerate(recall_category):
        recall = np.asarray(recall)
        if np.any(pd.isna(recall)):
            raise ValueError('Recall category contains N/A values.')

        # number of categories and correct recalls from each category
        categories = np.unique(recall)
        n = np.array([np.count_nonzero(recall == c) for c in categories])
        c = len(categories)

        # number of correct recalls
        r = len(recall)

        # observed, expected, and maximum clustering
        expected = np.sum((n * (n - 1)) / r)
        observed = np.count_nonzero(recall[:-1] == recall[1:])
        maximum = r - c
        if maximum == expected:
            # when maximum is the same as expected, ARC is undefined
            arc_scores[i] = np.nan
            continue
        arc_scores[i] = (observed - expected) / (maximum - expected)
    return arc_scores
