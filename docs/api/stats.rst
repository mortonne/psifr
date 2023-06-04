==========
Statistics
==========

.. currentmodule:: psifr.stats

The :py:mod:`~psifr.stats` module provides a set of functions for calculating statistics of study and recall sequences of individual participants.

Counting recalls by serial position and output position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    count_outputs

Counting transitions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    count_lags
    count_lags_compound
    count_category
    count_distance

Ranking transitions
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    percentile_rank
    rank_lags
    rank_distance
    rank_distance_shifted
