===========
Transitions
===========

.. currentmodule:: psifr.transitions

Counting transitions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    count_lags
    count_category
    count_distance

Ranking transitions
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    percentile_rank
    rank_lags
    rank_distance

Iterating over recalls
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

    outputs_masker
    transitions_masker

Transition measure base class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    TransitionMeasure
    TransitionMeasure.split_lists
    TransitionMeasure.analyze
    TransitionMeasure.analyze_subject

Transition measures
~~~~~~~~~~~~~~~~~~~

.. autosummary::

    TransitionOutputs
    TransitionLag
    TransitionLagRank
    TransitionCategory
    TransitionDistance
    TransitionDistanceRank
