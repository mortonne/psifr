========
Measures
========

.. currentmodule:: psifr.measures

The :py:mod:`~psifr.measures` module connects the lower-level statistics defined in the :py:mod:`~psifr.stats` module with high-level analyses defined in the :py:mod:`~psifr.fr` module. Each of the measures defined here takes table-format free-recall data, extracts information necessary for a given statistic, exports it to list format, calculates the statistic, and packages it within a results table for output.

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
