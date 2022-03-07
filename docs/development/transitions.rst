===========
Transitions
===========

Psifr has a core set of tools for analyzing transitions in free recall data.
These tools focus on measuring what transitions actually occurred, and which
transitions were possible given the order in which participants recalled items.

Actual and possible transitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculating a conditional response probability involves two parts: the frequency
at which a given event actually occurred in the data and frequency at which a
given event could have occurred. The frequency of possible events is
calculated conditional on the recalls that have been made leading up to each
transition. For example, a transition between item :math:`i` and item :math:`j`
is not considered "possible" in a CRP analysis if item :math:`i` was never
recalled. The transition is also not considered "possible" if, when item
:math:`i` is recalled, item :math:`j` has already been recalled previously.

Repeated recall events are typically excluded from the counts of both actual
and possible transition events. That is, the transition event frequencies are
conditional on the transition not being either to or from a repeated item.

Calculating a CRP measure involves tallying how many transitions of a given
type were made during a free recall test. For example, one common measure is
the serial position lag between items. For a list of length :math:`N`, possible
lags are in the range :math:`[-N+1, N-1]`. Because repeats are excluded, a lag
of zero is never possible. The count of actual and possible transitions for
each lag is calculated first, and then the CRP for each lag is calculated as
the actual count divided by the possible count.

The transitions masker
~~~~~~~~~~~~~~~~~~~~~~

The :py:func:`psifr.transitions.transitions_masker` is a generator that makes
it simple to iterate over transitions while "masking" out events such as
intrusions of items not on the list and repeats of items that have already
been recalled.

On each step of the iterator, the previous, current, and possible items are
yielded. The *previous*
item is the item being transitioned from. The *current* item is the item being
transitioned to. The *possible* items includes an array of all items that
were valid to be recalled next, given the recall sequence up to that point (not
including the current item).

.. ipython::

    In [1]: from psifr.transitions import transitions_masker

    In [2]: pool = [1, 2, 3, 4, 5, 6]

    In [3]: recs = [6, 2, 3, 6, 1, 4]

    In [4]: masker = transitions_masker(pool_items=pool, recall_items=recs,
       ...:                             pool_output=pool, recall_output=recs)

    In [5]: for op, prev, curr, poss in masker:
       ...:     print(op, prev, curr, poss)
       ...:

Only valid transitions are yielded, so the code
for a specific analysis only needs to calculate the transition measure of
interest and count the number of actual and possible transitions in each bin
of interest.

Four inputs are required:

`pool_items`
    List of identifiers for all items available for recall. Identifiers
    can be anything that is unique to each item in the list (e.g., serial
    position, a string representation of the item, an index in the stimulus
    pool).

`recall_items`
    List of identifiers for the sequence of recalls, in order. Valid recalls
    must match an item in `pool_items`. Other items are considered intrusions.

`pool_output`
    Output codes for each item in the pool. This should be whatever you need to
    calculate your transition measure.

`recall_output`
    Output codes for each recall in the sequence of recalls.

By using different values for these four inputs and defining different
transition measures, a wide range of analyses can be implemented.
