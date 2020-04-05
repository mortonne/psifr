Scoring data
============

After :doc:`importing free recall data</guide/import>`, we have a DataFrame with
a row for each study event and a row for each recall event. Next, we need to
score the data by matching study events with recall events.

Scoring list recall
-------------------

First, let's create a simple sample dataset with two lists:

.. ipython::

    In [5]: import pandas as pd

    In [6]: data = pd.DataFrame(
       ...:     {'subject': [1, 1, 1, 1, 1, 1,
       ...:                  1, 1, 1, 1, 1, 1],
       ...:      'list': [1, 1, 1, 1, 1, 1,
       ...:               2, 2, 2, 2, 2, 2],
       ...:      'trial_type': ['study', 'study', 'study',
       ...:                     'recall', 'recall', 'recall',
       ...:                     'study', 'study', 'study',
       ...:                     'recall', 'recall', 'recall'],
       ...:      'position': [1, 2, 3, 1, 2, 3,
       ...:                   1, 2, 3, 1, 2, 3],
       ...:      'item': ['absence', 'hollow', 'pupil',
       ...:               'pupil', 'absence', 'empty',
       ...:               'fountain', 'piano', 'pillow',
       ...:               'pillow', 'fountain', 'pillow']})

    In [7]: data

Next, we'll merge together the study and recall events by matching up
corresponding events:

.. ipython::

    In [7]: from psifr import fr

    In [8]: study = data.query('trial_type == "study"').copy()

    In [9]: recall = data.query('trial_type == "recall"').copy()

    In [10]: merged = fr.merge_lists(study, recall)

    In [11]: merged

For each item, there is one row for each unique combination of input and
output position. For example, if an item is presented once in the list, but
is recalled multiple times, there is one row for each of the recall attempts.
Repeated recalls are indicated by the `repeat` column, which is greater than
zero for recalls of an item after the first. Unique study events are indicated
by the `study` column; this excludes intrusions and repeated recalls.

Items that were not recalled have the `recall` column set to `False`. Because
they were not recalled, they have no defined output position, so `output` is
set to `NaN`. Finally, intrusions have an output position but no input position
because they did not appear in the list. There is an `intrusion` field for
convenience to label these recall attempts.

:py:func:`~psifr.fr.merge_lists` can also handle additional attributes beyond
the standard ones, such as codes indicating stimulus category or list condition.
See the :py:func:`~psifr.fr.merge_lists` documentation for details.

Filtering and sorting
---------------------

Now that we have a merged `DataFrame`, we can use `pandas` methods to quickly
get different views of the data. For some analyses, we may want to organize in
terms of the study list by removing repeats and intrusions. Because our data
are in a `DataFrame`, we can use the `DataFrame.query` method:

.. ipython::

    In [9]: merged.query('study')

Alternatively, we may also want to get just the recall events, sorted by
output position instead of input position:

.. ipython::

    In [11]: merged.query('recall').sort_values(['list', 'output'])

Note that we first sort by list, then output position, to keep the
lists together.
