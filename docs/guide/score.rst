Scoring data
============

After :doc:`importing free recall data</guide/import>`, we have a DataFrame with
a row for each study event and a row for each recall event. Next, we need to
score the data by matching study events with recall events.

Scoring list recall
-------------------

First, let's create a simple sample dataset with two lists. We can use
the :py:func:`~psifr.fr.table_from_lists` convenience function to create
a sample dataset with a given set of study lists and recalls:

.. ipython:: python

    from psifr import fr
    list_subject = [1, 1]
    study_lists = [['absence', 'hollow', 'pupil'], ['fountain', 'piano', 'pillow']]
    recall_lists = [['pupil', 'absence', 'empty'], ['pillow', 'pupil', 'pillow']]
    data = fr.table_from_lists(list_subject, study_lists, recall_lists)
    data

Next, we'll merge together the study and recall events by matching up
corresponding events using :py:func:`~psifr.fr.merge_free_recall`.
This scoring and merging step labels recall attempts
in terms of whether they were correct recalls, repeats, or intrusions. At the
same time, it also labels study events in terms of whether they were correctly
recalled, and, if so, at which output position they were recalled. Free-recall
analyses in Psifr are computed from data in this "merged" format.

.. ipython:: python

    merged = fr.merge_free_recall(data)
    merged

For each item, there is one row for each unique combination of input and
output position. For example, if an item is presented once in the list, but
is recalled multiple times, there is one row for each of the recall attempts.
Repeated recalls are indicated by the :code:`repeat` column, which is greater than
zero for recalls of an item after the first. Unique study events are indicated
by the :code:`study` column; this excludes intrusions and repeated recalls.

Items that were not recalled have the :code:`recall` column set to :code:`False`. Because
they were not recalled, they have no defined output position, so :code:`output` is
set to :code:`NaN`. Finally, intrusions have an output position but no input position
because they did not appear in the list. There is an :code:`intrusion` field for
convenience to label these recall attempts. The :code:`prior_list` and :code:`prior_input`
fields give information about prior-list intrusions (PLIs) of items from prior
lists. The :code:`prior_list` field gives the list where the item appeared and
:code:`prior_input` indicates the position in which is was presented on that list.

:py:func:`~psifr.fr.merge_free_recall` can also handle additional attributes beyond
the standard ones, such as codes indicating stimulus category or list condition.
See :ref:`custom-columns` for details.

Filtering and sorting
---------------------

Now that we have a merged :py:class:`~pandas.DataFrame`, we can use Pandas methods to quickly
get different views of the data. For some analyses, we may want to organize in
terms of the study list by removing repeats and intrusions. Because our data
are in a :py:class:`~pandas.DataFrame`, we can use the :py:meth:`~pandas.DataFrame.query` method:

.. ipython:: python

    merged.query('study')

Alternatively, we may also want to get just the recall events, sorted by
output position instead of input position:

.. ipython:: python

    merged.query('recall').sort_values(['list', 'output'])

Note that we first sort by list, then output position, to keep the
lists together.
