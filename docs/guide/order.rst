
.. ipython:: python
   :suppress:

   import numpy as np
   import pandas as pd
   import matplotlib as mpl
   import matplotlib.pyplot as plt

   plt.style.use('default')
   mpl.rcParams['axes.labelsize'] = 'large'
   mpl.rcParams['savefig.bbox'] = 'tight'
   mpl.rcParams['savefig.pad_inches'] = 0.1

   pd.options.display.max_rows = 15

============
Recall order
============

A key advantage of free recall is that it provides information not only about
what items are recalled, but also the order in which they are recalled. A
number of analyses have been developed to charactize different influences on
recall order, such as the temporal order in which the items were presented at
study, the category of the items themselves, or the semantic similarity between
pairs of items.

Each conditional response probability (CRP) analysis involves calculating the
probability of some type of transition event. For the lag-CRP analysis,
transition events of interest are the different lags between serial positions
of items recalled adjacent to one another. Similar analyses focus not on
the serial position in which items are presented, but the properties of the
items themselves. A semantic-CRP analysis calculates the probability of
transitions between items in different semantic relatedness bins. A special
case of this analysis is when item pairs are placed into one of two bins,
depending on whether they are in the same stimulus category or not. In Psifr,
this is referred to as a category-CRP analysis.

Lag-CRP
~~~~~~~

In all CRP analyses, transition probabilities are calculated conditional
on a given transition being available. For example, in a six-item list,
if the items 6, 1, and 4 have been recalled, then possible items that could
have been recalled next are 2, 3, or 5; therefore, possible lags at
that point in the recall sequence are -2, -1, or +1. The number of actual
transitions observed for each lag is divided by the number of times that
lag was possible, to obtain the CRP for each lag.

First, load some sample data and create a merged DataFrame:

.. ipython:: python

    from psifr import fr
    df = fr.sample_data('Morton2013')
    data = fr.merge_free_recall(df, study_keys=['category'])

Next, call :py:func:`~psifr.fr.lag_crp` to calculate conditional response
probability as a function of lag.

.. ipython:: python

    crp = fr.lag_crp(data)
    crp

The results show the count of times a given transition actually happened
in the observed recall sequences (:code:`actual`) and the number of times a
transition could have occurred (:code:`possible`). Finally, the :code:`prob` column
gives the estimated probability of a given transition occurring, calculated
by dividing the actual count by the possible count.

Use :py:func:`~psifr.fr.plot_lag_crp` to display the results:

.. ipython:: python

   @savefig lag_crp.svg
   g = fr.plot_lag_crp(crp)

The peaks at small lags (e.g., +1 and -1) indicate that the recall sequences
show evidence of a temporal contiguity effect; that is, items presented near
to one another in the list are more likely to be recalled successively than
items that are distant from one another in the list.

Lag rank
~~~~~~~~

We can summarize the tendency to group together nearby items using a lag
rank analysis. For each recall, this determines the absolute lag of all
remaining items available for recall and then calculates their percentile
rank. Then the rank of the actual transition made is taken, scaled to vary
between 0 (furthest item chosen) and 1 (nearest item chosen). Chance
clustering will be 0.5; clustering above that value is evidence of a
temporal contiguity effect.

.. ipython:: python

    ranks = fr.lag_rank(data)
    ranks
    ranks.agg(['mean', 'sem'])

Category CRP
~~~~~~~~~~~~

If there are multiple categories or conditions of trials in a list, we
can test whether participants tend to successively recall items from the
same category. The category-CRP estimates the probability of successively
recalling two items from the same category.

.. ipython:: python

    cat_crp = fr.category_crp(data, category_key='category')
    cat_crp
    cat_crp[['prob']].agg(['mean', 'sem'])

The expected probability due to chance depends on the number of
categories in the list. In this case, there are three categories, so
a category CRP of 0.33 would be predicted if recalls were sampled
randomly from the list.

Restricting analysis to specific items
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you may want to focus an analysis on a subset of recalls. For
example, in order to exclude the period of high clustering commonly
observed at the start of recall, lag-CRP analyses are sometimes
restricted to transitions after the first three output positions.

You can restrict the recalls included in a transition analysis using
the optional :code:`item_query` argument. This is built on the Pandas
query/eval system, which makes it possible to select rows of a
:code:`DataFrame` using a query string. This string can refer to any
column in the data. Any items for which the expression evaluates to
:code:`True` will be included in the analysis.

For example, we can use the :code:`item_query` argument to exclude any
items recalled in the first three output positions from analysis. Note
that, because non-recalled items have no output position, we need to
include them explicitly using :code:`output > 3 or not recall`.

.. ipython:: python

    crp_op3 = fr.lag_crp(data, item_query='output > 3 or not recall')
    @savefig lag_crp_op3.svg
    g = fr.plot_lag_crp(crp_op3)
