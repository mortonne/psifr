
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

Use :py:func:`~psifr.fr.plot_lag_crp` to display the results:

.. ipython:: python

   @savefig lag_crp.svg
   g = fr.plot_lag_crp(crp)

Lag rank
~~~~~~~~

We can summarize the tendency to group together nearby items using a lag
rank analysis. For each recall, this determines the absolute lag of all
remaining items available for recall then calculates their percentile
rank. Then the rank of the actual transition made is taken, scaled to vary
between 0 (furthest item chosen) and 1 (nearest item chosen). Chance
clustering will be 0.5.

.. ipython:: python

    ranks = fr.lag_rank(data)
    ranks
    ranks.agg(['mean', 'sem'])

Category CRP
~~~~~~~~~~~~

If there are multiple categories or conditions of trials in a list, we
can test whether participants tend to successively recall items from the
same category.

.. ipython:: python

    cat_crp = fr.category_crp(data, category_key='category')
    cat_crp
    cat_crp[['prob']].agg(['mean', 'sem'])
