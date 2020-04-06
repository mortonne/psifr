Lag-CRP analysis
================

First, load some sample data and create a merged DataFrame:

.. ipython::

    In [8]: from pkg_resources import resource_filename

    In [8]: import pandas as pd

    In [9]: from psifr import fr

    In [10]: data_file = resource_filename('psifr', 'data/Morton2013.csv')

    In [11]: df = pd.read_csv(data_file)

    In [13]: study = df.query('trial_type == "study"').copy()

    In [14]: recall = df.query('trial_type == "recall"').copy()

    In [15]: data = fr.merge_lists(study, recall)

Next, call :py:func:`~psifr.fr.lag_crp` to calculate conditional response
probability as a function of lag.

.. ipython::

    In [17]: crp = fr.lag_crp(data)

    In [17]: crp

The results can be easily plotted using Seaborn:

.. ipython::

   In [1]: import seaborn as sns

   In [2]: neg = crp.query('-5 <= lag < 0').reset_index()

   In [3]: pos = crp.query('5 >= lag > 0').reset_index()

   In [2]: sns.lineplot(x='lag', y='prob', color='b', data=neg)

   @savefig lag_crp.png
   In [2]: sns.lineplot(x='lag', y='prob', color='b', data=pos)
