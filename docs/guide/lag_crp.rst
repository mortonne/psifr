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

Use :py:func:`~psifr.fr.plot_lag_crp` to display the results:

.. ipython::

   @savefig lag_crp.png
   In [1]: g = fr.plot_lag_crp(crp, height=5)
