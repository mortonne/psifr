
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

Comparing conditions
====================

When analyzing a dataset, it's often important to compare different
experimental conditions. Psifr is built on the Pandas :py:class:`~pandas.DataFrame`, which
has powerful ways of splitting data and applying operations to it.
This makes it possible to analyze and plot different conditions using
very little code.

.. _custom-columns:

Working with custom columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, load some sample data and create a merged DataFrame:

.. ipython:: python

    from psifr import fr
    df = fr.sample_data('Morton2013')
    data = fr.merge_free_recall(
        df, study_keys=['category'], list_keys=['list_type']
    )
    data.head()

The :py:func:`~psifr.fr.merge_free_recall` function only includes columns from the
raw data if they are one of the standard columns or if they've explictly been
included using :code:`study_keys`, :code:`recall_keys`, or :code:`list_keys`.
:code:`list_keys` apply to all events in a list, while :code:`study_keys` and
:code:`recall_keys` are relevant only for study and recall events, respectively.

We've included a list key here, to indicate that the :code:`list_type`
field should be included for all study and recall events in each list, even
intrusions. The :code:`category` field will be included for all study events
and all valid recalls. Intrusions will have an undefined category.

Analysis by condition
~~~~~~~~~~~~~~~~~~~~~

Now we can run any analysis separately for the different conditions. We'll
use the serial position curve analysis as an example.

.. ipython:: python

    spc = data.set_index('list_type').groupby('list_type').apply(fr.spc)
    spc.head()

The call to `set_index` before `groupby` avoids a deprecation warning 
(Pandas 2.2 changed the behavior of `apply` after `groupby`).

The :code:`spc` DataFrame has separate groups with the results for each
:code:`list_type`.

.. warning::

    When using :code:`groupby` with order-based analyses like
    :py:func:`~psifr.fr.lag_crp`, make sure all recalls in all recall
    sequences for a given list have the same label. Otherwise, you will
    be breaking up recall sequences, which could result in an invalid
    analysis.

Plotting by condition
~~~~~~~~~~~~~~~~~~~~~

We can then plot a separate curve for each condition. All plotting functions
take optional :code:`hue`, :code:`col`, :code:`col_wrap`, and :code:`row`
inputs that can be used to divide up data when plotting.
Most inputs to :py:func:`seaborn.relplot` are supported.

For example, we can plot two curves for the different list types:

.. ipython:: python

    @savefig spc_list_type.svg
    g = fr.plot_spc(spc, hue='list_type').add_legend()

We can also plot the curves in different axes using the :code:`col` option:

.. ipython:: python

    @savefig spc_list_type_col.svg
    g = fr.plot_spc(spc, col='list_type')

We can also plot all combinations of two conditions:

.. ipython:: python

    spc_split = data.groupby(['list_type', 'category']).apply(fr.spc)
    @savefig spc_split.svg
    g = fr.plot_spc(spc_split, col='list_type', row='category')

Plotting by subject
~~~~~~~~~~~~~~~~~~~

All analyses can be plotted separately by subject. A nice way to do this is
using the :code:`col` and :code:`col_wrap` optional inputs, to make a grid
of plots with 6 columns per row:

.. ipython:: python

    @savefig spc_subject.svg
    g = fr.plot_spc(
        spc, hue='list_type', col='subject', col_wrap=6, height=2
    ).add_legend()
