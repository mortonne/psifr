
.. ipython:: python
   :suppress:

   import numpy as np
   import pandas as pd
   import matplotlib as mpl
   import matplotlib.pyplot as plt

   plt.style.use('default')
   mpl.rcParams['axes.labelsize'] = 'large'
   mpl.rcParams['savefig.bbox'] = 'tight'
   mpl.rcParams['savefig.pad_inches'] = 0.05

   pd.options.display.max_rows = 15

==================
Recall performance
==================

Serial position curve
~~~~~~~~~~~~~~~~~~~~~

First, load some sample data and create a merged DataFrame:

.. ipython::

    In [1]: from psifr import fr

    In [1]: df = fr.sample_data('Morton2013')

    In [1]: data = fr.merge_free_recall(df)

We can then calculate average recall for each serial position
using :py:func:`~psifr.fr.spc` and plot using :py:func:`~psifr.fr.plot_spc`.

.. ipython::

    In [1]: recall = fr.spc(data)

    @savefig spc.png
    In [1]: g = fr.plot_spc(recall)

Using the same plotting function, we can plot the curve for each
individual subject:

.. ipython::

    @savefig spc_indiv.png
    In [1]: g = fr.plot_spc(recall, col='subject', col_wrap=5)

Probability of Nth recall
~~~~~~~~~~~~~~~~~~~~~~~~~

We can also split up recalls, to test for example how likely participants
were to initiate recall with the last item on the list.

.. ipython::

    In [1]: prob = fr.pnr(data)

    In [1]: prob

This gives us the probability of recall by output position (:code:`'output'`)
and serial or input position (:code:`'input'`). This is a lot to look at all
at once, so it may be useful to plot just the first three output positions.
We can plot the curves using :py:func:`~psifr.fr.plot_spc`, which takes an
optional :code:`hue` input to specify a variable to use to split the data
into curves of different colors.

.. ipython::

    In [1]: pfr = prob.query('output <= 3')

    @savefig pnr.png
    In [1]: g = fr.plot_spc(pfr, hue='output').add_legend()
