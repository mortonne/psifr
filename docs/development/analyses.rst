========
Analyses
========

In Psifr, analyses take a full dataset in data frame format and calculate some statistic or statistics for each subject. Results are output in data frame format, which is flexible and allows for rich information to be output for each participant. For example, the probability of Nth recall analysis gives the conditional probability of recall of each serial position for each output position during recall.

Creating a Measure class
~~~~~~~~~~~~~~~~~~~~~~~~

Most analyses call a low-level statistics function that takes in list-format data. Here, we'll use the :code:`recall_probability` function we made in the :doc:`Statistics </development/stats>` section. To run this function separately for each subject in a dataset, we can create a Measure by inheriting from the :py:class:`~psifr.measures.TransitionMeasure` class. This will handle iterating over subjects, converting the data to list format, calculating the statistic, and outputting the data in a results table.

We just need to define an :code:`__init__` method to take in any user options and an :code:`analyze_subject` method to run the analysis for one subject.

Analysis method
^^^^^^^^^^^^^^^

The :code:`analyze_subject` method must take in :code:`subject`, :code:`pool`, and :code:`recall` inputs. The :code:`pool` and :code:`recall` inputs are dictionaries containing information from columns of the data frame, which have been converted into list format. Both :code:`pool` and :code:`recall` may contain three different types of information about study pools and recall sequences:

* :code:`items` indicate item identifiers. This is used to keep track of which items are available for recall and to match them up with recall sequences. Input position is commonly used for this.
* :code:`label` indicates item labels. This is a separate input from :code:`items` because sometimes an analysis requires other information about the items, such as stimulus category, which is not unique to that item.
* :code:`test` indicates values for testing whether a given item or transition between items should be included in the analysis. A separate test function must be supplied to check whether or not to include that transition or recall, based on its test value.

Note that these are just conventions; it's up to the individual analysis code to decide how (and whether) to use :code:`items`, :code:`label`, and :code:`test` inputs. Each of these entries is pulled from the data frame; the individual analysis determines which columns each entry corresponds to. For example, in lag-CRP analysis, both :code:`items` and :code:`label` are pulled from the :code:`input` column.

In general, :code:`items` inputs are required, but labels and tests are optional. In the case of :code:`recall_probability`, we just need to know how many study items there were in each list and how many items were recalled. Serial position works well in this case (though note that this code assumes that there are no repeated recalls or intrusions).

Measure initialization
^^^^^^^^^^^^^^^^^^^^^^

In our example, the initialization method doesn't require any user input. We'll just assume that input position (or serial position) is stored in the :code:`'input'` column. The :py:class:`~psifr.measures.TransitionMeasure` class has two required inputs: a column of the data table with identifiers for the pool of studied items, and a column with identifiers for recalled items. Here, we can just use serial position (the :code:`input` column in standard Psifr format) for both of these inputs.

Here, we just need to specify item identifiers, but other analyses may need additional information. For example, the :code:`category_crp` analysis uses :code:`input` to define item identifiers and a :code:`category` column to indicate stimulus category labels.

Analysis output
^^^^^^^^^^^^^^^

Putting everything together, we can define our Measure class:

.. ipython:: python

    import numpy as np
    import pandas as pd
    from psifr import measures

    def recall_probability(study_items, recall_items):
        recall_prob = []
        for study, recall in zip(study_items, recall_items):
            recall_prob.append(len(recall) / len(study))
        return recall_prob

    class Recall(measures.TransitionMeasure):
        def __init__(self):
            super().__init__('input', 'input')
        def analyze_subject(self, subject, pool, recall):
            stat = recall_probability(pool['items'], recall['items'])
            rec = pd.DataFrame({'recall': np.mean(stat)}, index=[subject])
            rec.index.name = 'subject'
            return rec

Note that we define a data frame that organizes the statistics for one subject, and includes subject as the index for the data frame. Analysis data frames may contain multiple indices, such as subject, input position, and output position, depending on the analysis.

Adding statistics and measures to Psifr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we were writing real code to be added to Psifr, we would add :code:`recall_probability` to the :py:mod:`psifr.stats` module, and :code:`Recall` to the :py:mod:`measures` module. We would also add a docstring to the :code:`recall_probability` function, following NumPy's standard `docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

As mentioned previously, we would also add a unit test of :code:`recall_probability` to the :code:`tests` directory. That test would ideally handle edge cases like repeats and intrusions. It would then show that our current implementation is flawed and needs to be modified to handle these edge cases correctly.

Adding an analysis to Psifr
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, create a high-level function to run the analysis. This function should take in a data frame in Psifr merged format as its first input.

.. ipython:: python

    def prec(data):
        measure = Recall()
        rec = measure.analyze(data)
        return rec

We can test it on some sample data:

.. ipython:: python

    from psifr import fr
    raw = fr.sample_data('Morton2013')
    data = fr.merge_free_recall(raw)
    prec(data)

If we were writing real code to be added to Psifr, we could add :code:`prec` to the :py:mod:`psifr.fr` module, thus making it available through the high-level :code:`fr` API. We would add a docstring for :code:`prec` describing the inputs and outputs in standard `NumPy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_. Ideally, at the end of this docstring, we'd include a doctest-compatible example of how to run an analysis and the expected output for that example. Finally, we would also add a unit test on some sample data in :code:`tests/test_fr.py`, thus adding a test of analysis functionality to Psifr's automated test suite.
