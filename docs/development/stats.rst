==========
Statistics
==========

Analyses in Psifr are designed to operate on complete datasets that are formatted as data frames. Most analyses have a corresponding low-level statistics function that operates on data for an individual participant. Statistics functions take in study and recall information in list format. These low-level functions make it easier to test the core logic of an analysis, and also provide an alternate API for users who have data that aren't in table format.

List format data
~~~~~~~~~~~~~~~~

In list format, data are formatted as lists of lists instead of tables. Note that the number of recalls (and the number of study events) may vary from list to list, so individual lists may vary in length in this format.

For example, the items presented and recalled in two lists can be written as two lists of lists:

.. ipython:: python

    study_items = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    recall_items = [[6, 4, 5, 1, 2], [5, 6, 2, 3]]

The study lists indicate the presentation order of items, and the recall lists indicate the order in which items were recalled. Here, for simplicity, items are labeled by their serial position, but item identifiers can also be in other formats, such as strings indicating a presented word. In this example, the participant recalled 5 items in list 1, and 4 items in list 2.

Defining a new statistic
~~~~~~~~~~~~~~~~~~~~~~~~

As an example of a statistics function, let's write a very simple function to calculate recall percentage. To keep things simple, this function will assume that there are no repeats or intrusions.

.. ipython:: python

    def recall_probability(study_items, recall_items):
        recall_prob = []
        for study, recall in zip(study_items, recall_items):
            recall_prob.append(len(recall) / len(study))
        return recall_prob

This is a very simple function, but it shows the basic design of most statistics functions. It iterates over lists and calculates some statistic. It is simple to write and run tests for this type of function. Here, we'll use the example list data we set up before.

.. ipython:: python

    recall_probability(study_items, recall_items)

Note that many analyses are much more complicated, as they require examining the details of individual recall transitions or sequences of recalls. See :doc:`transitions</development/transitions>` for information on designing a transition-based statistic.

Statistics functions should be placed in :py:mod:`psifr.stats`.

Writing a unit test
~~~~~~~~~~~~~~~~~~~

Before submitting a new statistic to be considered for the main Psifr codebase, you must first write a unit test. A unit test is a simple program that defines input to a function, runs it, and raises an exception if the output is not as expected. Writing a unit test forces you to define the expected output of your function, and ensures that the program runs as expected.

Psifr uses `Pytest <pytest.org>`_ for unit tests. Writing a test just requires adding a function starting with :code:`test_` to one of the modules in the :code:`tests` directory. This function should run your statistics function on some test data. If the output is not as expected, raise an exception to indicate that the test has failed. In a terminal in the base directory of the project, run :code:`pytest` to run all tests. You can also specify individual tests to run by specifying them using :code:`tests/test_mystats.py::myfunction` syntax. Failed tests can be quickly debugged using PDB using :code:`pytest --pdb`. See the Pytest documentation for details.

A great way to write a complex analysis is to calculate the expected result by hand, write a corresponding unit test, and only then work on actually writing the function. When the unit test passes, you will know that at least your test cases are being computed correctly. This process is called test-driven development, and it was used to write much of Psifr.

Automated testing
~~~~~~~~~~~~~~~~~

The Psifr project is set up to automatically run all unit tests defined in the :code:`tests` directory every time a commit is pushed to GitHub. Test results are reported in a badge on the project's GitHub page, which indicates whether tests are currently passing.
