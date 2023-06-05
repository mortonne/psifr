=================
Contributing code
=================

To contribute code to Psifr, first make a fork of the `main project <https://github.com/mortonne/psifr>`_. Next, create a branch with a name to indicate the feature you'll be working on. Commit all of your changes there, and follow the procedure below to add necessary functions to the Psifr API, document them, and add automated tests.

Once you have followed the procedure to the best of your ability, make a pull request to have your changes merged into the master branch of Psifr. Your code will be reviewed to determine whether it is a good fit for the project and whether it meets all of the project's standards. If your additions seem like a good fit, you may be asked to make additional changes before your contribution is incorporated into the Psifr codebase.

Developing new analyses usually involves a few steps:

* Write a function to take in information about study lists and recalls for one participant, and output statistics related to your measure. You should also write a unit test to make sure your statistic works as expected.
* Write a class inheriting from one of the Measure classes to set up a full analysis that will take table-formatted data and calculate your statistic for each participant.
* Add a function to the :py:mod:`psifr.fr` module to initialize your Measure, run it, and output a results table. Also add user documentation and a unit test for this function.
* Update the documentation, including both the relevant section of the user guide and the API reference documentation.

Read on for more details about this process.
