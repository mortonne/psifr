# Contributing to Psifr

Whether you are a novice or experienced software developer, 
all contributions and suggestions are welcome!

## Getting Started

If you are looking to contribute to the *Psifr* codebase, the 
best place to start is the [GitHub "issues" tab](https://github.com/mortonne/psifr/issues). 
This is also a great place for filing bug reports and making suggestions for 
ways in which the code and documentation can be improved.

## Filing Issues

If you notice a bug in the code or documentation, or have suggestions for 
how we can improve either, feel free to create an issue on the 
[GitHub "issues" tab](https://github.com/mortonne/psifr/issues) using 
[GitHub's "issue" form](https://github.com/mortonne/psifr/issues/new). 
The form contains some questions to help clarify the issue.

## Contributing to the Codebase

The code is hosted on [GitHub](https://www.github.com/mortonne/psifr), so 
you will need to use [Git](https://git-scm.com/) to clone the project and 
make changes to the codebase. Once you have obtained a copy of the code, 
you should create a development environment using Conda or virtualenv 
that is separate from your existing Python environment so that you can make 
and test changes without compromising your own work environment. Create a
branch within your clone of the project and make all changes there. To
install code locally for development, clone the project, checkout your
branch, and run `pip install -e .` from the main project directory. That
will install Psifr in editable mode so changes to the codebase will be
reflected the next time you import (or reload) Psifr.

Before submitting your changes for review, make sure to check that your 
changes do not break any tests. You can run the test suite by installing
[pytest](https://docs.pytest.org/en/stable/), then running `pytest` from
the main project directory.

Once your changes are ready to be submitted, push your changes 
to GitHub and then create a pull request. Your code will be reviewed, and
you may be asked to make changes before your code is approved to be
merged into the master branch. 
