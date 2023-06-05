# Contributing to Psifr

Whether you are a novice or experienced software developer, 
all contributions and suggestions are welcome! This file goes over the basic mechanics of contributing changes. If you want to contribute to the codebase, please also read the [Contributor Guide](https://psifr.readthedocs.io/en/latest/development/overview.html), which explains how the codebase is organized and walks through how to add a new analysis function.

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
make changes to the codebase. 

If you do not have edit access to the original repository (mortonne/psifr), 
then first make a fork of the repository. Next, create a branch (either within 
the original repository or your fork) with a name indicating what you'll be 
working on. Your changes from the main branch should all be committed to 
this branch.

Once you have created a branch on GitHub (and a fork if necessary), 
you should create a development environment using Conda or virtualenv 
that is separate from your existing Python environment so that you can make 
and test changes without compromising your own work environment. To
install code locally for development, clone the project, checkout your
branch, and run `pip install -e .` from the main project directory. That
will install Psifr in editable mode so changes to the codebase will be
reflected the next time you import (or reload) Psifr.

Unit tests will be run automatically when you open a pull request. 
You can also run the test suite manually on your machine by installing
[pytest](https://docs.pytest.org/en/stable/), then running `pytest` from
the main project directory.

Please run [black](https://black.readthedocs.io/en/stable/) on any modules 
you edit so that your code will have standard formatting. For example, run 
`black -S src/psifr/fr.py` to reformat the `fr.py` module. The `-S` flag
indicates that strings should be left alone, rather than being reformatted
to use double quotes.

When making commits to your branch, please format your commit messages
following the
[conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) 
standard. This make it easier to see what kinds of changes are being
proposed (e.g., `feat` for an added feature or `fix` for a proposed fix).

Once your changes are ready to be submitted, push your changes 
to GitHub and then create a pull request from your branch. Your code will 
be reviewed, and you may be asked to make changes before your code is 
approved to be merged into the master branch. 
