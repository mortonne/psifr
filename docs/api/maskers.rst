=======
Maskers
=======

.. currentmodule:: psifr.maskers

The :py:mod:`~psifr.maskers` module defines maskers, which are tools for iterating over recall sequences while screening out excluded output positions or transitions. The maskers also output a running list of which items that meet inclusion criteria that have not yet been recalled.

Iterating over output positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    outputs_masker

Iterating over transitions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    transitions_masker
    sequences_masker
    windows_masker
