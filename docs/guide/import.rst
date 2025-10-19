Importing data
==============

In Psifr, free recall data are imported in the form of a "long" format
table. Each row corresponds to one *study* or *recall* event. Study
events include any time an item was presented to the participant.
Recall events correspond to any recall attempt; this includes *repeats*
of items there were already recalled and *intrusions* of items that
were not present in the study list.

This type of information is well represented in a CSV spreadsheet,
though any file format supported by pandas may be used for input. To
import from a CSV, use :py:func:`pandas.read_csv`. For example:

.. code-block:: python

    import pandas as pd
    data = pd.read_csv("my_data.csv")

Trial information
-----------------

The basic information that must be included for each event is the
following:

subject
    Some code (numeric or string) indicating individual participants.
    Must be unique for a given experiment. For example, ``sub-101``.

list
    Numeric code indicating individual lists. Must be unique within
    subject.

trial_type
    String indicating whether each event is a ``study`` event or a
    ``recall`` event.

position
    Integer indicating position within a given phase of the list. For
    ``study`` events, this corresponds to *input position* (also
    referred to as *serial position*). For ``recall`` events, this
    corresponds to *output position*.

item
    Individual thing being recalled, such as a word. May be specified
    with text (e.g., ``pumpkin``, ``Jack Nicholson``) or a numeric code
    (``682``, ``121``). Either way, the text or number must be unique
    to that item. Text is easier to read and does not require any
    additional information for interpretation and is therefore
    preferred if available.

Example
-------

.. csv-table:: Sample data
    :header: "subject", "list", "trial_type", "position", "item"
    :widths: 8, 8, 8, 8, 8

    1, 1, "study", 1, "absence"
    1, 1, "study", 2, "hollow"
    1, 1, "study", 3, "pupil"
    1, 1, "recall", 1, "pupil"
    1, 1, "recall", 2, "absence"

Additional information
----------------------

Additional fields may be included in the data to indicate other
aspects of the experiment, such as presentation time, stimulus
category, experimental session, distraction length, etc. All of
these fields can then be used for analysis in Psifr.

Converting existing datasets
----------------------------

A number of archival free recall datasets are available in the Matlab-based EMBAM format.
Data archives for a number of studies are available from the [UPenn](https://memory.psych.upenn.edu/Data_Archive) and [Vanderbilt](https://memory.psy.vanderbilt.edu/w/index.php/Publications) memory labs.
If you have data in [EMBAM](https://github.com/vucml/EMBAM) format, use `matlab/frdata2table.m` to convert your data struct to a table with standard format.
Then use the Matlab function `writetable` to write a CSV file which can then be read into Python for analysis.
