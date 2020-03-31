# Psifr
[![Documentation Status](https://readthedocs.org/projects/psifr/badge/?version=latest)](https://psifr.readthedocs.io/en/latest/?badge=latest)

Advanced analysis and visualization of free recall data in Python.

In free recall, participants study a list of items and then name all of the items they can remember in any order they choose. Many sophisticated analyses have been developed to analyze data from free recall experiments, but these analyses are often complicated and difficult to implement. 

Psifr leverages the Pandas data analysis package to make precise and flexible analysis of free recall data faster and easier. The name Psifr is pronounced "cipher". It's taken from Psi, in reference to the field of psychology, and FR for free recall.

<div align="center">
  <div style="max-width:500px; margin:0 20px;">
    <img src="https://github.com/mortonne/psifr/blob/master/images/raster.png" alt="free recall visualization" width="500px">
    <div style="text-align:left; padding:10px 0;">
      Raster plot showing each recall in a free recall experiment. Color indicates serial position; yellow items were presented late in the list, while purple items were presented at the beginning. Magenta squares indicate intrusions of items that were not presented during the study list.
    </div>
  </div>
</div>

## Quickstart

See the Jupyter notebooks for demonstrations of reading and analyzing a sample free recall dataset:
* [Recall performance](https://github.com/mortonne/psifr-notebooks/blob/master/demo_recall.ipynb)
* [Temporal clustering](https://github.com/mortonne/psifr-notebooks/blob/master/demo_clustering.ipynb)

See the [documentation](https://psifr.readthedocs.io/en/latest/?badge=latest) for more details.

## Installation

First get a copy of the code from GitHub:

```bash
git clone git@github.com:mortonne/psifr.git
```

Then install:

```bash
cd psifr
python setup.py install
```

## Importing data

Generally the best way to get your data into shape for analysis in Psifr is to create a CSV file with one row for each event in the experiment, including study events (i.e., item presentations) and all recall attempts (including repeats and intrusions). Your spreadsheet should have the following columns:
* `subject` - some code that is unique for each subject. May be text or a number.
* `list` - list number.
* `item` - item identifier (text or numeric code). Must be unique to a specific item.
* `position` - for study events, codes position within the list (i.e., serial position), starting from 1 for the first item in the list; for recall events, codes output position (i.e., recall attempt number), starting from 1 for the first recall attempt.
* `trial_type` - indicates whether each row corresponds to 'study' or 'recall'.

These are just the basic necessary fields. You can also include columns for any other information you need to know about each trial. See `data/cfr_raw_data.csv` for an example.

If you have data in the standard [EMBAM](https://github.com/vucml/EMBAM) format, use `scripts/frdata2table.m` to convert your data struct to a table with standard format. Then use the Matlab function `writetable` to write a CSV file which can then be read into Python for analysis.

See the [recall performance](https://github.com/mortonne/psifr-notebooks/blob/master/demo_recall.ipynb) for an example of reading in data from a CSV file for analysis.

## Related projects

### EMBAM
Analyses supported by Psifr are based on analyses implemented in the Matlab toolbox [EMBAM](https://github.com/vucml/EMBAM).

### pybeh
[pybeh](https://github.com/pennmem/pybeh) is a direct Python port of EMBAM that supports a wide range of analyses.

### Quail
[Quail](https://github.com/ContextLab/quail) runs automatic scoring of free recall data, supports calculation and plotting of some common free recall measures, and has tools for measuring the "memory fingerprint" of individuals.
