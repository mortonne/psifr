# Psifr
[![PyPI version](https://badge.fury.io/py/psifr.svg)](https://badge.fury.io/py/psifr)
[![Documentation Status](https://readthedocs.org/projects/psifr/badge/?version=latest)](https://psifr.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/mortonne/psifr.svg?branch=master)](https://travis-ci.com/mortonne/psifr)
[![codecov](https://codecov.io/gh/mortonne/psifr/branch/master/graph/badge.svg)](https://codecov.io/gh/mortonne/psifr)
[![status](https://joss.theoj.org/papers/712d4452e465229d61d0e281d3d6f299/status.svg)](https://joss.theoj.org/papers/712d4452e465229d61d0e281d3d6f299)
[![DOI](https://zenodo.org/badge/248593723.svg)](https://zenodo.org/badge/latestdoi/248593723)

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

## Installation

You can install the latest stable version of Psifr using pip:

```bash
pip install psifr
```

You can also install the development version directly from the code
repository on GitHub:

```bash
pip install git+git://github.com/mortonne/psifr
```

## Quickstart

To plot a serial position curve for a sample dataset:

```python
from psifr import fr
df = fr.sample_data('Morton2013')
data = fr.merge_free_recall(df)
recall = fr.spc(data)
g = fr.plot_spc(recall)
```

See the [user guide](https://psifr.readthedocs.io/en/latest/guide/overview.html) for detailed documentation on importing and analyzing free recall datasets.

Also see the Jupyter notebooks for more analysis examples:
* [Recall performance](https://github.com/mortonne/psifr-notebooks/blob/master/demo_recall.ipynb)
* [Temporal clustering](https://github.com/mortonne/psifr-notebooks/blob/master/demo_lag_crp.ipynb)

## Importing data

Generally the best way to get your data into shape for analysis in Psifr is to create a CSV file with one row for each event in the experiment, including study events (i.e., item presentations) and all recall attempts (including repeats and intrusions). See [importing data](https://psifr.readthedocs.io/en/latest/guide/import.html) for details.

If you have data in the standard [EMBAM](https://github.com/vucml/EMBAM) format, use `scripts/frdata2table.m` to convert your data struct to a table with standard format. Then use the Matlab function `writetable` to write a CSV file which can then be read into Python for analysis.

## Related projects

### EMBAM
Analyses supported by Psifr are based on analyses implemented in the Matlab toolbox [EMBAM](https://github.com/vucml/EMBAM).

### pybeh
[pybeh](https://github.com/pennmem/pybeh) is a direct Python port of EMBAM that supports a wide range of analyses.

### Quail
[Quail](https://github.com/ContextLab/quail) runs automatic scoring of free recall data, supports calculation and plotting of some common free recall measures, and has tools for measuring the "memory fingerprint" of individuals.

## Contributing to Psifr

Contributions are welcome to suggest new features, add documentation, and identify bugs. See the [contributing guidelines](.github/CONTRIBUTING.md) for an overview. 
