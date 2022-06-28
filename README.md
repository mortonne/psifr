# Psifr
[![PyPI version](https://badge.fury.io/py/psifr.svg)](https://badge.fury.io/py/psifr)
[![Documentation Status](https://readthedocs.org/projects/psifr/badge/?version=latest)](https://psifr.readthedocs.io/en/latest/?badge=latest)
[![Pytest](https://github.com/mortonne/psifr/actions/workflows/pytest.yml/badge.svg)](https://github.com/mortonne/psifr/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/mortonne/psifr/branch/master/graph/badge.svg)](https://codecov.io/gh/mortonne/psifr)
[![status](https://joss.theoj.org/papers/712d4452e465229d61d0e281d3d6f299/status.svg)](https://joss.theoj.org/papers/712d4452e465229d61d0e281d3d6f299)
[![DOI](https://zenodo.org/badge/248593723.svg)](https://zenodo.org/badge/latestdoi/248593723)

Advanced analysis and visualization of free recall data in Python.

Psifr has tools to visualize recall sequences, to make it easier to see general trends in the data. Psifr also provides targeted analyses to examine factors that may affect recall, such as recency, primacy, temporal contiguity, stimulus category, and semantic relatedness. Analyses are customizable, allowing flexible filtering of included data to help answer precise questions. Extensive automated testing is used to ensure the consistency and correctness of analysis results. 

The name Psifr is pronounced "cipher". It's taken from Psi, in reference to the field of psychology, and FR for free recall.

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
pip install git+https://github.com/mortonne/psifr
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

Generally the best way to get your data into shape for analysis in Psifr is to create a CSV (or TSV) file with one row for each event in the experiment, including study events (i.e., item presentations) and all recall attempts (including repeats and intrusions).
See [importing data](https://psifr.readthedocs.io/en/latest/guide/import.html) for details.

A number of archival free recall datasets are available in the Matlab-based EMBAM format.
Data archives for a number of studies are available from the [UPenn](https://memory.psych.upenn.edu/Data_Archive) and [Vanderbilt](https://memory.psy.vanderbilt.edu/w/index.php/Publications) memory labs.
If you have data in [EMBAM](https://github.com/vucml/EMBAM) format, use `matlab/frdata2table.m` to convert your data struct to a table with standard format.
Then use the Matlab function `writetable` to write a CSV file which can then be read into Python for analysis.

## Citation

If you use Psifr, please cite the paper:

Morton, N. W., (2020). 
Psifr: Analysis and visualization of free recall data. 
Journal of Open Source Software, 5(54), 2669, https://doi.org/10.21105/joss.02669

## Publications using Psifr

Hong, B., Barense, M. D., Pace-Tonna, C. A. & Mack, M. L. (2022). 
Emphasizing associations from encoding affects free recall at retrieval. 
Proceedings of the Annual Meeting of the Cognitive Science Society 453–460.
https://escholarship.org/uc/item/2gw1s36q

## Related projects

### EMBAM
Analyses supported by Psifr are based on analyses implemented in the Matlab toolbox [EMBAM](https://github.com/vucml/EMBAM).

### pybeh
[pybeh](https://github.com/pennmem/pybeh) is a direct Python port of EMBAM that supports a wide range of analyses.

### Quail
[Quail](https://github.com/ContextLab/quail) runs automatic scoring of free recall data, supports calculation and plotting of some common free recall measures, and has tools for measuring the "memory fingerprint" of individuals.

## Contributing to Psifr

Contributions are welcome to suggest new features, add documentation, and identify bugs. See the [contributing guidelines](.github/CONTRIBUTING.md) for an overview. 
