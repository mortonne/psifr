# Psifr
[![PyPI version](https://badge.fury.io/py/psifr.svg)](https://badge.fury.io/py/psifr)
[![Documentation Status](https://readthedocs.org/projects/psifr/badge/?version=latest)](https://psifr.readthedocs.io/en/latest/?badge=latest)
[![Pytest](https://github.com/mortonne/psifr/actions/workflows/pytest.yml/badge.svg)](https://github.com/mortonne/psifr/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/mortonne/psifr/branch/master/graph/badge.svg)](https://codecov.io/gh/mortonne/psifr)
[![status](https://joss.theoj.org/papers/712d4452e465229d61d0e281d3d6f299/status.svg)](https://joss.theoj.org/papers/712d4452e465229d61d0e281d3d6f299)
[![DOI](https://zenodo.org/badge/248593723.svg)](https://zenodo.org/badge/latestdoi/248593723)

Advanced analysis and visualization of free recall data in Python.

Features:
* A large library of advanced analyses, tested against published benchmarks
* Flexible analysis customization and plotting
* Tools for exploratory analysis of large datasets
* Extensive automated testing to ensure analysis correctness
* Based around a simple and flexible table-based data format
* Comprehensive [documentation](https://psifr.readthedocs.io/en/stable/api/overview.html) and [user guide](https://psifr.readthedocs.io/en/stable/guide/overview.html)

The name Psifr is pronounced "cipher". It's taken from Psi, in reference to the field of psychology, and FR for free recall.

<div align="center">
  <div style="max-width:500px; margin:0 20px;">
    <img src="https://github.com/mortonne/psifr/blob/master/images/raster.png" alt="free recall visualization" width="500px">
    <div style="text-align:left; padding:10px 0;">
      Raster plot showing each recall in a free recall experiment. Color indicates serial position; yellow items were presented late in the list, while purple items were presented at the beginning. Magenta squares indicate intrusions of items that were not presented during the study list.
    </div>
  </div>
</div>

## Citation

If you use Psifr, please help support open-source scientific software by citing it in your publications.

Morton, N. W., (2020). 
Psifr: Analysis and visualization of free recall data. 
Journal of Open Source Software, 5(54), 2669, https://doi.org/10.21105/joss.02669

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

## Data format

Psifr expects data to be in a simple standard format. For example, if subject 1 studied a list of the words "absence", "hollow", "pupil", then recalled "pupil", "absence", the data would be represented in a spreadsheet like this:

| subject | list | trial_type | position | item    |
| ------: | ---: | :--------- | -------: | :------ |
|       1 |    1 | study      |        1 | absence |
|       1 |    1 | study      |        2 | hollow  |
|       1 |    1 | study      |        3 | pupil   |
|       1 |    1 | recall     |        1 | pupil   |
|       1 |    1 | recall     |        2 | absence |

See [importing data](https://psifr.readthedocs.io/en/latest/guide/import.html) for details.

## Related projects

### EMBAM
Some of the analyses supported by Psifr are based on analyses implemented in the Matlab toolbox [EMBAM](https://github.com/vucml/EMBAM).

### pybeh
[pybeh](https://github.com/pennmem/pybeh) is a direct Python port of EMBAM that supports a wide range of analyses.

### Quail
[Quail](https://github.com/ContextLab/quail) runs automatic scoring of free recall data, supports calculation and plotting of some common free recall measures, and has tools for measuring the "memory fingerprint" of individuals.

## Publications using Psifr

See Google Scholar for a list of [publications using Psifr](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=15633316861794439696). 

## Contributing to Psifr

Contributions are welcome to suggest new features, add documentation, and identify bugs. See the [contributing guidelines](.github/CONTRIBUTING.md) for an overview. 
