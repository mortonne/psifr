---
title: 'Psifr: Analysis and visualization of free recall data'
tags:
  - Python
  - psychology
  - memory
  - research
authors:
  - name: Neal W Morton
    orcid: 0000-0002-2631-2710
    affiliation: 1
affiliations:
 - name: Research Fellow, Center for Learning and Memory, The University of Texas at Austin
   index: 1
date: 24 August 2020
bibliography: paper.bib
---

# Summary

Research on human memory has been strongly influenced by data from free 
recall experiments, wherein participants study a list of items (such as 
words) and then freely recall them in any order they wish [@Murdock:1962]. 
Free recall provides an extremely rich dataset that not only reflects 
which items were recalled but also the order in which they were recalled. 
However, analysis of free recall data is difficult, as many different influences
on recall must be taken into account. 
For example, one influential analysis, conditional response probability 
as a function of lag (lag-CRP), has been used to measure the tendency of 
participants to successively recall items that were originally presented 
near to each other in time [@Kahana:1996].
This analysis requires taking into account the items that are still
available for recall at each transition between recalled items. 
This analysis may need to be made conditional on other factors, such as
the category of the items being recalled [@Polyn:2011; @Morton:2017], 
thus complicating the analysis further.

`Psifr` (pronounced like "cypher") was developed to consolidate a number 
of free recall analysis methods (often implemented in MATLAB) within a flexible 
Python package. 
The `Psifr` package includes core utilities that simplify
and standardize a number of different analyses of recall sequences,
including analyses focused on serial position [@Murdock:1962],
temporal order [@Kahana:1996; @Polyn:2011], 
stimulus category [@Polyn:2009; @Morton:2016], and the semantic meaning 
of presented items [@Howard:2002]. 
The core utilities are also designed to facilitate implementation of 
extensions to tailor analyses for specific experiments.

![Example of a serial position curve showing the probability of recalling 
an item based on its position in the list. Plots may be flexibly divided by 
condition using grouping semantics supported by Seaborn. In this case,
different list types (mixed-category or pure-category) are plotted as separate
curves.\label{fig:spc}](spc_list_type.pdf)

# Statement of Need

Existing packages for analysis of free recall data include `EMBAM`
and `Quail`. 
[EMBAM](https://github.com/vucml/EMBAM), which is a fork of the 
[Behavioral Toolbox](http://memory.psych.upenn.edu/Behavioral_toolbox), 
is implemented in MATLAB, making it difficult to use with the extensive 
data science ecosystem in Python. 
The [pybeh](https://github.com/pennmem/pybeh) package is a Python port 
of `EMBAM` written using `numpy`.
As it is a fairly direct port of EMBAM, `pybeh` does not make use of some of 
the advantages  of Python, such as the advanced data science packages of 
`Pandas` and `Seaborn` [@Reback:2020; @Waskom:2020].
`Quail`, a Python package, provides some similar functionality to `Psifr`,
including analysis of recall order [@Heusser:2017]. 
However, while `Quail` uses a custom data structure to store free 
recall sequences, `Psifr` uses `Pandas` `DataFrame` objects. 
This design makes it possible for the user to make full use 
of the split-apply-combine operations of `Pandas` to quickly run complex analyses.

![Serial position curve split by list type, with a separate panel for each
participant in an experiment.\label{fig:spc_subject}](spc_subject.pdf)

Similarly, `Psifr` makes available the full power of the `Seaborn` 
visualization package to provide expressive visualization capabilities. 
The plotting functions in `Psifr` allow the user to easily view analysis 
results in different ways; for example, an analysis of recall by serial 
position can be visualized either as a single plot with error bars 
(\autoref{fig:spc}) or as a grid of individual plots for each 
participant in the experiment (\autoref{fig:spc_subject}).
`Psifr` also supports creation of raster plots 
(\autoref{fig:raster}), a method for visualizing whole 
free recall datasets to facilitate quick discovery of patterns in
the order of recalls [@Romani:2016].

![Raster plot displaying the order of every recall for one participant.
Each marker indicates one recall, and the color of the marker reflects
the serial position of the recalled item.
\label{fig:raster}](raster.pdf)

`Psifr` was designed to be used by memory researchers and students.
It is currently being used in two ongoing projects that require advanced
analysis and visualization. 
The interface is designed to simplify common tasks while also allowing 
for substantial customization to facilitate analysis of specific episodic
memory experiments.
Advanced visualization further helps to support better understanding of 
complex datasets. 
The source code for `Psifr` has  been archived to Zenodo with the linked DOI: TBD.

# Acknowledgements

Thanks to Sean Polyn for helpful discussions.
`Psifr` is inspired by functions initially developed for `EMBAM`,
which was developed by Sean Polyn, Neal Morton, Richard Lawrence,
Joshua McCluey, Karl Healey, Lynn Lohnas, and Jin Jeon. 
`Psifr` also takes inspiration from the `Quail` package developed 
by Andrew Heusser, Paxton Fitzpatrick, Campbell Field, Kirsten Ziman, 
and Jeremy Manning.

# References