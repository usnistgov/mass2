# MASS 2: Microcalorimeter Analysis Software Suite


<center>
<img src="mass2_logo_128.png" alt="Mass 2 logo">
</center>

## Overview of Mass 2

MASS is a Python software suite designed to analyze pulse records from high-resolution, cryogenic microcalorimeters. We use MASS with pulse records from x-ray and gamma-ray spectrometers, performing a sequence of analysis, or "calibration" steps to extract a high-precision estimate of the energy from each record.

With raw pulse records and MASS, you can:

* Analyze data from one or multiple detectors at one time.
* Analyze data from one or more raw pulse data files per detector.
* Analyze a fixed dataset taken in the past, or perform "online" analysis of a dataset still being acquired.
* Analyze data from time-division multiplexed (TDM) and microwave-multiplexed (µMUX) systems.
* Choose and apply data cuts.
* Compute and apply "optimal filters" of various types.
* Fix complex line shapes in an energy spectrum.
* Estimate and apply accurate functions for absolute-energy calibration.
* Win friends and influence people.

## Major concepts

Mass2 is built on a few core technologies:

1. [Pola.rs](https://pola.rs/), a high-performance modern dataframe library. Organizes the data structures.
2. [Numpy](https://numpy.org/). For numerical arrays and scientific computation.
3. [Scipy](https://scipy.org/). Additional algorithms for scientific computation.

Unlike Mass version 1, Mass 2 leverages existing libraries for data organization. This offers the advantage that documentation, bug fixes, and testing are provided by the larger open-source software community.
