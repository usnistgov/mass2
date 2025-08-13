

# MASS: The Microcalorimeter Analysis Software System, version 2

<!-- Couldn't figure out how to get markdown to resize the drawing w/o going to full HTML -->
<p align="left">
    <img src="doc/mass2_logo_128.png", width="160" title="Microcal Analysis">
</p>


MASS is the work of physicists from NIST Boulder Labs and the University of Colorado, with substantial contributions from:

* [Joe Fowler](https://github.com/joefowler/), project director
* [Galen O'Neil](https://github.com/ggggggggg/), co-director
* [Dan Becker](https://github.com/danbek/)
* Young-Il Joe
* Jamie Titus
* Many collaborators, who have made many bug reports, bug fixes, and feature requests.

## Introduction

MASS is a software suite designed to analyze pulse records from high-resolution, cryogenic microcalorimeters. We use MASS with pulse records from x-ray and gamma-ray spectrometers, performing a sequence of analysis, or "calibration" steps to extract a high-precision estimate of the energy from each record.

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

As of this writing (August 8, 2025), Mass2 consists of 9000 lines of Python (plus 2500 lines of test code and 2500 lines of examples in Marimo notebooks). It used to use extension modules in Cython, but we removed the last of these in early 2025.

### Versions
* MASS version 2 was begin in August 2025. It is still unstable, in alpha or pre-alpha status. Find it at https://github.com/usnistgov/mass2
* MASS version 1 was begun in November 2010. Bug-fix development continues. Find it at https://github.com/usnistgov/mass


### Version 2 versus Version 1
Mass version 2 differs from version 1 in how it handles bookkeeping and the organization of data.

Version 1 had some weaknesses that we are trying to solve in Mass 2. In version 1, all per-pulse data were stored as separate array--each a different attribute of an incredibly complicated class. All of the important arrays were actually HDF5 datasets. This approach enabled automatic backup of an analysis in progress, via the HDF5 file. Unfortunately, it also led to a great deal of bookkeeping code, and the HDF5 backing was only possible for the specific quantities defined in Mass. If you wanted to compute any per-pulse quantities not anticipated by Mass, they could not easily be stored in or loaded from the same HDF5 file. Also, the set of available operations on an HDF5-backed dataset was similar to, but not _identical_ to a normal numpy array, which led to no small amount of confusion. Other problems with Mass 1 included: a code object for cutting bad pulses and labelling pulses according to the "experiment state" whose interface was impossible to remember; too-tight coupling between analysis tools and the data structures they operated on; and the absence of a clear listing of the fully supported quantities.

But the biggest problem with Mass version 1 was how tighly it mixed up two conceptually distinct tasks:
1. _learning_ the parameters of an analysis (such as computing an optimal filter or an energy-calibration curve), and
2. _performing_ the analysis on a set of data (applying that filter or calibration curve).
Mass 1 assumed that you'd acquire a full set of data across multiple hours, and then you'd both learn from and analyze the full set, probably with multiple iterations to tweak the analysis steps. Then any future data set would be analyzed ab initio, without information from earlier work.

Mass 2 is meant to improve on the previous approach, without throwing away the existing, useful tools (optimal filters, fluorescence line shapes and line fitters). It improves in two ways. First, to reduce our reliance on homemade code to organize our data, we are turning to the high-performance [Polars dataframe library](https://pola.rs/). Polars is capable of reading and writing data in several industry-standard formats, including Apache Parquet files, or it can store data in and read data from a full-fledged database. Most importantly, Polars uses a column-oriented design, with major performance benefits for the sort of analysis problem Mass addresses.

Mass 2 also makes it much easier to design and reuse an analysis pipeline that converts raw pulses into microcalorimeter energy estimates. The sequence of analysis steps, with all its channel-specific parameters, is designed to be easy to store (as a Python pickle file) and then to replay later on a new set of data.

We expect that the major differences between Mass versions 1 and 2 will be disruptive to experienced users. They will be large enough to justify storing the code as two separate repositories, [mass2](https://github.com/usnistgov/mass2) and [mass](https://github.com/usnistgov/mass). We also expect that they will lead to an overall better analysis experience.



## Installation
Mass requires Python version 3.10 or higher. (GitHub tests it automatically with Python versions 3.10 and 3.13.) We strongly recommend installation inside a virtual environment to help with dependencies and avoiding conflicts. Here are some ways to install Mass 2:

1. For a data-acquisition computer, install the whole software suite together: https://github.com/usnistgov/microcal-daq/. [Instructions](#1-install-the-microcal-daq-meta-package)
2. For a personal workstation or other non-lab computer, install using the modern [uv](https://docs.astral.sh/uv/) package manager. [Instructions](#2-use-the-uv-package-manager)
3. If you have conda installed already, you can install Mass 2 within conda. [Instructions](#3-in-a-conda-environment)
4. If you have python installed already, you can install Mass system-wide. [Instructions](#4-no-virtual-environment)

We recommend option #1 for a DAQ computer and #2 for a personal computer


### 1. Install the `microcal-daq` meta-package

We recommend the `caldaq` approach for a laboratory computer that will be acquiring data, as well as analyzing it. For a personal workstation, `caldaq` is not a great idea: it installs many packages you don't need, and it won't succeed on non-Linux computers unless you take special care to install non-Python requirements. (If you want to use it on a Linux computer without all the usual DAQ packages, you can change the `repository-info.yaml` file before using the `calidaq` script.)

You can find [microcal-daq at GitHub](https://github.com/usnistgov/microcal-daq).

### 2. Use the `uv` package manager

For non-lab workstations, we recommend using the fast, modern [uv package manager](https://docs.astral.sh/uv/). In brief, installation looks like the following. It assumes you want to install in an environment called `mass`. Change the variable `NEW_ENVIRONMENT` in the following lines, if you prefer a different environment name.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# The installer adds the location of the new program to your $PATH but only in future shells;
# you have to re-source .bashrc (or start a new shell) to get access now:
. ~/.bashrc

# Test that uv is in your path, and install python
uv --version
uv python install 3.12
uv python update-shell

export NEW_ENVIRONMENT=analysis
cd ~
uv venv ${NEW_ENVIRONMENT}
source ${NEW_ENVIRONMENT}/bin/activate
echo ". ~/${NEW_ENVIRONMENT}/bin/activate" >> ~/.bashrc
```

Now that you have `uv` and a new environment, and you've activated that environment, you can install `mass2`. Navigate to the desired directory and install it. The `-e` flag means "editable", i.e., so you can edit the code and have changes be reflected.
```bash
cd /wherever/he/mass/directory/should/live
#
uv pip install --upgrade pip uv
uv pip install -e git+https://github.com/usnistgov/mass2.git#egg=mass2
```

The above (HTTPS) cloning method is probably simpler initially (no ssh setup). On the other hand, users who contribute to MASS will prefer to set up password-free connections with an ssh key. For them, instead of using the last line above, contributors should use ssh-based cloning:
```bash
uv pip install -e git+ssh://git@github.com/usnistgov/mass2.git#egg=mass
```

If you install in any virtual environment, the install location will be inside the `MYVENV/src/mass` where `MYVENV` is the name of your venv. You can switch git branches and update from GitHub in that directory and have everything take effect immediately.



### 3. In a Conda environment

If you are installing Python through the conda package manager, either the full Anaconda Python, or the slimmed-down Miniconda, you will install MASS into a conda environment. **Warning!** The site https://anaconda.com/ is now blocked from NIST for reasons having to do with licensing and lawyers, so you are unlikely to be able to use either full Anaconda or Miniconda on a NIST computer.

#### 3a. Install `miniconda` or `miniforge`

For computers off the NIST site, see [Installing miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) for instructions on how to install miniconda. It's a 3-liner at the terminal: you download the installation script, run it, and delete it. Easy!

If you want to try to make the `miniforge` installer work, you can find instructions at https://github.com/conda-forge/miniforge/. Miniforge does not get packages from anaconda.com, yet it still seems to require extra steps to make it work from NIST.

#### 3b. Install mass2 in a conda environment

If you manage to install conda, it is possible to install MASS in your base conda environment, but you shouldn't. That would defeat a key purpose of Conda, which is to let you work on separate projects that might require conflicting versions of libraries such as `numpy`. In this example, we assume you want your Conda environment to be named `analysis`:
```bash
conda create --name analysis -y
conda activate analysis
pip install --upgrade pip
pip install -e git+https://github.com/usnistgov/mass2.git#egg=mass
```

The above (HTTPS) cloning method is probably simpler initially (no ssh setup), but users who contribute to MASS might prefer to set up password-free connections with an ssh key. For them, instead of using the last line above, contributors might want to use ssh-based cloning:
```bash
pip install -e git+ssh://git@github.com/usnistgov/mass2.git#egg=mass
```

You'll also need to remember to activate the `analysis` environment in each terminal where you want to use MASS, via
```bash
conda activate analysis
```
This could be made automatic in your `.bashrc` or `.profile` file, if you like.

When I tried this, the MASS source code was installed in `~/src/mass`.



### 4. No virtual environment

To install mass in `~/somewhere/to/install/code/mass2` you do this:
```
cd ~/somewhere/to/install/code
pip install -e git+ssh://git@github.com/usnistgov/mass2.git#egg=mass2
```

If you don't add an ssh key to your account the following might work (depending on GitHub's current security policies), but you'll need to type in a password for installation and each update:
```
pip install -e git+https://github.com/usnistgov/mass2#egg=mass2
```

In the above instance, mass will be installed as `~/somewhere/to/install/code/src/mass2`. That penultimate directory (`src/`) follows from pip's rules.

### Using nonstandard releases, tags, branches, or commits
If you want to install a certain branch `branchname`, you can go to the installation directory and use the usual git commands to change branches, or you can install directly from the branch of choice with the syntax `@branchname`, like this:
```
pip install -e git+ssh://git@github.com/usnistgov/mass2.git@branchname#egg=mass2
```

The same syntax `@something` can be used for tags, branch names, or specific commit hashes.

### Updating the installation

The `-e` argument to the `pip install` command makes development really easy: you can change python files, and the next time you import mass the new files will be used.



## Scripts
Mass installs 2 scripts (as of August 2025). These are `ljh_merge` and `ljh_truncate`. You can check `pyproject.toml` and look for the `project.scripts` section to see if the list has been changed. These should be executable from your terminal from anywhere without typing `python` before them, though you may need to add something to your path. If you need to add something to your path, please use this approach to make them part of the MASS installation. The scripts all have help accessible via, e.g., `ljh2off --help`.


# User Tips

## Configuring iPython and Matplotlib

I like to have Matplotlib start automatic in "interactive mode" and to use the Qt5 backend. Therefore I have a file `~/.matplotlib/matplotlibrc` whose contents include
```yaml
# Also see defaults, which live in a file whose partial path is
# .../site-packages/matplotlib/mpl-data/matplotlibrc
backend:       Qt5Agg
interactive:   True
timezone:      US/Mountain
```

That's for a Mac or Linux. I understand that on Windows, you'd put the file in `C:\Users\YourUsername\.matplotlib\matplotlibrc`.

There are also settings I like to automatically make for any iPython session. You can create any 1+ python scripts in the `~/.ipython/profile_default/startup/` directory. They will be executed in lexical order. I have the following in mine. You'll notice that I specifically avoid importing mass2, so as not to slow down iPython when I am doing something that doesn't involve mass.

File `~/.ipython/profile_default/startup/50-imports.py`
```python
import pylab as plt
import numpy as np
import scipy as sp
import h5py
print("Imported pylab, numpy, scipy, and h5py")
plt.ion()  # Make pylab start in interactive mode

# ENABLE AUTORELOAD
ip = get_ipython()
ip.run_line_magic('load_ext', 'autoreload')
ip.run_line_magic('autoreload', '2')
print("Imported autoreload. (Use magic '%autoreload 0' to disable,)")%
```

File `~/.ipython/profile_default/startup/60-favorites.py`:
```python
"""
favorites.py

Functions that I want loaded in every ipython session.
"""

import os
import numpy as np
import pylab as plt

from pylab import clf, plot, subplot, scatter, semilogy, semilogx, loglog


def myfigure(size, fignum=9):
    """Generate a figure #9 to replace the current figure. Do so only if
    the requested size tuple (width,height) is different by more than 0.01 inches
    in either dimension from the current figure's size."""
    curr_size = plt.gcf().get_size_inches()
    if abs(curr_size[0] - size[0]) + abs(curr_size[1] - size[1]) > .01:
        plt.close(fignum)
    return plt.figure(fignum, figsize=size)


def imshow(matrix, fraction=.09, *args, **kwargs):
    """Plot a matrix using pylab.imshow with rectangular pixels
    and a color bar. Argument 'fraction' is passed to the colorbar.
    All others go to pylab.imshow()"""
    plt.clf()
    plt.imshow(matrix, interpolation='none', *args, **kwargs)
    plt.colorbar(fraction=fraction)


def hist(x, bins=100, range=None, *args, **kwargs):
    """Plot a histogram using the (non-default) 'histtype="step"' argument
    and default bins=100.."""
    kwargs.setdefault('histtype', 'step')
    return plt.hist(x, bins=bins, range=range, *args, **kwargs)


print("Loaded favorites.")
```

# Development Tips

## Code style

We are using the tool `ruff` to check for violations of Python programming norms. You can manually check for these by running either of these two complementary commands from the top-level directory of the MASS repository:

```
ruff check

# Or if the following, equivalent statement is more memorable:
make ruff
```


You'll need to install `ruff` via uv, pip, macports, or whatever for these to work (e.g., `uv pip install ruff`).

Ideally, you'll have zero warnings from ruff. This was not true for a very long time, but we finally reached full compliance with MASS release `v0.8.2`.

To _automate_ these tests so that it's easy to notice noncompliant new code as you develop MASS, there are tools you can install and activate within the VS Code development platform.

1. Install the tool `Ruff` for VS code. The offical identifier is `charliermarsh.ruff`.
2. Checking its settings (there are 15 of them at this time, which you can find by checking the VS code settings for `@ext:charliermarsh.ruff`). I found most default settings worked fine, but I did have to change:
   1. I added two lines to the `Ruff > Lint: Args` setting, to make it run the way we want:
      * `--line-length=135`
      * `--preview`
   2. I chose to have Ruff run `onType` rather than `onSave`, because the former did not cause noticeable burdens.
3. Install the tool `Flake8` for VS code, official identifier: `ms-python.flake8`
4. In its settings, add the line `--max-line-length 135` to the setting `Flake8: Args`

## Pre-commit

We are just starting to experiment with the `pre-commit` script, a tool to automate code checking before you make a commit. Watch this space for future information.
```bash
cd /where/your/installation/lives/mass2
uv pip install pre-commit
pre-commit install
```

Then `pre-commit` will (??) run automatically before each commit. Or not--I'm not totally sure. You can always say `pre-commit` from the `mass2` directory to run all the relevant checks (will modify some files in-place).


## Tests

If you look for files in the `tests/` directory, they will have examples of how the tested functions are called. You can learn a fair amount from looking at them.

Run all the tests on your system and make sure they pass!. From the `mass2` directory, say `pytest`. Tests require that you installed via `pip install -e ...`.

### Auto-run subsets of tests
`pytest-watch --pdb -- mass2/core` run from the source directory will run only the tests in `mass2/core`, automatically, each time a file is saved. Upon any error, it will drop into pdb for debugging. You will need to `pip install pytest-watch` first.

## Working on docs + tests
Change directory into `doc`, then:

  * For Posix (Mac/Linux) `make doctest html && open _build/html/index.html`
  * For Windows cmd shell `make doctest html && start _build/html/index.html`
  * For Windows Powershell `./make doctest;./make html;start _build/html/index.html`

Read about RST (reStructuredText) format. It is weird. My most common mistake is forgetting the blank line between `.. blah` statements and the following text. See the [Sphinx docs](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) for details about its syntax.
