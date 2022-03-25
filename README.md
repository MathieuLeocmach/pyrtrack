# PyrTrack


A Python package to localise and size spherical particles of various sizes as
explained in Mathieu Leocmach, & Hajime Tanaka *A novel particle tracking method with individual particle size measurement and its application to ordering in glassy hard sphere colloids.* **Soft Matter** 9, 1447–1457 (2013).
doi:[10.1039/c2sm27107a](https://doi.org/10.1039/c2sm27107a).

Author: Mathieu Leocmach

Pyrtack localises spherical particles of arbitrarily relative sizes from either 2D or 3D (confocal) images either in a dilute or crowded environment. Moreover this method allows us to estimate the size of each particle reliably.

## Licence, citations, contact

This code is under GPL 3.0 licence. See LICENCE file.

Please cite the above publication in any scientific publication using
this software.

Contact Mathieu LEOCMACH, Institut Lumière Matière, UMR-CNRS 5306, Lyon,
France mathieu.leocmach AT univ-lyon1.fr

## Install

Dependencies are numpy, scipy and numba. Tested with python 3.

You can install with pip: pip install -e .

## Usage

You can find a 2D tutorial in (pyrtrack/notebooks/Multiscale.ipynb)[https://github.com/MathieuLeocmach/pyrtrack/blob/master/pyrtrack/notebooks/Multiscale.ipynb
