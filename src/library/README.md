# Structure of the `library` directory

This document gives a short overview over the structure of the `library` 
directory.


## Packages

The following packages are available:

- `config`: Bundles all configuration and set-up modules together, including
  logging.
- `data_acquisition`: Modules that wrap `illustris_python` or directly load
  data from the simulation hdf5 files. Also allows for some pre-processing
  where sensible (e.g. unit conversion, validation etc.).
- `loading`: Utilities to load plot data previously saved to file. Loading
  functions implicitly define schema of data files.
- `plotting`: Utilities for plotting data. Modules should **not** save any plots
  to file but rather return figure and axes objects.
- `processing`: Modules for processing data and utilities for data processing,
  such as parallelization tools. All kind of statistics, calculations, numerics
  etc. are found in this package.


## Top-level modules

The following top-level modules exist:

- `compute`: Functions for computing physical quantities.
- `constants`: Physical constants, global constants, simulation-specific constants.
- `units`: Tools for unit conversion and unit verification.
