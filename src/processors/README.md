# The `processors` package: manual

This document gives a brief overview over the `processors` package and how to 
best use its contents.

## What is this package?

The processors package contains classes that are meant to unite the acquisition, 
processing and plotting of data. Every processor aims to create one specific
(set of) plot(s). Since the code required for most of these plots shares a lot
of similarities, these common code fragments are unified in a base class. All
processors inherit this shared functionality from this base class. 

The processors also set out a simple way to acquire, process and plot data by
means of a common set of methods that can be called on instances of every 
processor to produce results. 

## How do I use a processor?

Any processor class can be instantiated. Once an instance exists, it provides
three (in some cases more) public methods that can acquire and process, plot or
load existing data:

- `get_data` loads the required simulation data and processes it into the format
  required for the plot.
- `load_data` can load previously saved processed data from file. This is used
  to avoid repeated long processing times when only adjusting plot details (such
  as a color or marker property).
- `plot_data` will create and save the plot of the data, that the processor is
  responsible for creating. It requires that the data has been acquired using
  either of the previous two methods.

A typical use of a processor could look like this:

```Python
>>> from processors import sample_processor
>>> p = sample_processor.SampleProcessor()
>>> p.get_data()
>>> p.plot_data()
```

The finished plots will show up in the directory specified by the configuration
in the [config](./../config.py) module.

## How do processors store data?

The loaded data is stored in attributes. The data that is directly calculated
from the gas cell temperatures of every halo is placed in the `data` attribute,
but subclasses may define other attributes for other data. Usually, any data
fields unique to that subclass are documented in the class docstring and/or 
are defined in the constructor method as stubs, so you can find available
fields there.

> Note that the primary data inside the `data` attribute is not necessarily
> sufficient for the plot. In some cases, it may even be irrelevant and only
> secondary data is required for the plots!

## What does the processor actually do "under the hood"?

The `plot_data` and `load_data` methods are pretty straight-forward and do 
exactly what the name suggests: the `load_data` method looks for the given
file and extracts the data from it, placing it into the attributes where it is
required. The `plot_data` method will take this data and produce a plot of it.

The `get_data` method is more involved: At the heart of the method, gas cells
from every halo in the simulation are loaded and their temperture is calculated
and then processed according to the subclass specifications. But it can also
load auxiliary data, post-process data and perform other tasks required to
produce the final data set. 

In detail, the `get_data` method performs the following steps:

1. Load the masses of all halos and create a list of IDs for them. This is done
   by calling the `_get_halo_data` method.
2. Before loading and processing the primary data, any further data can be
   loaded and processed by overwriting the base class' `_getauxilary_data` 
   method. It is called after setting up the halo data and before the main
   data acquisition step.
3. Gather primary data, using either multiprocessing or doing so sequentially. 
   The primary data is that data which is calculated per halo from every halos
   gas cell temperatures. This is done in three steps:
   1. For every halo, get all gas cells and calculates the gas cell temperatures.
   2. Pass the array of temperatures and gas data to the `_process_temperatures` 
      method which may be overwritten by subclasses of the base processor.
   3. Gather the results of the processing in an array and assigns this array 
      to the `data` attribute.
4. In a final (optional) step, the primary and auxiliary data may be further
   processed inside of the `_post_process_data`. This is useful to perform post-
   processing or verification of data. Post-processed data is then assigned to
   the corresponding attributes of the instance.

Most implementations of `get_data` also allow writing that data, which is 
ultimately required for plotting, to file. 

## How do I write my own processor?

Start by subclassing the `BaseProcessor` class. To fill this skeleton class with
functionality, you *must* subclass these methods:

- `plot_data`: Must implement plotting the data and saving the figure to file.
- `load_data`: Must implement a method to load data saved to file by
  `_post_process_data` and/or `_get_auxilary_data`.
- `_process_temperatures`: Take an array of gas cell temperatures for every
  halo and populate the `data` attribute with some data calculated from it.

You *can* then optionally subclass these methods:

- `_get_auxilary_data`: Load any additional data not related to gas temperature.
- `_fallback`: Return a fallback dataset for any halo that contains no gas. By
  default, this returns an array of zeros of the expected length. Subclasses
  should implement a fallback appropriate for the task (for example an array of
  NaNs might be suitable in some cases).
- `_post_process_data`: Is called after primary data is loaded and processed.
  This method can be used to further process it (for example to average it over
  all halos) or to verify it. It can also be used for any teardown operations
  such as closing streams, purging memory, etc.
- `_skip_halo`: Given a halo ID decides whether the computation of gas cell
  temperatures of this halo may be skipped. Useful to avoid calculations for
  halos whose data is irrelevant to the plot.


You *must not* change these methods (unless you are certain you know what you 
are doing):

- `_get_halo_data`: Only overwrite this when you need additional data from halos
  (such as radius or position). Even in this case, the `masses` and `indices`
  attributes *must* be populated by this method!
- `get_data`: Since this method calls all the private methods of the class,
  overwriting it without care will break the processor. You may extend it, if
  necessary, but prefer using `_post_process_data` for any tasks to perform
  after loading and processing the primary data.
- `_get_data_multiprocessed`, `_get_data_sequentially` and `_get_data_step`:
  These are required by `get_data` and should not be overwritten or extended.

You can get further information about the purpose of everyone of these methods
from their docstrings or using the `help` function in an interactive shell:

```Python
>>> help(BaseProcessor._post_process_data)
```

It is recommended for clarity to define all additional data fields that your
subclass will need in the constructor `__init__` method (this also avoids
issues with linting tools and is just good practice). For memory reasons, it
is recommended to not allocate memory for these fields, at least not when using
multiprocessing! The `multiprocessing` module will create a copy of the instance
for every process - and you will run out of memory fast if your instance has
pre-allocated memory for every field. Set fields to a default of `None`, i.e.
treat data fields as type `Optional[NDArray]`.

> NOTE: Due to the same reason, allocating memory for the data and then placing 
> it into the allocated memory space will not work during multiprocessing: 
> every process will receive its own copy of the instance, with every copy of 
> the data field containing a reference to a different location in memory. 
> Assigning values to this array in memory is effectively useless, as the array 
> is destroyed  together with the instance when the process is terminated; 
> **Data assigned to pre-allocated arrays in memory does not persist beyond 
> multiprocessed operations!**
