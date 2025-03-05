# Tracer data files

This directory contains the hdf5 archive containing data for cool gas in clusters, 
identified at redshift zero and tracked back in time.

This file is meant to act as data documentation for the archive. The archive file in
turn is located in this directorz. It is called `cool_gas_history.hdf5`.

## What this data is

The files are meant to trace all those particles that become cool gas at redshift zero,
back in time throughout all snapshots, using tracer particles. It contains various
properties of all the particles that eventually, at redshift zero, end up in gas cells 
with temperatures below $10^{4.5} K$ and within two virial radii of their host halos. The 
datasets are not straightforward to use, they require some understanding of the structure 
they are meant to be used on. This file documents what the archive contains and how to
use its contents.

## Structure of hdf5 file

The hdf5 file has 352 data groups corresponding to the 352 original zoom-in regions of
the TNG-Cluster simulation. They are named `ZoomRegion_XXX` where `XXX` is the index of
the zoom-region, padded with zeros. For example `ZoomRegion_000` is the first zoom-in
region, containing halo 0.

Each group has multiple datasets. They are mostly arrays of shape `(100, N)` where `N` is
the number of tracers that end up in gas cells that have temperature log(T) < 4.5 and
are within two virial radii of their host halo at redshift zero. This may not translate
to there also being `N` unique particles at redshift zero that fulfill these criteria, as
multiple tracers can be in the same gas cell.

The following section details the different contents of the file and explains them and
their usage.

### Header

The header is a group titled `Header`. The header of the file contains the following 
information:

- `EarliestMPBSnapshot`: This is the number of the first snapshot in which the merger tree
  contains a main progenitor for all 352 primary subhalos. It is therefore the earliest
  snapshot to which any analysis based on the subhalos can be performed back to. Which 
  subhalo is considered the primary subhalo, i.e. whose MPB is tracked back depends on the
  snapshot at which the tracer data is selected. The primary subhalo of every halo is found
  at this same snapshot. This is the `OriginSnapNum` snapshot (see below). Note that some 
  halos may have primary subhalos that can trace their main progenitor branch further back 
  in time; this number only gives the earliest snapshot at which _all_ primaries have an
  identified main progenitor.
- `OriginSnapNum`: The snapshot at which the cool gas tracers were identified. This is the
  snapshot at which gas cells were selected to have temperature log(T) < 4.5 and distance
  to their host halo of less than two virial radii. The primary subhalo of every halo is
  also chosen at this snapshot. 

These are only available through the `.attrs` attribute of the file, since they are not full
datasets.

### Tracer data 

Every zoom-in region group has the following three datasets for tracer data:

- `particle_indices`: This field contains the indices of the traced particles at all
  snapshots. It has shape `(100, N)` where the first axis orders the snapshots. The
  snapshots are ordered in ascending order, meaning index `i` selects the i-th snapshot.
  The second axis is the array index of the traced particles at this redshift. These are
  indices into an artificial array of all particles of type 0, 4, and 5 in this zoom-in
  region at this snapshot, concatenated in this order. If two or more tracers are located 
  in the same gas cell, its ID will be in the array twice or multiple times. The order 
  of this array is such that the same tracer always is at the same position, i.e. the 
  j-th index is always pointing to the parent cell of the j-th tracer.
- `particle_type_flag`: This is an array of shape `(100, N)` which associates every
  entry of `particle_ids` with its particle type. Entries can only be 0, 4 or 5 for
  gas particles, star & wind particles, or black hole particles respectively.
- `total_particle_num`: This is an array of shape `(100, 3)` containing the total number
  of particles of each type in the entire zoom-in region. They are useful to shift the
  indices "to the left" to turn them from indices into the artificially constructed
  contiguous array into indices into an array of only particles of one type. See the
  explanation below for details on what this means.
- `uniqueness_flags`: This is an array of the same shape as the indices. It assigns to
  every index either a 1 or a 0. Every index's first occurrence in `particle_indices` is
  assigned a 1, and every subsequent duplicate is assigned a 0, so that these flags mark
  the unique occurrences of indices. One can use these flags to use particle data and
  avoid accidentally having a single particle contribute multiple times to a statistic
  purely by virtue of having more than one tracked tracer in it. See below for an example
  how uniqueness particles may be used.

### Particle data

Alongside the tracer data, the file also contains particle data. Each of the following
datasets is an array of shape `(100, N)`. They associate every index with the property
listed below. For example, the dataset `Temperature` assigns every index in `particle_indices`
the temperature of the corresponding particle. Some of these quantities only exist for 
certain particle types, for example temperature exists only for gas particles. For these
quantities, all entries that belong to a particle type that does not support it, are 
filled with `np.nan`. They can be excluded using the particle type flags:

```Python
with h5py.File("cool_gas_history.hdf5", "r") as file:
    temperatures = file["ZoomRegion_000/Temperature"][()]
    flags = file["ZoomRegion_000/particle_type_flag"][()]

gas_temperatures = temperatures[flags == 0]  # selet only gas cells
```

The following list are all available quantities:

- `Temperature`: Temperature of the particle in Kelvin. Exists only for gas cells (type 0).
- `Density`: Density of the gas particles in solar masses per ckpc cubed. Available only
  for gas particles; NaN for stars and BHs.
- `Mass`: The mass of the particle in units of solar masses.
- `DistanceToMP`: Distance of the particle to the main progenitor of the clusters primary
  subhalo in comoving kpc (ckpc). Available for all particles.
- `ParentHaloIndex`: The index of the halo to which the particle currently belongs. This
  index is the index into the array of halo properties as loaded from the group catalogue.
  Particles that are not bound to a halo have this set to -1.
- `ParentSubhaloIndex`: The index of the subhalo to which the particle currently belongs.
  This is the same as the `ParentHaloIndex`, but for SUBFIND subhalos. It is set to -1 for
  particles not bound to a subhalo (note that they might be bound to a halo though).

## How to use the indices

The indices stored in these files are indices into a contiguous array of all particles
of type 0, 4, and 5 in that order. That means the indices point to positions in an array
in which the first X entries are the particles of type 0, the next Y entries are particles
of type 4, and the remaining Z entries are particles type 5. It is easiest to think of
this in terms of code, so the following example will explain it best.

### Indexing all particles at the same time

Let's assume you wish to get only the IDs of all particles (gas, stars, *and* black 
holes) at snapshot 8 from zoom-in region 200 that contain tracers that will eventually 
end up in cool gas at redshift zero. You can achieve this the following way:

```Python
import numpy as np
import hdf5
from library.data_acquisition import particle_daq

# load particle IDs from sim
particle_ids_list = []
for part_type in [0, 4, 5]:
    particle_ids_list.append(
        particle_daq.get_particle_ids(
            "path/to/tngcluster/sim/data",
            snap_num=8,
            part_type=part_type,
            zoom_id=200,
        )
    )

# create a contiguous array of IDs
particle_ids = np.concatenate(particle_ids_list, axis=0)

with hdf5.File("cool_gas_data.hdf5", "r") as f:
    particle_indices = f["ZoomRegion_200/particle_ids"][8, :]
    particle_type_flags = f["ZoomRegion_008/particle_type_flags"][8, :]

selected_ids_only = particle_ids[indices]
```

If you want to get only those particles that are of type 4 at snapshot 8, you can select
them the following way (assuming the same code as above):

```Python
indices_into_stars = particle_indices[particle_type_flags == 4]
star_ids_only = particle_ids[indices_into_stars]
```

### Indexing only a particular particle type

If you wish to get only gas particles, you can of course directly use the indices from
the file, as the indices assume the gas cells to be the first section of the concatenated
array, and therefore it makes no difference whether you load all three particle types or
just the gas particle data - the indices will point to the correct locations:

```Python
import numpy as np
import hdf5
from library.data_acquisition import particle_daq

# create an array of gas particle IDs
particle_ids = particle_daq.get_particle_ids(
    "path/to/tngcluster/sim/data",
    snap_num=8,
    part_type=0,  # gas particles
    zoom_id=200,
)

with hdf5.File("cool_gas_data.hdf5", "r") as f:
    particle_indices = f["ZoomRegion_200/particle_ids"][8, :]
    particle_type_flags = f["ZoomRegion_008/particle_type_flags"][8, :]

indices_into_gas = particle_indices[particle_type_flags == 0]  # gas only
gas_ids_only = particle_ids[indices_into_gas]
```

However, if you were to attempt to naively alter this code for type 4 or 5 by simply
replacing the `part_type=0` with `part_type=4` in the loading function, you would see
an `IndexError` trying to run the code. This is because the indices saved to file expect
the full array of all particles, so attempting to only load one part of it makes the
array too short and the saved indices will point to positions far beyond the length of
just the star particles array. 

The solution is to shift the indices such that they can be used on only the particle
type 4 or type 5 arrays. This can be done by subtracting from the indices the number of 
particles that come before them in the contiguous array. For example, the indices for 
star particles can be shifted by subtracting from its indices the number of gas particles
in the current zoom-in region, so that the indices then range from 0 to N, where N is 
the number of star particles in the zoom-in region and can thus be used directly on an
array of star particle data.

For this purpose, the files also contain a shape `(3, )` array containing the number of
particles in the snapshot. They can be easily used to shift indices:

```Python
import numpy as np
import hdf5
from library.data_acquisition import particle_daq

# create an array of BH particle IDs
particle_ids = particle_daq.get_particle_ids(
    "path/to/tngcluster/sim/data",
    snap_num=8,
    part_type=5,  # BH particles
    zoom_id=200,
)

with hdf5.File("cool_gas_data.hdf5", "r") as f:
    particle_indices = f["ZoomRegion_200/particle_ids"][8, :]
    particle_type_flags = f["ZoomRegion_008/particle_type_flags"][8, :]
    particle_lens = f["ZoomRegion_008/total_particle_num"][8, :]

indices_into_bh = particle_indices[particle_type_flags == 5]  # BH only
# shift indices to left
indices_into_bh = indices_into_bh - particle_lens[0]  # subtract gas
indices_into_bh = indices_into_bh - particle_lens[1]  # subtract stars
bh_ids_only = particle_ids[indices_into_bh]
```

### Avoiding duplicate contributions by the same particle

Let's assume you wish to know the mean temperature of the gas particles at snapshot
8 of zoom-in 200. You might be tempted to just take the mean of the `Temperature`
field of the traced particles and call it a day. That would however be dangerous: any
particle can contain more than one tracer and since the saved arrays follow _tracers_
and not _particles_, it is possible that two or more entries of a particle data array 
actually come from one and the same particle that just so happens to include multiple
tracers. 

To avoid having particles containing multiple tracers contribute to a statistic multiple
times, we can use the `uniqueness_flags` field to exclude all duplicated particles. The
field assigns all unique particles and the first occurrence of duplicated particles a `1`,
so we can limit the array of our quantity to only the indices where `uniqueness_flags` is
one:

```Python
import h5py
import numpy as np


with h5py.File("cool_gas_data.hdf5", "r") as f:
    temperatures = f["ZoomRegion_200"]["Temperature"][8, :]
    uniqueness_flags = f["ZoomRegion_200"]["uniqueness_flags"][8, :]

unique_temperatures = temperatures[uniqueness_flags == 1]

mean_temperature = np.nanmean(unique_temperatures)  # correct value
```
