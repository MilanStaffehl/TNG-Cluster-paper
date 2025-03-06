# Cool gas history archive

This file contains a description of the data in the hdf5 archive. The archive contains data for cool gas in clusters, identified at redshift zero and tracked back in time.

This file is meant to act as data documentation for the archive. The archive file in turn is located in the directory specified by your `config.yaml`. It is called `cool_gas_history.hdf5`. If you are on the VERA cluster and found this file via a link in a data directory, you may already have this archive available to you.

## What the archive data is

The cool gas history archive is meant to trace cool redshift zero gas in clusters of the TNG-Cluster simulation back in time throughout all snapshots, using tracer particles. It contains various properties of all the particles that eventually, at redshift zero, end up in gas cells with temperatures below $10^{4.5} K$ and within two virial radii of their host halos. The datasets are not straightforward to use; they require some understanding of the structure they are meant to be used on. This file documents what the archive contains and how to use its contents.

## How to generate the archive

Read the instructions in the [README](./../README.md) to generate the archive file. You can find them in the _data generation_ section.

## Structure of hdf5 file

The hdf5 file has 352 data groups corresponding to the 352 original zoom-in regions of the TNG-Cluster simulation. They are named `ZoomRegion_XYZ` where `XYZ` is the index of the zoom-region, padded with zeros. For example `ZoomRegion_000` is the first zoom-in region, containing halo 0. 

Each group has multiple datasets. They are mostly arrays of shape `(100, N)` where `N` is the number of tracers that end up in gas cells that have temperature log(T) < 4.5 and are within two virial radii of their host halo at redshift zero. This may not translate to there also being `N` unique *particles* at redshift zero that fulfill these criteria, as multiple tracers can be in the same gas cell.

The following section details the different contents of the file and explains them and their usage.

### Header

The header is a group titled `Header`. The header of the file contains the following information:

- `EarliestMPBSnapshot`: This is the number of the first snapshot in which the merger tree can identify a main progenitor for all 352 primary subhalos simultaneously. It is therefore the earliest snapshot to which any analysis based on the merger tree can be performed back to. Note that some halos may have primary subhalos that can trace their main progenitor branch further back in time than this; this number only gives the earliest snapshot at which _all_ primaries have an identified main progenitor. The subhalo whose main progenitor branch (MPB) to follow is identified at the `OriginSnapNum` snapshot (see below). 
- `OriginSnapNum`: The snapshot at which the cool gas tracers were identified. This is the snapshot at which gas cells were selected to have temperature log(T) < 4.5 and distance to their host halo of less than two virial radii. The primary subhalo of every halo is also chosen at this snapshot. Typically, this is done at redshift zero, i.e. at snapshot 99.
- `TotalPartNum`: Total number of identified tracers across all clusters, i.e. the sum of the number of tracers per cluster. This is useful to allocate memory for loading data for every individual tracer from every cluster.

These are only available through the `.attrs` attribute of the file, since they are not full datasets.

### Tracer data 

For every group, a set of `N` tracers is identified to reside within cool gas particles of the corresponding cluster. They are referred to as "selected tracers". These tracers are associated with exactly one particle of the simulation each at every redshift. Every zoom-in region group has the following datasets for these selected tracers:

- `particle_indices`: This field contains the indices of the particles associated with the selected tracers, at all snapshots. It has shape `(100, N)` where the first axis orders the snapshots. The snapshots are ordered in ascending order, meaning index `i` selects the i-th snapshot. The second axis is ordered by the tracers. The order of this axis is such that the same tracer always is at the same position, i.e. the j-th index is always pointing to the parent cell of the j-th tracer. The array contains the "contiguous array index" of the traced particles at this redshift. These are indices into an artificial array of all particles of type 0, 4, and 5 in this zoom-in region at this snapshot, concatenated in this order. If two or more tracers are located in the same gas cell, its ID will be in the array twice or multiple times. See [below](#how-to-use-the-indices) for an example of how to work with these indices.
- `particle_type_flag`: This is an array of shape `(100, N)` which associates every entry of `particle_ids` with its particle type. Entries can only be 0, 4 or 5 for gas particles, star & wind particles, or black hole particles respectively.
- `total_particle_num`: This is an array of shape `(100, 3)` containing the total number of particles of each type in the entire zoom-in region. They are useful to shift the indices "to the left" to turn them from indices into the artificially constructed contiguous array into indices into an array of only particles of one type. See the explanation below for details on what this means.
- `uniqueness_flags`: This is an array of shape `(100, N)`. It assigns to every tracer either a 1 or a 0, depending on whether the corresponding index in `particle_indices` has already occured in the current snapshot previously, i.e. whether the tracer shares its particle with another selected tracer. Every index's first occurrence in `particle_indices` is assigned a 1, and every subsequent duplicate is assigned a 0, so that these flags mark the unique occurrences of indices per snapshot. One can use these flags to avoid accidentally having a single particle contribute multiple times to a statistic purely by virtue of having more than one selected tracer in it. See below for an example how uniqueness of tracer particles may be used.

### Particle data

Alongside the tracer data, the file also contains data for the particles that the tracers are associated with. Each of the following datasets is an array of shape `(100, N)`. They associate every tracer with a property of the corresponding particle, listed below. For example, the dataset `Temperature` assigns every index in `particle_indices` the temperature of the corresponding particle. Some of these quantities only exist for certain particle types, for example temperature exists only for gas particles. For these quantities, all entries that belong to a particle type that does not support it, are filled with `np.nan`. They can be excluded using the particle type flags:

```Python
with h5py.File("cool_gas_history.hdf5", "r") as file:
    temperatures = file["ZoomRegion_000/Temperature"][()]
    flags = file["ZoomRegion_000/particle_type_flag"][()]

gas_temperatures = temperatures[flags == 0]  # selet only gas cells
```

The following list contains all available quantities:

- `Temperature`: Temperature of the particle in Kelvin. Exists only for gas cells (type 0).
- `Density`: Density of the gas particles in solar masses per ckpc cubed. Available only for gas particles; NaN for stars and BHs.
- `Mass`: The mass of the particle in units of solar masses.
- `DistanceToMP`: Distance of the particle to the main progenitor of the clusters primary subhalo in comoving kpc (ckpc). Available for all particles.
- `ParentHaloIndex`: The index of the halo to which the particle currently belongs. This index is the index into the array of halo properties as loaded from the group catalogue. Particles that are not bound to a halo have this set to -1.
- `ParentSubhaloIndex`: The index of the subhalo to which the particle currently belongs. This is the same as the `ParentHaloIndex`, but for SUBFIND subhalos. It is set to -1 for particles not bound to a subhalo (note that they might be bound to a halo though).
- `ParentCategory`: A flag describing the structure the particle belongs to, depending on its parent halo and parent subhalo. The flags are unsigned 32bit integers. Their meaning is listed below. The "primary halo" of a cluster always refers to the halo that hosts the main progenitor of its redshift zero primary subhalo.
  - 0: unbound, no parent ("unbound")
  - 1: bound to halo that is not the primary halo of the cluster ("other halo")
  - 2: bound to the primary halo of the cluster, but not any subhalo ("inner fuzz")
  - 3: bound to the primary halo of the cluster and its primary subhalo ("central galaxy")
  - 4: bound to the primary halo of the cluster and any other subhalo ("satellite")
  - 255: faulty entry, cannot assign category (caused by missing entry for the corresponding snap in the SUBLINK merger tree)
- `FirstCrossingRedshift`: The redshift at which the tracer crosses $2R_{200c}$ for the first time. Redshift is interpolated between snapshots.
- `FirstCrossingSnapshot`: The last snapshot at which the tracer is still outside $2R_{200c}$ before crossing it for the first time.
- `LastCrossingRedshift`: The redshift at which the tracer crosses $2R_{200c}$ for the last time. Redshift is interpolated between snapshots. May be identical to `FirstCrossingRedshift` if the tracer enters this distance only once.
- `LastCrossingSnapshot`: The last snapshot at which the tracer is still outside $2R_{200c}$ before crossing it for the last time. May be identical to `FirstCrossingSnapshot` if the tracer enters this distance only once.
- `FirstCrossingRedshift1Rvir`: The redshift at which the tracer crosses $R_{200c}$ for the first time. Redshift is interpolated between snapshots.
- `FirstCrossingSnapshot1Rvir`: The last snapshot at which the tracer is still outside $R_{200c}$ before crossing it for the first time.
- `LastCrossingRedshift1Rvir`: The redshift at which the tracer crosses $R_{200c}$ for the last time. Redshift is interpolated between snapshots. May be identical to `FirstCrossingRedshift1Rvir` if the tracer enters this distance only once.
- `LastCrossingSnapshot1Rvir`: The last snapshot at which the tracer is still outside $R_{200c}$ before crossing it for the last time. May be identical to `FirstCrossingSnapshot1Rvir` if the tracer enters this distance only once.
- `FirstCoolingRedshift`: The redshift at which the gas cell associated with the tracer cools below the temperature threshold of $10^{4.5}\,\rm K$ for the first time. Redshift is interpolated between snapshots. Stellar particles are assumed to be in the hot temperature regime.
- `FirstCoolingSnapshot`: The last snapshot at which the tracer is still above the temperature threshold of $10^{4.5}\,\rm K$ before cooling below it for the first time.
- `LastCoolingRedshift`: The redshift at which the gas cell associated with the tracer cools below the temperature threshold of $10^{4.5}\,\rm K$ for the last time. Redshift is interpolated between snapshots. Stellar particles are assumed to be in the hot temperature regime. May be identical to `FirstCoolingRedshift` if the gas associated with the tracer cools below the threshold only once. 
- `LastCrossingSnapshot`: The last snapshot at which the tracer is still above the temperature threshold of $10^{4.5}\,\rm K$ before cooling below it for the last time. May be identical to `FirstCoolingSnapshot` if the gas associated with the tracer enters this distance only once.


## How to use the indices

The indices stored in these files are indices into an artificially created contiguous array of all particles of type 0, 4, and 5 in that order. That means the indices point to positions in an array in which the first X entries are the particles of type 0, the next Y entries are particles of type 4, and the remaining Z entries are particles type 5. This array additionally only contains the particles of one zoom-in region, i.e. only of the `i`-th and `i + 352`-nd file of the TNG-Cluster snapshot files. 

### How to construct the artificial contiguous array

The array into which the indices of `particle_indices` point is constructed by loading only the particles in the FoF-file (`i`-th file of the snapshot, e.g.  `snap_099.9.hdf5`) and the fuzz-file (`i + 352`-nd file of the snapshot, e.g. `snap_099.361.hdf5`) of the corresponding zoom-in region and concatenating them in the following order:

1. Gas particles in the FoF-file (type 0)
2. Gas particles in the fuzz-file (type 0)
3. Star particles in the FoF-file (type 4)
4. Star particles in the fuzz-file (type 4)
5. BH particles in the FoF-file (type 5)
6. BH particles in the fuzz-file (type 5)

It is easiest to think of this in terms of code, so the following example will explain it best.

### Indexing all particles at the same time

Let's assume you wish to get only the IDs of all particles (gas, stars, *and* black holes) at snapshot 8 from zoom-in region 200 that contain tracers that will eventually end up in cool gas at redshift zero. You can achieve this the following way:

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

selected_ids_only = particle_ids[particle_indices]
```

If you want to get only those particles that are of type 4 at snapshot 8, you can select them the following way (assuming the same code as above):

```Python
indices_into_stars = particle_indices[particle_type_flags == 4]
star_ids_only = particle_ids[indices_into_stars]
```

### Indexing only a particular particle type

If you wish to get only gas particles, you can of course directly use the indices from the file, as the indices assume the gas cells to be the first section of the concatenated array, and therefore it makes no difference whether you load all three particle types or just the gas particle data - the indices will point to the correct locations:

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

However, if you were to attempt to naively alter this code for type 4 or 5 by simply replacing the `part_type=0` with `part_type=4` in the loading function, you would see an `IndexError` trying to run the code. This is because the indices saved to file expect the full contiguous array of all particles, so attempting to only load one part of it makes the array too short and the saved indices will point to positions far beyond the length of just the star particles. 

The solution is to shift the indices such that they can be used on only the particle type 4 or type 5 arrays. This can be done by subtracting from the indices the number of particles that come before them in the contiguous array. For example, the indices for star particles can be shifted by subtracting from its indices the number of gas particles in the current zoom-in region, so that the indices then range from 0 to N, where N is the number of star particles in the zoom-in region and can thus be used directly on an array of star particle data.

For this purpose, the archive also contains a shape `(3, )` array containing the number of particles in the snapshot: `total_particle_num`. They can be easily used to shift indices:

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

Let's assume you wish to know the mean temperature of the traced gas particles at snapshot 8 of zoom-in 200. You might be tempted to just take the mean of the `Temperature` field of the traced particles and call it a day. That would however be dangerous: any particle can contain more than one tracer and since the saved arrays follow _tracers_ and not _particles_, it is possible that two or more entries of a particle array actually come from one and the same particle that just so happens to include multiple selected tracers. 

To avoid having particles containing multiple tracers contribute to a statistic multiple times, we can use the `uniqueness_flags` field to exclude all duplicated particles. The field assigns all unique particles and the first occurrence of duplicated particles a `1`, so we can limit the array of our quantity to only the indices where `uniqueness_flags` is one:

```Python
import h5py
import numpy as np


with h5py.File("cool_gas_data.hdf5", "r") as f:
    temperatures = f["ZoomRegion_200"]["Temperature"][8, :]
    uniqueness_flags = f["ZoomRegion_200"]["uniqueness_flags"][8, :]

unique_temperatures = temperatures[uniqueness_flags == 1]

mean_temperature = np.nanmean(unique_temperatures)  # correct value: only 1 value per gas cell
```

### Returning archive indices to global indices

The rather esoteric indices saved in the archive file are convenient when working with individual zoom-in regions, and when always loading all particle types at once. However, sometimes the need may arise to return these indices to global particle indices, i.e. to return them back to the index they have in the full list of particles in the entire simulation volume. For this purpose, a function `index_to_global` exists. It can be used to turn the archived indices into global simulation particle indices again:

```Python
import h5py

from library import cgh_archive


# load particle data and particle types
with h5py.File("cool_gas_data.hdf5", "r") as f:
    particle_indices = f["ZoomRegion_200/particle_indices"][8, :]
    particle_types = f["ZoomRegion_200/particle_type_flag"][8, :]
    total_part_num = f["ZoomRegion_200/total_particle_num"][8, :]
    
# return indices to global
global_indices = cgh_archive.index_to_global(
    200,  # zoom-in ID
    particle_indices,
    particle_types,
    total_part_num,
    base_path="path/to/tngcluster/sim/data",
    snap_num=8,
)
```
