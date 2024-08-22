# Master thesis: cool gas in galaxy clusters

![Tests status badge](https://github.com/MilanStaffehl/thesisProject/actions/workflows/testing.yml/badge.svg)

> [!CAUTION]
> 
> This `README` is in parts outdated. It will be updated in the future to be
> more accurate.

This repository contains the source code for my master thesis about cool gas in
galaxy clusters at the ITA (University Heidelberg).

> [!WARNING]
> 
> This project is still in development. Expect drastic changes to occur 
> between commits.

## Prerequisites

The code in this repository requires Python 3.10 or higher to run.

You need access to the simulation data of the IllustrisTNG simulation.
Visit the [TNG website](https://www.tng-project.org/) for details.


## Installation

Technically speaking, the code base can be used by simply cloning it onto any
local machine. However, there are three major obstacles to using it this way:

1. The code expects certain subdirectories to exist that are not tracked by VC.
2. The code has third-party dependencies that are not available on PyPI or conda.
3. The code is written for execution on clusters, not PCs. 

The first two problems can be remedied by "installing" the repository. This 
repository comes with an installation Python script that will set up the 
expected directories. By default, it will place these directories inside the 
project directory (they are ignored by git by default). Alternatively, you can 
specify the location of the required directories manually. 

### Using the install script

Simply run `install.py` from the root directory of the project. You do not need
to run it from a dedicated Python environment as it does not alter any env vars
nor does it require any third-party dependencies.

After running the install script, install the dependencies (ideally inside of a
new venv). Start by installing the Illustris helper scripts using

```bash
cd external/illustris_python
git clone git@github.com:illustristng/illustris_python.git
cd illustris_python
pip install -e .
```

This installs the helper scripts inside the current environment in editable mode. 
Then install the remaining dependencies from PyPI using

```bash
cd ../../  # return to project root
pip install -r requirements.txt
```

if you only want to use the code as-is. If you wish to contribute to the code
base, use of pre-commit and the related pre-commit hooks is recommended. Run

```bash
cd ../../  # return to project root
pip install -r requirements-dev.txt
pre-commit install
```

instead. This installs and sets up pre-commit and dependencies. 

### Specifying custom directories

The install script will create the directories that are specified inside the
[`config.yaml`](./config.yaml) configuration file. They will be used as the root directory
for saving data files and figures respectively by the scripts. The default
location for these directories is within the project directory.

If you wish to place the figures and data home directories elsewhere, you can 
specify your own directories for data and figures. To do so, set the `data_home` 
and `figures_home` directory paths inside the [`config.yaml`](./config.yaml) file to the 
desired location, before running the install script.

Please note that this will still create the given home directories if they do
not yet exist. These directories will also be populated with a substructure of
subdirectories when running the scripts. Therefore, it is recommended to use
new directories to avoid conflicts with existing files.


## Executing code

The code is written to be executed on clusters. Execution on PCs is not
recommended and in some cases impossible: some scripts may use up to 1 TB of 
memory!

The intended way for this code to be executed is by using the Python scripts
inside the `scripts` directory. They come alongside batch scripts for submission
to slurm, in order to make use of the full computational power of the cluster.
Use either the Python scripts (be careful with RAM and CPU cores usage!) or 
submit batch jobs using the batch scripts. Note that the batch scripts are
adapted to the Vera cluster of the MPIA and not all of them will work on any
arbitrary cluster "out of the box".

If you want to know more about one of the Python scripts, use it together with
the `-h` flag:

```shell
python <name_of_the_script>.py -h
```

### Customizing simulation base paths

If you want to use this code outside the Vera cluster of the MPIA, you will
need to update the `base_paths` dictionary of the simulation data inside 
the [`config.yaml`](./config.yaml) module to wherever the simulation data of the 
different simulations are stored. Each entry of the `base_path` dictionary
should be a key-value pair, with the key being the name of the simulation as, 
and the value being the full path to the directory in which the `snapdir_XYZ` 
and `groups_XYZ` snapshot directories are located. Note that the name you give
to each simulation path can be arbitrary and does not need to match anything
in particular; it is merely the name by which you can select the simulation
in most scripts (via the `--sim` flag).

If you are using this code on the Vera cluster, the default settings should 
work "out of the box".


## Organization

The repository is organized into the following directories:

- `notebooks`: The notebooks directory contains Jupyter notebooks. The notebooks
  contain primarily test code, some on-the-side experiments and probing plots
  (that is, plots that are meant to get an overview over simulation data). 
- `scripts:` The scripts directory contains executable Python scripts that
  can be used to create plots. It also contains batch job scripts for use with
  slurm. The directory is organized into subdirectories. These subdirectories 
  are roughly divided by the plot type the scripts inside them are meant to
  produce. The names of the directories correspond to those of the project
  [milestones](#milestones). You can find out more about what each of these 
  milestones and subdirectories contain by reading the [Milestones](#milestones)
  or the GitHub milestones. 
- `src`: The source directory bundles all code that is used to generate plots
  and data for this project. Itis structured into three main packages:

  - `library`: The library directory contains the logic of this project, organized 
    into modules and packages. It is the heart of the project, containing all
    code that performs actual work, calculations and tasks. It is further
    subdivided:

    - `library.config`: This package contains modules that are used to set up the
      environment for pipelines and to create containers for globally required
      variables. It also contains modules to set up logging.
    - `library.data_acquisition`: This package contains modules whose purpose 
      it is to load simulation data directly from the simulation hdf5 files. 
      The functions in these modules can act as a simple wrapper to `illustris_python`
      functions, but some may also pre-process data. 
    - `library.loading`: As opposed to `data_acquisition`, the `loading` package
      contains modules that support loading processed data from file that was
      previously saved by a pipeline job (i.e. data from which a plot was created).
      It implicitly also defines the format in which pipelines must save the data.
    - `library.plotting`: This package contains modules that provide utilities for
      plotting data. 
    - `library.processing`: This is the largest package and contains various 
      modules for data processing. Data loaded with `data_acquisition` can be 
      further processed with code from this package. It also contains utilities 
      for parallelization and statistics.

  - `pipelines`: The pipelines directory contains modules with pipeline classes.
    These classes are used to bundle together code from `library` in a sensible 
    order to achieve the task of loading, processing, saving to file and plotting 
    data from the simulations. They are also solely responsible for file IO. 
    Similar to the `scripts` directory, this directory is divided into topics 
    that match the milestones of this project.

Alongside these directories tracked by the VC, the install script will also
create directories that will be populated by the Python and/or batch scripts:

- `data`: The data directory holds processed data produced by the scripts. It
  is organised into milestones. If you change its location in the `config.yaml`
  it ight not be inside the project directory.
- `external`: The external directory is used to install external dependencies
  locally, most noticeably the Illustris Python helper package.
- `figures`: The figures directory holds the finished plots produced by the
  scripts. It is organised into milestones. As with the `data` directory, you
  can change its location by setting it in the `config.yaml`. 

### Where do I find...?

The stucture of the repository might be a little confusing until one has 
familiarized oneself with it. If you just want to find something specific 
quickly, you can find some guidance here:

- **Re-usable code snippets:** You are most likely to find code that you might 
  wish to re-use inside the `src.library` directory. The modules and packages 
  therein are more or less intuitively named. Look for the module/package that 
  closest describes what you are looking for!
- **Batch scripts for slurm:** Batch scripts for the different tasks are 
  situated in the `scripts` directory. Consult the [Milestones](#milestones) to find 
  out what job you are looking for and then select the corresponding subdir
  of `scripts`. Here you will then find the scripts in the `batch` directory.
  Note that outside of the MPIA Vera cluster, you will most likely have to
  adapt the scripts to your clusters environment.
- **Scripts to reproduce plots:** Use the Python scripts under `scripts`.
  Consult the [Milestones](#milestones) to find out which of the subdirectories
  you need to look into. All the Python scripts in `scripts` have a CLI, so
  to find out how to use them, simply run `python <script name>.py -h`.
- **Code for topic X:** If you are looking for code (or output) of a specific
  topic, search the `library` directory for the package whose name sounds most 
  like what you need. Inside of it, you can go through the modules that are 
  related to the topic.
- **The finished plots:** If you installed the code using the `install.py`
  script, you will find the figures under the `figures` directory in the
  subdirectory of the milestone they belong to. If you cannot find them there,
  check the [`config.yaml`](./config.yaml) file for the `figures_home` path.
  You will find your figures under this path.


## Milestones

You will notice that the directories for figures, data and scripts are divided
into certain topics such as `temperature_histograms`. These topics are the topics
of a project milestone. A milestone in the context of this project is a smaller 
task, usually consisting of a single type of plot to produce. Every milestone 
aims to answer a small scientific question: how is the temperature in halo gas 
distributed? How does the cool gas mass change with halo mass? At what radius 
of a halo can we find most cool gas?

The following is a list of existing and tentative milestones:

- `temperature_distributions`: The distribution of gas temperatures in halos of
  the entire TNG300-1 mass range. These are plotted as 1D-histograms of gas
  mass fraction vs. temperature.
- `mass_trends`: The trend of gas fraction with halo mass, split by temperature
  regime (hot, warm, cool gas).
- `radial_profiles`: The trend of temperature and density of different gas
  components with halocentric distance. For temperature, these are shown as a
  2D histogram temperature vs. halocentric distance out to two times the virial
  radius. Gor density, these are simple density profiles. 


## Metadata

**Author:** Milan Staffehl
