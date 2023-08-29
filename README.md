# Master thesis: cool gas in galaxy clusters

This repository contains the source code for my master thesis about cool gas in
galaxy clusters at the ITA (University Heidelberg).

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

The first two problems can be remedied by "installing" the code. This repository
comes with an installation shell script that will set up the expected directories. 
Alternatively, you can specify the location of the required directories manually.

### Using the install script

Simply run `install.py` from the root directory of the project. You do not need
to run it from a dedicated Python environment as it does not alter any env vars
nor does it require any third-party dependencies.

After running the install script, install the dependencies (ideally inside of a
new venv). Start by installing the Illustris helper scripts using

```bash
cd external
git clone git@github.com:illustristng/illustris_python.git
cd illustris_python
pip install .
```

This installs the helper scripts inside the current environment. Then install 
the remaining dependencies from PyPI using

```bash
cd ../../  # return to project root
pip install requirements.txt
```

if you only want to use the code as-is. If you wish to contribute to the code
base, use of pre-commit and the related pre-commit hooks is recommended. Run

```bash
cd ../../  # return to project root
pip install requirements-dev.txt
pre-commit install
```

instead. This installs and sets up pre-commit and dependecies. 

### Specifying custom directories

If you wish to use this code without relying on the expected directory structure,
you can specify your own directories for data and figures. To do so, set the
`data_home` and `figures_home` directory paths inside the 
[`config.yaml`](./config.yaml) file to the desired location, before running the
install script.

Please note that this will still create a set of subdirectories that are 
required to work with the default file paths and names in this project. If you
wish to control the location of data and plot files in full, every script also
offers the option to specify the directory where to save them.


## Executing code

The code is written to be executed on clusters. Execution on PCs is not
recommended and in some cases impossible: some scripts use up to 250 GB of 
memory. 

If you want to use this code outside of the Vera cluster of the MPIA, you will
need to update the `simulation_home` directory of the simulation data inside 
the [`config.yaml`](./config.yaml) module to wherever the simulation data is 
stored. Note that this directory must refer to the parent directory of the
individual simulation data directories, i.e. it should not contain the data
directly. If you are using this code on the Vera cluster, the default settings 
should work "out of the box".

The intended way for this code to be executed is by using the Python scripts
inside the `scripts` directory. They come alongside batch scripts for submission
to slurm, in order to make use of the full computational power of the cluster.
Use either the Python scripts (be careful with RAM and CPU cores usage!) or 
submit batch jobs using the batch scripts.

If you want to know more about one of the python scripts, use the `-h` flag.


## Milestones

You will notice that directories set up by the install script, the GitHub
milestones and the [ROADMAP](./ROADMAP.md) all contain three-digit numbers.
These numbers denote the "milestones" of this project. A milestone in the 
context of this project is a small task, usually consisting of a single type of 
plot to produce. Every milestone aims to answer a small scientific question: 
how is the temperture in halo gas distributed? How does the cool gas mass change
with halo mass? 

The milestones are documented in the [ROADMAP](./ROADMAP.md). Here you can find
a comprehensive list of all milestones, the question they seek to answer and a
short result. If you run scripts from the `scripts` directory, the output plots
will be automatically sorted into the corresponding `figures/XYZ` directory of
the milestone they belong to. 


## Organization

The repository is organized into the following directories:

- `notebooks`: The notebooks directory contains Jupyter notebooks. The notebooks
  contain primarily test code, some on-the-side experiments and probing plots
  (that is, plots that are meant to get an overview over simulation data).
- `scripts:` The scripts directory contains executable Python scripts that
  can be used to create plots. It also contains batch job scripts for use with
  slurm. The directory is organized into subdirectories. These subdirectories 
  are numbered and every number corresponds to one of the "milestones" of the 
  thesis project. To find out what these numbers mean and what each milestone 
  is about, consult either the [ROADMAP](./ROADMAP.md) or the GitHub milestones. 
- `src`: The source directory contains re-usable code, organized into modules
  and packages. These code snippets are the core of the project. All data
  loading, processing and plotting is done somewhere inside this code.
- `src.processors`: The processors package contains modules which in turn
  contain classes that are responsible for doing the bulk of the work: They
  load, process and plot data, all in one. The scripts inside the `scripts`
  directory do little more than instantiate one of these processor classes and
  call their methods in the correct order to produce the desired output.
  Wherever possible, re-usable code is factored out of these classes into the 
  `src` directory or the base processor class.

Alongside these directories tracked by the VC, the install script will also
create directories that will be populated by the Python and/or batch scripts:

- `data`: The data directory holds processed data produced by the scripts. It
  is organised into milestones.
- `external`: The external directory is used to install external dependencies
  locally, most noticeably the Illustris Python helper package.
- `figures`: The figures directory holds the finished plots produced by the
  scripts. It is organised into milestones.

### Where do I find...?

The stucture of the repository might be a little confusing until one has 
familiarized oneself with it. If you just want to find something specific 
quickly, you can find some guidance here:

- **Re-usable code snippets:** You are most likely to find code that you might 
  wish to re-use inside the `src` directory. The modules and packages therein 
  are more or less intuitively named. Look for the module/package that closest 
  describes what you are looking for!
- **Batch scripts for slurm:** Batch scripts for the different tasks are 
  situated in the `scripts` directory. Consult the [ROADMAP](./ROADMAP.md) to
  find out what job you are looking for and then select the correspond subdir
  of `scripts`. 
- **Scripts to reproduce plots:** Use the Python scripts under `scripts`.
  Consult the [ROADMAP](./ROADMAP.md) to find out which of the subdirectories
  you need to look into. All the Python scripts in `scripts` have a CLI, so
  to find out how to use them, simply run `python <script name>.py -h`.
- **Code for topic X:** If you are looking for code (or output) of a specific
  topic, consult the [ROADMAP](./ROADMAP.md) and see if you can find it. Then
  go to the subdirectory of `scripts` of the corresponding milestone and look
  at the content of the Python scripts there. They will likely point you to a
  class inside the `processors` package. From here, you can find all the code
  related to the task of that processor and milestone.
- **The finished plots:** If you installed the code using the `install.sh`
  script, you will find the figures under the `figures` directory in the
  subdirectory of the milestone they belong to. If you cannot find them there,
  check the [`config`](./src/config.py) module for the `FIGURES_HOME` path.
  You will find your figures under this path.


## Metadata

**Author:** Milan Staffehl
