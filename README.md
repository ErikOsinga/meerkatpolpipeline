# meerkatpolpipeline
MeerKAT polarisation data reduction and analysis


Assuming Python 3.11 for the bookkeeping scripts in this directory, though most actual software is run in Singularity containers.

## Installation

Install e.g. with

```
conda create -n meerkatpol python=3.11

conda activate meerkatpol

conda install conda-forge::uv

git clone git@github.com:ErikOsinga/meerkatpolpipeline.git

cd meerkatpolpipeline

uv pip install .
```

or if you want an editable, dev tooled installation, replace the last line with

```
uv pip install -e .[dev]
```


## Running a Prefect server

### Local prefect server
The simplest thing is to start a local prefect server (e.g. open a second terminal)

```
conda activate meerkatpol

prefect server start

```
which will start a server accessible at http://localhost:4200/

The meerkat pol pipeline should automatically find the running server and report it's results to the prefect server. Check out the results at http://localhost:4200/runs

### Remote prefect server 



