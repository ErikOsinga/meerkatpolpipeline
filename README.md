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





