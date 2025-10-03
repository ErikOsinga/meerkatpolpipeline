## Installation
We assume Python 3.11 (or later) for the pipeline, although most actual software is run in Singularity containers.

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