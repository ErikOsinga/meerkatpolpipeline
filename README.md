# meerkatpolpipeline
MeerKAT polarisation data reduction and analysis


Assuming Python 3.11 for the bookkeeping scripts in this directory, though most actual software is run in Singularity containers.

## Running the pipeline
[TODO]

e.g.
```
python meerkatpolpipeline/flows/Lband_reduction.py --cli-config-file ./meerkatpolpipeline/tests/temp_sample_configuration.yaml --working-dir ./meerkatpolpipeline/tests/temp
```

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

The meerkat pol pipeline should automatically find the running server and report its results to the prefect server. Check out the results at http://localhost:4200/runs

### Remote prefect server 






## Common problems / errors

### 1. Runtime error: main thread is not in main loop
```
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7f61f06b59e0>
Traceback (most recent call last):
  File "/net/lofar4/data2/osinga/software/miniconda/installation/envs/meerkatpol/lib/python3.11/tkinter/__init__.py", line 410, in __del__
    if self._tk.getboolean(self._tk.call("info", "exists", self._name)):
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: main thread is not in main loop
Exception ignored in: <function Variable.__del__ at 0x7f61f06b59e0>
```
Likely an issue with Prefect+Tkinter having bad interactions. According to an LLM:
> Tk/Tcl requires all create/destroy calls to happen on the main thread. When the task exits, Tk objects are finalized from a non-main thread → “main thread is not in main loop” spam → Tcl_AsyncDelete: async handler deleted by the wrong thread → hard abort/core dump.



Try setting 
`export MPLBACKEND=Agg`
in the environment where you call the pipeline. 


### 2. Caracal logs too large for Prefect
You can set the max log size larger with the following env var, 3 MB should be fine for caracal logs
```export PREFECT_LOGGING_TO_API_MAX_LOG_SIZE=3000000``
