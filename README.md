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

The meerkat pol pipeline should automatically find the running server and report its results to the prefect server. Check out the results at http://localhost:4200/runs


## Running the pipeline

Simply call the pipeline as follows:

```
python meerkatpolpipeline/flows/Lband_reduction.py --cli-config-file ./meerkatpolpipeline/tests/temp_sample_configuration.yaml --working-dir ./meerkatpolpipeline/tests/temp
```

using a configuration file that you can create starting from the example configuration file.

### Recommended preamble to the script
A summary of environment variables to set to make the pipeline behave well
```
# set max log size to 3 MB
export PREFECT_LOGGING_TO_API_MAX_LOG_SIZE=3000000
# use Agg backend for MPL
export MPLBACKEND=Agg

# start pipeline
python meerkatpolpipeline/flows/Lband_reduction.py \
  --cli-config-file ./my_lband_config.yaml \
  --working-dir /data2/osinga/my_target1/ \
  --append-to-flowname "target1"

# --append-to-flowname is handy to keep track of different runs.
```



## Remote prefect server
If you want to log to a remote prefect server.

You can optionally start a Prefect server on another machine (e.g. some perpetually available VM). 
Let's say it's accessible at user@remoteaddress.com.

After starting a server on that machine, you can make sure the pipeline communicates with it as follows:

```
echo "Opening SSH tunnel to prefect server host "
# open connection
ssh -fNT -L 127.0.0.1:4200:localhost:4200 user@remoteaddress.com

# set which port to communicate results to 
export PREFECT_API_URL="http://localhost:4200/api"
```





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
```
export MPLBACKEND=Agg
```
in the environment where you call the pipeline. 


### 2. Caracal logs too large for Prefect
You can set the max log size larger with the following env var, 3 MB should be fine for caracal logs

```
export PREFECT_LOGGING_TO_API_MAX_LOG_SIZE=3000000
```



### 3. Caracal crashes in G-step (19) on gain calibrator
with 
```
Caught exception: Table DataManager error: Invalid operation: TSM: no array in row 267281 of column MODEL_DATA in /stimela_mount/msdir/mcxcJ1358_uncalibrated_preprocessed-cal.ms/table.f25
```

I'm not sure what's causing this as previous gaincal steps do work and there seems to be nothing writing to the MODEL_DATA column in between the steps, but I found that if this happens, it's because of the `crystalball_skymodel: true` setting. Try using the other config template [caracal-polcal-strategy-no-crystalbal.yml](https://github.com/ErikOsinga/meerkatpolpipeline/blob/main/templates/caracal-polcal-strategy-no-crystalbal.yml) that has this set to False. Note that the skymodel might be (?) less accurate then. See also [this caracal issue](https://github.com/caracal-pipeline/caracal/issues/1408)





