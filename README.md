# meerkatpolpipeline
MeerKAT polarisation data reduction and analysis pipeline. 


This pipeline supports a range of functionalities, including downloading uncalibrated data, cross-calibration, verification of results against known calibrator models, (DI/DD) self-calibration, polarization cube imaging, RM synthesis, and the creation of preliminary science plots (*work in progress*).


The pipeline uses [Prefect](https://www.prefect.io/) to track task dependencies and logs.

<img width="1095" height="798" alt="example-pipeline-dashboard" src="https://github.com/user-attachments/assets/44fd8df2-3d08-487c-95c7-53a9af30e247" />




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

You will have to install CARAcal separately, see https://github.com/caracal-pipeline/caracal and https://caracal.readthedocs.io/en/stable/caracalREADME.html

For DI/DD self-calibration you will also need to have a singularity container with the LOFAR software. I recommend those built by Frits Sweijen, download them at https://tikk3r.github.io/flocs/.


## Running a Prefect server

### Local prefect server
The simplest thing is to start a local prefect server (e.g. open a second terminal window):

```
conda activate meerkatpol

prefect server start

```
which will start a server accessible at http://localhost:4200/

The meerkat pol pipeline should automatically find the running server and report its results to the prefect server. Check out the results at http://localhost:4200/runs


## Running the pipeline

Simply call the pipeline as follows:

```
python meerkatpolpipeline/flows/Lband_reduction.py --cli-config-file ./meerkatpolpipeline/templates/pipeline-template-config.yaml --working-dir ./path/to/workdir/
```

using a configuration file that you can create starting from the example configuration file. You can find the example configuration file in `./templates/pipeline-template-config.yaml`

### Recommended preamble to the script
A summary of environment variables to set to make the pipeline behave well
```
# set max log size to 3 MB
export PREFECT_LOGGING_TO_API_MAX_LOG_SIZE=3000000
# use Agg backend for MPL
export MPLBACKEND=Agg

# set max open file limit high for polarisation cubes
ulimit -n 9000

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


### 4. Illegal instruction in RM-tools (SIGILL)

It could be that the hardware you're trying to run on is not agreeing with the FFT instructions. Try downgrading finufft, e.g.

```
uv pip install finufft==2.2.0
```


