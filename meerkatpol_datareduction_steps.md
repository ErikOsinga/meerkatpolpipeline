# Script directory

`DIR=/net/rijn9/data2/osinga/meerkatBfields/Abell754/bologna/all_scripts`


## 1. Download uncalibrated data from SARAO:

- flags: static, cam, data_lost, ingest_rfi
- clip 163 3885
- quack 1
- f true
- t
- a

> **Note #1:**

    You can achieve this configuration by setting "default calibration", then remove "all" from --applycal and adding -t to tarball)

> **Note #2:**

    You can use mvf_download.py to download the rdb from SARAO, then run mvftoms.py locally to generate the desired ms with/without calibration applied (it requires katdal, https://github.com/ska-sa/katdal/), example:
    mvf_download.py rdblink .
    mvftoms.py -o Bullet_L_1529816457.MS -f --flags static,cam,data_lost,ingest_rfi -t -a -C 163,3885 --quack 1 1529816457/1529816457_sdp_l0.full.rdb

> **Note #3:**

    Or, see example $DIR/1_download_measurementset/download.sh

## 2. Run CARACAL:

- Install caracal (https://github.com/caracal-pipeline/caracal). See following link for installation instruction: https://pine-mandrill-ced.notion.site/How-to-install-the-github-version-of-caracal-2326498fafa18077b4a0fb76e0fd22cf
- Use the strategies (they are different for UHF and L band) in $DIR/2_caracal/*yml
- Run CARACAL: $DIR/2_caracal/run_caracal.sh
- In the output directory, check diagnostic_plots (!)


## 3. Check polarization calibrators (3C138 or 3C286):

- Split the polcal: $DIR/3_check_calibrator/split3c286.sh
- Image the polcal with wsclean: $DIR/3_check_calibrator/go_wsclean.sh
- Plot results vs model: $DIR/3_check_calibrator/go_processfield.sh

    # CHECKED: RMion correction should be negative for southern hemisphere

> **Note #1:**

    This step could be easily automated


## 4. Run facetselfcal:

- Install facetselfcal (https://github.com/rvweeren/lofar_facet_selfcal)
- Do the channel trimming, and flagging (no further flagging should be done for polarization): $DIR/4_facetselfcal/step1_facetselfcal_clip_and_flag_*.sh
- Do a DI run: $DIR/4_facetselfcal/step2_facetselfcal_DI_*.sh
- Do a DD run: $DIR/4_facetselfcal/step3_facetselfcal_DD_*.sh
- Extract: $DIR/4_facetselfcal/step3_facetselfcal_extract_*.sh
- Eventually do a new DI or DD on the extracted field repeating the steps above

> **Note #1:**

    Merge the steps?


## 5. IUQV cubes 12 channels:

- Make image with L+UHF: $DIR/5_iquv_cubes_12chan/step1_go_wsclean_combined12chan.sh
- Make image with single band (L|UHF): $DIR/5_iquv_cubes_12chan/step1_go_wsclean_combined12chan.sh
- Create catalog with pyBDSF: $DIR/5_iqu_cubes_12chan/step2_go_pydsf.sh

> **Note #1:**

    Be aware of the input ms in step1_go_wsclean_combined12chan.sh, depending if you want to images the band independently or jointly

> **Note #2:**

    Trim the edges of the catalog to avoid warning while running rms1d

> **Note #3:**

- no-mf-weighting \ should be used in wsclean when single channel images should be used for science


## 6. Check IUQV cubes vs NVSS:

- Run the script $DIR/6_check_iquv_vs_nvss/go_processfield_integrated.sh

> **Note #1:**

    This step may be redundant if the channels to flag are always the same

> **Note #2:**

    The script is still quite hardcoded, but this step may not be necessary in the future


## 7. Resample the frequency axis:

- Run $DIR/7_grid_frequency_axis/apply_cvel2.py

> **Note #1:**

    Make the width an argument + read starting frequency from the ms (the argument band will disappear)

> **Note #3:**

    This step needs to be done only if L is combined with UHF


## 8. IQU cubes for RMS:

- Make image with wsclean, see $DIR/8_create_iqu_cubes/step1_compute_wsclean_command.py
- Plot beam vs frequency: $DIR/8_create_iqu_cubes/step2_go_beam_vs_freq.sh
- Convolve channels at the same resolution: $DIR/8_create_iqu_cubes/step3_convolve.sh

> **Note #1:**

    Write a python script that edits the wsclean command of step 5 adding the right number of channels to obtain the desired channel width

> **Note #2:**

    Improve step3_convolve.sh, it can be a python script?

> **Note #3:**

    Again make sure -no-mf-weighting \ is turned on in the WSCLEAN command

## 9. RMS1D:

- Clone the POSSUM_pipeline repository (private) and RMtools (https://github.com/CIRADA-Tools/RM-Tools)
- Recompute channels to be flaged to provide as input in create_datacube.py (called by combine_cubes_and_rmsynth1d.sh)
- Run creating flagged cubes+RMS: $DIR/9_rmsynth1d/combine_cubes_and_rmsynth1d.sh

> **Note #1:**

    Check the parset config_rmsynth_1d.ini, especially: input_catalogue and snr_lowlim, note that the string sourcelist_filename should contain the same number of snr_lowlim (!)


## 10. Inspect RMS1D:

- Run $DIR/10_validate_rmsynth1d/go_validation.sh
- See if more channels need to be flagged, if so, rerun from step 9
- Compare the resolved sources to the single component sources (using the S_code of the pyBDSF catalog)

## 11. RMS3D:

- Run RMS3d: $DIR/11_rmsynth3d/go_rmsynth3d.sh

> **Note #1:**

    Check phimax and dPhi, they depend if you use a single band or L+UHF


## 12. Inspect and validate RMS3D:

- Mask low pixels  using the snrpi.fits image (e.g. mask pixels with SNR<8).
- Ricean bias correction
- Use the S_code = M from pyBDSF to select only extended sources
- Plot RM map for each extended source, with RM histogram (e.g. Vacca+12)

> **Note #1:**

    Mostly to-do.


## 13. Science:

- Correct for Galactic RM: $DIR/13_science/add_galactic_rm_hutschenreuter.py
- Make RM bubble plot: $DIR/13_science/go_bubble.sh
- Make voronoi map $DIR/13_science/voronoi_rm_plot.py
- 
