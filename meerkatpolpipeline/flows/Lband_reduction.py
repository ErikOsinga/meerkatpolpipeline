"""A prefect based pipeline that:
- will perform meerKAT data processing
- given an input strategy file
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from configargparse import ArgumentParser
from prefect import flow, tags, task  #, unmapped
from prefect.logging import get_run_logger

from meerkatpolpipeline.caracal import _caracal
from meerkatpolpipeline.casacrosscal import casacrosscal
from meerkatpolpipeline.check_calibrator.check_calibrator import (
    check_calibrator,
    image_gaincal,
    image_primary,
)
from meerkatpolpipeline.check_nvss.compare_to_nvss import compare_to_nvss
from meerkatpolpipeline.configuration import (
    Strategy,
    get_options_from_strategy,
    load_and_copy_strategy,
    log_enabled_operations,
)
from meerkatpolpipeline.cube_imaging.beam_info_vs_freq import generate_beam_plots
from meerkatpolpipeline.cube_imaging.combine_to_imagecube import combine_to_cube
from meerkatpolpipeline.cube_imaging.convolve_beam import BeamParams, convolve_images
from meerkatpolpipeline.cube_imaging.cube_imaging import (
    create_stokesIQU_cube_channels_from_imagelists,
    go_wsclean_cube_imaging_target,
)
from meerkatpolpipeline.download.clipping import copy_and_clip_ms
from meerkatpolpipeline.download.download import download_and_extract
from meerkatpolpipeline.measurementset import load_field_intents_csv, msoverview_summary
from meerkatpolpipeline.rmsynth.rmsynth1d import run_rmsynth1d
from meerkatpolpipeline.sclient import run_singularity_command
from meerkatpolpipeline.selfcal import _facetselfcal
from meerkatpolpipeline.utils.utils import (
    execute_command,
    filter_sources_within_radius,
    find_calibrated_ms,
    get_fits_image_center,
)
from meerkatpolpipeline.validation import (
    flag_image_freqs,
    flagstat_vs_freq,
    rms_vs_freq,
    validate_field,
)
from meerkatpolpipeline.wsclean.wsclean import (
    get_imset_from_prefix,
    get_pbcor_mfs_image_from_imset,
    remove_mfs_image_from_imset,
)


@flow(name="MeerKAT pipeline", log_prints=True)
def process_science_fields(
    strategy: Strategy,
    working_dir: Path
) -> None:
    """
    Flow that will execute all the enabled steps as tasks. 

    Each task will be done in a subdirectory
    """

    logger = get_run_logger()

    enabled_operations = log_enabled_operations(strategy)

    lofar_container = Path(strategy['lofar_container'])
    casa_container = Path(strategy['casa_container'])

    # check for additional bind directory
    if 'casa_additional_bind' in strategy:
        casa_additional_bind = strategy['casa_additional_bind']
        if isinstance(casa_additional_bind, str):
            casa_additional_bind = [casa_additional_bind]
        casa_additional_bind = [Path(path) for path in casa_additional_bind]
    else:
        casa_additional_bind = []

    ########## step 1: download & clip channels ##########
    if "download_preprocess" in enabled_operations:
        download_options = get_options_from_strategy(strategy, operation="download_preprocess")
        download_workdir = working_dir / "download"

        # create subdirectory 'download'
        download_workdir.mkdir(exist_ok=True) # runs can be repeated

        # First check if the preprocessed MS already exists. If so, we can skip the whole step.
        ms_path = download_workdir / download_options['ms_name']
        preprocessed_ms = ms_path.parent / f"{ms_path.stem}_preprocessed.ms"
        if preprocessed_ms.exists():
            logger.info(f"Preprocessed MS already exists at {preprocessed_ms}, skipping download_preprocess step.")

        else: 

            #### 1.1 download and extract
            task_start_download = task(download_and_extract, name="download_and_preprocess")
            ms_path = task_start_download(download_options, working_dir=download_workdir)

            # get MS summary
            task_msoverview_summary = task(msoverview_summary, name="msoverview_summary")
            ms_summary = task_msoverview_summary(
                binds=[str(ms_path.parent)],
                container=lofar_container,
                ms=ms_path,
                output_to_file= download_workdir / "msoverview_summary.txt",
            )

            #### 1.2 parang correction

            # grab the script from the meerkatpolpipeline package
            from meerkatpolpipeline.download import download  # cant import casa scripts
            parang_script = Path(download.__file__).parent / "go_correct_parang.py"
                        
            cmd_parang = f"""python {parang_script} \
                --running-inside-sing \
                --test-already-done \
                {ms_path}
            """

            task_parang_correction = task(run_singularity_command, name="run_parang_correction")
            task_parang_correction(
                lofar_container,
                cmd_parang,
                bind_dirs=[ms_path.parent,parang_script.parent],
                max_retries=1
            )

            logger.info(f"Preprocessed MS does not yet exist at {preprocessed_ms}, will copy and potentially clip channels from {ms_path}.")
            
            task_copy_and_clip_ms = task(copy_and_clip_ms, name="copy_and_clip_ms")
            task_copy_and_clip_ms(
                ms_path=ms_path,
                output_ms=preprocessed_ms,
                ms_summary=ms_summary, 
                clip_assumed_nchan=download_options['clip_assumed_nchan'],
                clip_chan_start=download_options['clip_chan_start'],
                clip_chan_end=download_options['clip_chan_end'],
                casa_container=casa_container,
                bind_dirs = [ms_path.parent, preprocessed_ms.parent] + casa_additional_bind
            )

        logger.info("Download and preprocessing step completed.")
        logger.info(f"Preprocessed MS can be found at {preprocessed_ms}")

        # TODO: clean up .tar.gz file if user requests? and maybe also the original MS?

    else:
        logger.warning("Download step is disabled, skipping download and preprocessing.")
        ms_path = download_workdir / download_options['ms_name']
        preprocessed_ms = ms_path.parent / f"{ms_path.stem}_preprocessed.ms"
        logger.info(f"Assuming preprocessed MS is available at {preprocessed_ms}. If not, please enable the download step in the strategy file.")


    ########## step 2: cross-calibration with either casa or caracal ##########
    if "crosscal" in enabled_operations:
        crosscal_base_dir = working_dir / "crosscal" # /caracal or /casacrosscal
        crosscal_base_dir.mkdir(exist_ok=True) # runs can be repeated

        field_intents_csv = crosscal_base_dir / "field_intents.csv"

        logger.info("Cross-calibration step is enabled, starting cross-calibration.")

        crosscal_options = get_options_from_strategy(strategy, operation="crosscal")


        # get MS summary, optionally with scan intents if user wants auto determined calibrators
        task_msoverview_summary = task(msoverview_summary, name="msoverview_preprocessed")
        ms_summary = task_msoverview_summary(
            binds=[str(preprocessed_ms.parent)],
            container=lofar_container,
            ms=preprocessed_ms,
            output_to_file= crosscal_base_dir / "msoverview_summary.txt",
            get_intents=crosscal_options["auto_determine_obsconf"]
        )
        if not crosscal_options['auto_determine_obsconf']:
            # then the calibrators should be set by the user. Write to an intent file
            logger.info(f"Auto-determination of obsconf is disabled, writing user-defined field intents to {field_intents_csv}.")
            _caracal.write_crosscal_csv(crosscal_options, ms=preprocessed_ms, output_path=field_intents_csv)

        logger.info(f"{ms_summary=}")
        

        # Then do specifically caracal or casa crosscal
        
        ############ 2. option 1: caracal cross-calibration step ############
        if crosscal_options['which'] == 'caracal':
            logger.info("Caracal cross-calibration step is enabled, starting caracal cross-calibration.")
            crosscal_dir = crosscal_base_dir / 'caracal'
            crosscal_dir.mkdir(exist_ok=True) # runs can be repeated


            # set up tasks
            task_cleanup_caracal = task(_caracal.cleanup_caracal_run, name="cleanup_caracal_run")
            task_caracal_crosscal = task(_caracal.do_caracal_crosscal, name="caracal_crosscal")

            # Check if caracal was already done but maybe files were not moved from /download to /caracal
            calibrated_cal_ms = find_calibrated_ms(
                crosscal_base_dir.parent,
                preprocessed_ms,
                look_in_subdirs=[Path('download')],
                suffix="-cal.ms"
            )
            calibrated_target_ms = find_calibrated_ms(
                crosscal_base_dir.parent,
                preprocessed_ms,
                look_in_subdirs=[Path('download')],
                suffix=f"-{strategy['targetfield']}-corr.ms"
            )
            if calibrated_cal_ms is not None and calibrated_target_ms is not None:
                # if both are found, we assume the caracal run was already done
                logger.info(f"Found already calibrated target MS at {calibrated_target_ms}. Moving to /crosscal/caracal/ directory.")
                
                # move calibrated MS & caracal output from "$workdir/download/" to "$workdir/crosscal/caracal/" 
                calibrated_cal_ms, calibrated_target_ms = task_cleanup_caracal(
                    caracal_rundir=calibrated_target_ms.parent,
                    preprocessed_ms_name=preprocessed_ms.stem,
                    calibrated_cal_ms=calibrated_cal_ms,
                    calibrated_target_ms=calibrated_target_ms,
                    output_dir = (crosscal_base_dir / 'caracal')
                )


            else:
                # Do the actual caracal run.

                # note: this task also tests whether the calibrated MS exists in the crosscal_base_dir/caracal_crosscal directory
                calibrated_cal_ms, calibrated_target_ms = task_caracal_crosscal(
                    crosscal_options,
                    preprocessed_ms,
                    crosscal_base_dir,
                    ms_summary,
                    lofar_container # only required if user overwrites input MS.
                )

                if calibrated_cal_ms is None or calibrated_target_ms is None:
                    raise ValueError(
                        "Caracal cross-calibration did not return valid calibrated MS paths. Please check the caracal logs."
                    )
            
                logger.info(f"Caracal cross-calibration completed. Calibrated MS can be found at {calibrated_target_ms} and {calibrated_cal_ms}.")
                # move calibrated MS & caracal output from "$workdir/download/" to "$workdir/crosscal/caracal/" 
                calibrated_cal_ms, calibrated_target_ms = task_cleanup_caracal(
                    caracal_rundir=calibrated_target_ms.parent,
                    preprocessed_ms_name=preprocessed_ms.stem,
                    calibrated_cal_ms=calibrated_cal_ms,
                    calibrated_target_ms=calibrated_target_ms,
                    output_dir = (crosscal_base_dir / 'caracal')
                )


        ############ 2. option 2: casa cross-calibration step ############
        elif crosscal_options['which'] == 'casacrosscal':
            logger.info("Casa crosscal step is enabled, starting casa cross-calibration.")
            crosscal_dir = crosscal_base_dir / 'casacrosscal'
            crosscal_dir.mkdir(exist_ok=True) # runs can be repeated

            task_casa_crosscal = task(casacrosscal.do_casa_crosscal, name="casa_crosscal")
            calibrated_cal_ms, calibrated_target_ms = task_casa_crosscal(
                crosscal_options, 
                preprocessed_ms, 
                crosscal_dir, 
                ms_summary,
                casa_container,
                lofar_container,
                bind_dirs = [
                    preprocessed_ms.parent, # input MS location
                    crosscal_dir # output MS location
                ] + casa_additional_bind # any additional bindings
            )




        ############ 2. option 3: oopsie, user has made a mistake ############
        else:
            logger.error(f"Invalid crosscal option '{crosscal_options['which']}'. Expected 'caracal' or 'casacrosscal'.")
            raise ValueError(f"Invalid crosscal option '{crosscal_options['which']}'. Expected 'caracal' or 'casacrosscal'.")

    else:
        logger.warning("Crosscal is disabled. Skipping cross-calibration.")
        crosscal_dir = None

    if crosscal_dir is None:
        crosscal_base_dir = working_dir / "crosscal" # /caracal or /casacrosscal
        logger.warning(f"No cross-calibration step was performed, checking for calibrated MS in {crosscal_base_dir} subdirectories")
        
        calibrated_cal_ms = find_calibrated_ms(
            crosscal_base_dir,
            preprocessed_ms,
            suffix="-cal.ms"
        )
        calibrated_target_ms = find_calibrated_ms(
            crosscal_base_dir,
            preprocessed_ms,
            suffix="-corr.ms"
        )
    
    
        if calibrated_cal_ms is None or calibrated_target_ms is None:
            raise ValueError(
                f"No calibrated target/cal measurement set found in {crosscal_base_dir}. Please enable either caracal or casacrosscal step in the strategy file."
            )
        logger.info(f"Found calibrated target MS at {calibrated_target_ms} and calibrated cal MS at {calibrated_cal_ms}.")
        logger.info("Assuming field intents CSV is available in the crosscal directory. If not, please enable crosscal step.")

        # set crosscal dir wherever the calibrated MS was found
        crosscal_dir = calibrated_target_ms.parent

    ########## step 3: check polarisation calibrator ##########
    if "check_calibrator" in enabled_operations:
        check_calibrator_workdir = working_dir / "check_calibrator"
        check_calibrator_workdir.mkdir(exist_ok=True)

        check_calibrator_options = get_options_from_strategy(strategy, operation="check_calibrator")

        # get polcal field from field intents
        field_intents_dict = load_field_intents_csv(field_intents_csv)
        _, polcal_field = _caracal.obtain_by_intent(field_intents_dict, "polcal")
        
        # check for user overwrite
        if check_calibrator_options['crosscal_ms'] is None:
            check_calibrator_options['crosscal_ms'] = calibrated_cal_ms
        if check_calibrator_options['polcal_field'] is None:
            check_calibrator_options['polcal_field'] = polcal_field

        # split calibrator, make images, and validation plots
        task_check_calibrator = task(check_calibrator, name="check_calibrator")
        task_check_calibrator(
            check_calibrator_options,
            working_dir=check_calibrator_workdir,
            casa_container=casa_container,
            lofar_container=lofar_container,
            bind_dirs = [
                check_calibrator_options['crosscal_ms'].parent, # input MS location
                check_calibrator_workdir # output MS location
            ] + casa_additional_bind # any additional bindings
        )

        if check_calibrator_options['image_gaincal']:

            _, gaincal_field = _caracal.obtain_by_intent(field_intents_dict, "gaincal")

            task_image_gaincal = task(image_gaincal, name="image_gaincal")
            task_image_gaincal(
                check_calibrator_options,
                gaincal_field=gaincal_field,
                working_dir=check_calibrator_workdir,
                casa_container=casa_container,
                lofar_container=lofar_container,
                bind_dirs = [
                    check_calibrator_options['crosscal_ms'].parent, # input MS location
                    check_calibrator_workdir # output MS location
                ] + casa_additional_bind # any additional bindings
            )

        if check_calibrator_options['image_primary']:

            _, primary_field = _caracal.obtain_by_intent(field_intents_dict, "fluxcal")

            task_image_primary = task(image_primary, name="image_primary")
            task_image_primary(
                check_calibrator_options,
                primary_field=primary_field,
                working_dir=check_calibrator_workdir,
                casa_container=casa_container,
                lofar_container=lofar_container,
                bind_dirs = [
                    check_calibrator_options['crosscal_ms'].parent, # input MS location
                    check_calibrator_workdir # output MS location
                ] + casa_additional_bind # any additional bindings
            )


    else:
        logger.warning("Check calibrator step is disabled, skipping checking of polarisation calibrator.")


    ########## step 4: facetselfcal ##########
    if "selfcal" in enabled_operations:

        selfcal_workdir = working_dir / "selfcal"
        selfcal_workdir.mkdir(exist_ok=True)

        # figure out where the calibrated target MS is and copy it to selfcal_workdir if needed
        target_ms = selfcal_workdir / calibrated_target_ms.name
        if not target_ms.exists():
            logger.info(f"Copying calibrated target MS from {calibrated_target_ms} to {target_ms}.")
            shutil.copytree(calibrated_target_ms, target_ms)


        #### 4.1: preprocess MS: get rid of timesteps with low number of baselines, do additional clipping and flagging, splitting irregular time axis

        selfcal_options = get_options_from_strategy(strategy, operation="selfcal")

        # start preprocessing
        task_facetselfcal_preprocess = task(_facetselfcal.do_facetselfcal_preprocess, name="facetselfcal_preprocess")
        all_preprocessed_mses = task_facetselfcal_preprocess(selfcal_options, target_ms, selfcal_workdir, lofar_container)
        logger.info(f"Preprocessing step completed. MSes found at {all_preprocessed_mses}")


        #### 4.2: DI self calibration
        DIcal_workdir = selfcal_workdir / "DIcal"
        DIcal_workdir.mkdir(exist_ok=True)
        task_facetselfcal_DIcal = task(_facetselfcal.do_facetselfcal_DI, name="facetselfcal_DI")
        mses_after_DIcal = task_facetselfcal_DIcal(selfcal_options, all_preprocessed_mses, DIcal_workdir, lofar_container)
        logger.info(f"DIcal step completed. MSes found at {mses_after_DIcal}")

        #### 4.3 DD self calibration
        DDcal_workdir = selfcal_workdir / "DDcal"
        DDcal_workdir.mkdir(exist_ok=True)
        task_facetselfcal_DDcal = task(_facetselfcal.do_facetselfcal_DD, name="facetselfcal_DD")
        mses_after_DDcal = task_facetselfcal_DDcal(selfcal_options, mses_after_DIcal, DDcal_workdir, lofar_container)
        logger.info(f"DDcal step completed. MSes found at {mses_after_DDcal}")

        #### 4.4 Extraction of target field
        # done in same directory as DDcal
        DDcal_workdir = selfcal_workdir / "DDcal"
        DDcal_workdir.mkdir(exist_ok=True)
        task_facetselfcal_extract = task(_facetselfcal.do_facetselfcal_extract, name="facetselfcal_extract")
        # note that inside do_facetselfcal_extract, we remove one .copy from the filename, since facetselfcal with
        # start !=0 will automatically use the .ms.copy files that should be there after DDcal. (see mses_after_DDcal)
        corrected_extracted_mses = task_facetselfcal_extract(selfcal_options, mses_after_DDcal, DDcal_workdir, lofar_container)
        logger.info(f"extract step completed. MSes found at {corrected_extracted_mses}")

        logger.info("All selfcal steps fully completed.")
    
    else:
        logger.warning("Selfcal step is disabled, trying to find extracted MSes...")
        
        selfcal_workdir = working_dir / "selfcal"
        DDcal_workdir = selfcal_workdir / "DDcal"
        
        logger.info(f"Checking for extracted MSes in {DDcal_workdir}")
        corrected_extracted_mses = list(sorted(DDcal_workdir.glob("[!plotlosoto]*.subtracted_ddcor")))
        assert len(corrected_extracted_mses) != 0, f"Found {len(corrected_extracted_mses)} mses in {DDcal_workdir}. However, expected at least one... Please enable selfcal step."



    ########## step 5: IQUV cube image 12 channel ##########
    cube_imaging_workdir = working_dir / "coarse_cube_imaging"
    cube_imaging_options = get_options_from_strategy(strategy, operation="coarse_cube_imaging")
    if 'coarse_cube_imaging' in enabled_operations:
        cube_imaging_workdir.mkdir(exist_ok=True)

        # Make a 12 channel cube of the target in IQU, from the extracted dataset for a quick look

        # check for user overwrite
        if cube_imaging_options['corrected_extracted_mses'] is None:
            cube_imaging_options['corrected_extracted_mses'] = corrected_extracted_mses

        # Make quicklook 12-channel cubes in IQU of the target field, with pb correction.
        task_image_smallcubes = task(go_wsclean_cube_imaging_target, name="wsclean_smallcubes_target")
        imageset_I, imageset_Q, imageset_U, imageset_I_mfs = task_image_smallcubes(
            ms = corrected_extracted_mses,
            working_dir = cube_imaging_workdir,
            lofar_container=lofar_container,
            cube_imaging_options=cube_imaging_options
        )

        if cube_imaging_options['run_pybdsf']:
            # run PYBDSF on the MFS image
            MFS_image = get_pbcor_mfs_image_from_imset(imageset_I_mfs)

            # pybdsf multiprocessing and prefect have some weird deadlock
            # run pybdsf in a separate process instead of a task
            from meerkatpolpipeline.utils import runpybdsf
            pybdsf_script = Path(runpybdsf.__file__).parent / "runpybdsf.py"
            pybdsf_cmd = f"python {pybdsf_script} {MFS_image} --outdir {cube_imaging_workdir}"
            execute_command(pybdsf_cmd, logfile=cube_imaging_workdir / 'pybdsf_logs.txt')

            # task_pybdsf = task(_runpybdsf, name="run_pybdsf_on_mfs")
            # sourcelist_fits, sourcelist_reg, rmsmap = task_pybdsf(
            #             outdir=cube_imaging_workdir,
            #             filename=MFS_image,
            #             adaptive_rms_box=True,
            #             logger=logger
            # )

            # Assuming pybdsf results in these files
            sourcelist_fits = cube_imaging_workdir / 'sourcelist.srl.fits'
            sourcelist_reg = cube_imaging_workdir / 'sourcelist.srl.reg'
            rmsmap = cube_imaging_workdir / 'rms_map.fits'

            # filter sources within a certain radius if requesteds
            if cube_imaging_options['filter_pybdsf_cat_radius_deg'] is not None:
                sourcelist_fits_filtered = sourcelist_fits.parent / f"sourcelist_filtered_within_{cube_imaging_options['filter_pybdsf_cat_radius_deg']}_deg.fits"
                sourcelist_reg_filtered = sourcelist_fits_filtered.parent / f"sourcelist_filtered_within_{cube_imaging_options['filter_pybdsf_cat_radius_deg']}_deg.reg"

                center_coord = get_fits_image_center(MFS_image)

                logger.info(f"Filtering PYBDSF source catalogue to only include sources within {cube_imaging_options['filter_pybdsf_cat_radius_deg']} deg of center {center_coord.to_string('hmsdms')}")

                filter_sources_within_radius(
                    sourcelist_fits,
                    regions_path=sourcelist_reg,
                    center_coord=center_coord,
                    radius_deg=cube_imaging_options['filter_pybdsf_cat_radius_deg'],
                    output_path=sourcelist_fits_filtered,
                    regions_output_path=sourcelist_reg_filtered,
                )

            else:
                sourcelist_fits_filtered = sourcelist_fits
                sourcelist_reg_filtered = sourcelist_reg

    else:
        wsclean_output_dir = cube_imaging_workdir / "IQUimages"
        logger.warning(f"Small cube imaging step is disabled, skipping small cube imaging. Looking for IQU cubes in {wsclean_output_dir}...")

        full_prefix = str(wsclean_output_dir / ( cube_imaging_options['targetfield']+'_stokesI') )

        imageset_I = get_imset_from_prefix(
            prefix=full_prefix,
            pol="i",
            validate=True,
            chanout=12, # default
            pbcor_done=True, # default
            can_be_pbcor = ["image"] # default, coarse_cube_imaging only does pbcor for 'image' files.
        )

        full_prefix = str(wsclean_output_dir / ( cube_imaging_options['targetfield']+'_stokesQU') )

        imageset_Q = get_imset_from_prefix(
            prefix=full_prefix,
            pol="q",
            validate=True,
            chanout=12, # default
            pbcor_done=True, # default
            can_be_pbcor = ["image"] # default, coarse_cube_imaging only does pbcor for 'image' files.
        )
        
        imageset_U = get_imset_from_prefix(
            prefix=full_prefix,
            pol="u",
            validate=True,
            chanout=12, # default
            pbcor_done=True, # default
            can_be_pbcor = ["image"] # default, coarse_cube_imaging only does pbcor for 'image' files.
        )

        # TODO: how to deal with pybdsf catalogue filtering if this step is disabled?


    if not sourcelist_fits.exists() or not sourcelist_reg.exists() or not rmsmap.exists(): # will fail if coarse cube imaging is disabled
        msg = f"PYBDSF results not found in {cube_imaging_workdir}, please check if PYBDSF was run correctly. Expected {sourcelist_fits}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # TODO: switch step 6 and 7, attempt to compare NVSS to the top_n_spectra, after we determine these automatically.
    ########## step 6: preliminary check of IQUV cubes vs NVSS ##########
    if "compare_to_nvss" in enabled_operations:
        nvss_comparison_workdir = working_dir / "nvss_comparison"
        nvss_comparison_workdir.mkdir(exist_ok=True)
        
        logger.info("Starting compare_to_nvss step")

        nvss_comparison_options = get_options_from_strategy(strategy, operation="compare_to_nvss")

        # compare to NVSS
        task_compare_nvss = task(compare_to_nvss, name="compare_to_nvss")
        task_compare_nvss(
            nvss_comparison_options,
            nvss_comparison_workdir,
            imageset_I=imageset_I,
            imageset_Q=imageset_Q,
            imageset_U=imageset_U
        )

    
    ########## step 7: after we've run PYBDSF on small cube, can check spectra of brightest sources ##########
    if "validation" in enabled_operations:
        validate_field_workdir = working_dir / "validation"
        validate_field_workdir.mkdir(exist_ok=True)

        validation_options = get_options_from_strategy(strategy, operation="validation")

        logger.info("Starting validate field step")

        check_spectra_task = task(validate_field.plot_top_n_source_spectra, name="check_spectra")
        spectra_check_workdir = validate_field_workdir / "spectra_check"
        spectra_check_workdir.mkdir(exist_ok=True)

        check_spectra_task(
            imageset_I,
            imageset_Q,
            imageset_U,
            sourcelist_reg_filtered,
            sourcelist_fits_filtered,
            output_prefix=spectra_check_workdir / "top_n_spectra",
            validation_options=validation_options,
            logger=logger
        )

        # check flagging percentage
        compute_flagstat_task = task(flagstat_vs_freq.compute_flagstat_vs_freq, name="compute_flagstat_vs_freq")

        # NOTE: 746 MHz bandwidth in L-band typically, 12 channels is roughly 60 MHz channel
        future = compute_flagstat_task.submit(
            ms_paths=corrected_extracted_mses,
            bin_width_mhz=10, # MHz
            chunk_rows=4096, # default 
        )
        flag_results_per_ms, centers_mhz, avg_flag_pct, sum_counts = future.result()

        # Plotting flagging percentage per MS
        for ms, c_mhz, flag_pct, _ in flag_results_per_ms:
            out_path = validate_field_workdir / "plots" / f"flag_vs_freq_{ms.name}.png"
            flagstat_vs_freq.plot_flag_vs_freq(c_mhz, flag_pct, out_path, f"Flagging vs Frequency: {ms.name}")

        # Plotting average flagging vs freq
        avg_out = validate_field_workdir / "plots" / "flag_vs_freq_avg.png"
        # compute the frequency coverage of the coarse cube image.
        _, image_nu_Hz, _ = validate_field.sort_files_by_frequency(imageset_I.image_pbcor)
        image_nu_Hz = np.asarray(image_nu_Hz) # assumes Hz in header.
        # overlay the 12 channels from the coarse cube imaging, to see where they end up on the flagged spectrum
        flagstat_vs_freq.plot_flag_vs_freq(centers_mhz, avg_flag_pct, avg_out, "Flagging vs Frequency: AVERAGE", image_freqs_MHz=image_nu_Hz/1e6)

        # Then check spectra with flagging above a certain threshold
        check_spectra_task(
            imageset_I,
            imageset_Q,
            imageset_U,
            sourcelist_reg_filtered,
            sourcelist_fits_filtered,
            output_prefix=spectra_check_workdir / "top_n_spectra_flagged",
            validation_options=validation_options,
            centers_mhz=centers_mhz,
            avg_flag_pct=avg_flag_pct,
            mask_above_flag_threshold=validation_options['flag_threshold_pct'],
            logger=logger
        )



    ########## step 8: Resample the frequency axis in the MS if requested (required for L-band + UHF imaging) ##########
    # TODO

    ########## step 9: Many-channel IQU imaging ##########
    fine_cube_imaging_workdir = working_dir / "fine_cube_imaging"
    if "fine_cube_imaging" in enabled_operations:
        fine_cube_imaging_workdir.mkdir(exist_ok=True)

        fine_cube_imaging_options = get_options_from_strategy(strategy, operation="fine_cube_imaging")

        # check for user overwrite
        if fine_cube_imaging_options['corrected_extracted_mses'] is None:
            fine_cube_imaging_options['corrected_extracted_mses'] = corrected_extracted_mses
        
        fine_cube_imaging_options['also_image_for_mfs'] = False # no need to do MFS in fine cube imaging

        # Make many-channel cubes in IQU of the target field, with pb correction.
        task_image_finecubes = task(go_wsclean_cube_imaging_target, name="wsclean_finecubes_target")
        imageset_I_fine, imageset_Q_fine, imageset_U_fine, _ = task_image_finecubes(
            ms = fine_cube_imaging_options['corrected_extracted_mses'],
            working_dir = fine_cube_imaging_workdir,
            lofar_container=lofar_container,
            cube_imaging_options=fine_cube_imaging_options,
            finecube=True
        )

        # Remove MFS image from fine cube imageset since not needed
        imageset_I_fine = remove_mfs_image_from_imset(imageset_I_fine)
        imageset_Q_fine = remove_mfs_image_from_imset(imageset_Q_fine)
        imageset_U_fine = remove_mfs_image_from_imset(imageset_U_fine)

        # Plot beam vs frequency
        logger.info("Generating beam vs frequency plots after fine cube imaging...")
        # beam_limit_arcsec = 15 # 15 arcsec default beam limit. Can make user input
        beam_limit_arcsec = fine_cube_imaging_options['beam_limit_asec']
        task_beam_plots = task(generate_beam_plots, name="plot_beam_vs_freq")
        beamdata_i, beamdata_q = task_beam_plots(
            i_input=imageset_I_fine.image_pbcor,
            q_input=imageset_Q_fine.image_pbcor,
            output_dir=fine_cube_imaging_workdir / "beam_plots",
            yline_arcsec=beam_limit_arcsec,
            ylim_arcsec=None,
            show=False,
        )

        num_chan_largebeam_i = np.sum(beamdata_i.bmaj_deg > beam_limit_arcsec/3600.)
        logger.info(f"Number of channels in Stokes I with beam (major axis) larger than {beam_limit_arcsec} arcsec: {num_chan_largebeam_i} out of {len(beamdata_i.bmaj_deg)}")
        num_chan_largebeam_q = np.sum(beamdata_q.bmaj_deg > beam_limit_arcsec/3600.)
        logger.info(f"Number of channels in stokes Q with beam (major axis) larger than {beam_limit_arcsec} arcsec: {num_chan_largebeam_q} out of {len(beamdata_q.bmaj_deg)}")

        # perc_chan_largebeam_i = (num_chan_largebeam_i / len(beamdata_i.bmaj_deg)) * 100.
        # perc_chan_largebeam_q = (num_chan_largebeam_q / len(beamdata_q.bmaj_deg)) * 100.

        target_beam = BeamParams(
            bmaj_arcsec=beam_limit_arcsec,
            bmin_arcsec=beam_limit_arcsec,
            bpa_deg=0
        )
        
        # convolve stokes I to common resolution
        logger.info("Starting convolution of fine cube images to common resolution...")
        task_convolve_images = task(convolve_images, name="convolve_finecube_images")
        with tags("stokes-i"):
            stokesI_convolved_images: list[Path] = task_convolve_images(
                inputs=imageset_I_fine.image_pbcor,
                target_beam=target_beam,
                output_dir=fine_cube_imaging_workdir / "convolved_images",
                suffix_mode="beam",
                overwrite=False
            )

        with tags("stokes-q"):
            # convolve stokes Q to common resolution
            stokesQ_convolved_images: list[Path] = task_convolve_images(
                inputs=imageset_Q_fine.image_pbcor,
                target_beam=target_beam,
                output_dir=fine_cube_imaging_workdir / "convolved_images",
                suffix_mode="beam",
                overwrite=False
            )

        with tags("stokes-u"):
            # convolve stokes U to common resolution
            stokesU_convolved_images: list[Path] = task_convolve_images(
                inputs=imageset_U_fine.image_pbcor,
                target_beam=target_beam,
                output_dir=fine_cube_imaging_workdir / "convolved_images",
                suffix_mode="beam",
                overwrite=False
            )

        logger.info(f"Amount of images in Stokes I after convolution: {len(stokesI_convolved_images)}")
        logger.info(f"Amount of images in Stokes Q after convolution: {len(stokesQ_convolved_images)}")
        logger.info(f"Amount of images in Stokes U after convolution: {len(stokesU_convolved_images)}")

        # make sure we're building cubes consistently
        assert len(stokesI_convolved_images) == len(stokesQ_convolved_images) == len(stokesU_convolved_images), "Number of convolved images in Stokes I, Q, and U do not match!"

        # compute rms vs channel index after convolution
        task_compute_rms = task(rms_vs_freq.compute_rms_from_imagelist, name="compute_rms_after_convolution")
        with tags("stokes-i"):
            rms_per_I_image: np.ndarray = task_compute_rms(stokesI_convolved_images)
        with tags("stokes-q"):
            rms_per_Q_image: np.ndarray = task_compute_rms(stokesQ_convolved_images)
        with tags("stokes-u"):
            rms_per_U_image: np.ndarray = task_compute_rms(stokesU_convolved_images)

        # put all channel images into an StokesIQUCubeChannels object
        stokes_iqu_cube_channels = create_stokesIQU_cube_channels_from_imagelists(
            stokesI_convolved_images=stokesI_convolved_images,
            stokesQ_convolved_images=stokesQ_convolved_images,
            stokesU_convolved_images=stokesU_convolved_images,
            rms_per_I_channel=rms_per_I_image,
            rms_per_Q_channel=rms_per_Q_image,
            rms_per_U_channel=rms_per_U_image,
        )

        # plot rms vs channel index after convolution
        task_plot_rms = task(rms_vs_freq.plot_rms_vs_channel_from_imlist, name="plot_rms_after_convolution")
        with tags("stokes-i"):
            task_plot_rms(
                stokesI_convolved_images,
                rms_per_I_image,
                output_dir=fine_cube_imaging_workdir / "beam_plots",
                output_prefix="stokesI"
            )
        with tags("stokes-q"):
            task_plot_rms(
                stokesQ_convolved_images,
                rms_per_Q_image,
                output_dir=fine_cube_imaging_workdir / "beam_plots",
                output_prefix="stokesQ"
            )
        with tags("stokes-u"):
            task_plot_rms(
                stokesU_convolved_images,
                rms_per_U_image,
                output_dir=fine_cube_imaging_workdir / "beam_plots",
                output_prefix="stokesU"
            )


        # flag channels above a certain rms threshold
        rms_qu_average = (stokes_iqu_cube_channels.rms_per_Q_channel + stokes_iqu_cube_channels.rms_per_U_channel) / 2.
        # make sure these are indices of the original cube before any channels are removed !!
        bad_channel_indices = stokes_iqu_cube_channels.channel_numbers[rms_qu_average > fine_cube_imaging_options['channel_rms_limit_Jybeam']]
        logger.info(f"Number of channels flagged due to high RMS (> {fine_cube_imaging_options['channel_rms_limit_Jybeam']} Jy/beam) in Q/U after convolution: {len(bad_channel_indices)}")

        # also flag channels above a certain flag percentage
        flag_mask, _ = flag_image_freqs.flag_image_freqs(
            centers_mhz, avg_flag_pct, stokes_iqu_cube_channels.frequencies/1e6, # make sure both frequencies are in same units
            threshold_pct=fine_cube_imaging_options['channel_flag_limit_pct'],
            outside="extend"
        )
        logger.info(f"Number of channels flagged due to high flag percentage (> {fine_cube_imaging_options['channel_flag_limit_pct']} %) after convolution: {np.sum(flag_mask)}")
        # these can overlap, so combine them uniquely. Again make sure indices correspond to WSClean indexing before any channels are removed
        bad_channel_indices = np.unique( np.concatenate( (bad_channel_indices, stokes_iqu_cube_channels.channel_numbers[flag_mask]) ) )
        
        logger.info(f"Total number of channels flagged after convolution: {len(bad_channel_indices)} out of {len(stokesI_convolved_images)}")


        logger.info("Combining channel images to image cube...")
        # combine Stokes I images into image cubes, flagging bad channels
        task_combine_to_cube = task(combine_to_cube, name="combine_finecube_images_to_cubes")
        stokesIcube = fine_cube_imaging_workdir / "cubes" / f"{fine_cube_imaging_options['targetfield']}_stokes_i_pbcor_convolved.fits"
        with tags("stokes-i"):
            task_combine_to_cube(
                file_input=stokesI_convolved_images,
                reference_chan0=stokesI_convolved_images[0],
                output=stokesIcube,
                nchan=len(imageset_I_fine.image_pbcor), # note that we require nchan to be the original number of channels before convolution and flagging
                width_Mhz=fine_cube_imaging_options['chanwidth_MHz'],
                flag_chans=bad_channel_indices,
                overwrite=fine_cube_imaging_options['overwrite_cube'],
                logger=logger
            )

        # combine stokes Q images to cube
        stokesQcube = fine_cube_imaging_workdir / "cubes" / f"{fine_cube_imaging_options['targetfield']}_stokes_q_pbcor_convolved.fits"
        with tags("stokes-q"):
            task_combine_to_cube(
                file_input=stokesQ_convolved_images,
                reference_chan0=stokesQ_convolved_images[0],
                output=stokesQcube,
                nchan=len(imageset_Q_fine.image_pbcor), # note that we require nchan to be the original number of channels before convolution and flagging
                width_Mhz=fine_cube_imaging_options['chanwidth_MHz'],
                flag_chans=bad_channel_indices,
                overwrite=fine_cube_imaging_options['overwrite_cube'],
                logger=logger
            )

        # combine stokes U images to cube
        stokesUcube = fine_cube_imaging_workdir / "cubes" / f"{fine_cube_imaging_options['targetfield']}_stokes_u_pbcor_convolved.fits"
        with tags("stokes-u"):
            task_combine_to_cube(
                file_input=stokesU_convolved_images,
                reference_chan0=stokesU_convolved_images[0],
                output=stokesUcube,
                nchan=len(imageset_U_fine.image_pbcor), # note that we require nchan to be the original number of channels before convolution and flagging
                width_Mhz=fine_cube_imaging_options['chanwidth_MHz'],
                flag_chans=bad_channel_indices,
                overwrite=fine_cube_imaging_options['overwrite_cube'],
                logger=logger
            )

    else:
        # assume cubes have been built
        stokesIcube = fine_cube_imaging_workdir / "cubes" / f"{fine_cube_imaging_options['targetfield']}_stokes_i_pbcor_convolved.fits"
        stokesQcube = fine_cube_imaging_workdir / "cubes" / f"{fine_cube_imaging_options['targetfield']}_stokes_q_pbcor_convolved.fits"
        stokesUcube = fine_cube_imaging_workdir / "cubes" / f"{fine_cube_imaging_options['targetfield']}_stokes_u_pbcor_convolved.fits"

    
    ########## step 10: RM synthesis 1D ##########
    rmsynth1d_workdir = working_dir / "rmsynth1d"
    if "rmsynth1d" in enabled_operations:
        rmsynth1d_workdir.mkdir(exist_ok=True)

        rmsynth1d_options = get_options_from_strategy(strategy, operation="rmsynth1d")

        # check for user override of catalogue file
        if rmsynth1d_options['catalog_file'] is None:
            rmsynth1d_options['catalog_file'] = sourcelist_fits_filtered

        # Run RM synthesis in 1D on the image cubes
        task_rmsynth1d = task(run_rmsynth1d, name="rmsynth_1d")
        task_rmsynth1d(
            rmsynth1d_options,
            stokesI_cube_path=stokesIcube,
            rmsynth1d_workdir=rmsynth1d_workdir,
        )


    ########## step 11: Verify RMSynth1D ##########


    ########## step 12: RM synthesis 3D ##########


    ########## step 13: Verify RMSynth3D ##########


    ########## step 14: Science plots ##########

    
def setup_run(
    strategy_path: Path,
    working_dir: Path,
    append_to_flowname: str = ""
) -> None:
    
    # load strategy and copy timestamp to working dir
    strategy = load_and_copy_strategy(strategy_path, working_dir)

    # determine target from strategy file, in the caracal option
    target = strategy['targetfield']

    print(f"Starting pipeline for {target=}")

    # when testing without prefect
    # process_science_fields(
    #     strategy=strategy,
    #     working_dir=working_dir
    # )

    process_science_fields.with_options(
        name=f"MeerKAT pipeline - {target} {append_to_flowname}"
        # , task_runner=dask_task_runner
    )(
        strategy=strategy,
        working_dir=working_dir.resolve() # resolve in case relative path
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="MeerKAT L-band data processing")

    parser.add_argument(
        "--cli-config-file", type=str, help="Path to strategy configuration file"
    )
    parser.add_argument(
        "--working-dir", type=str, default="./", help="Path to main working directory. Default ./"
    )
    parser.add_argument(
        "--append-to-flowname", type=str, default="", help="String to attach to the flow name. Default ''"
    )

    return parser


def cli() -> None:

    parser = get_parser()

    args = parser.parse_args()

    setup_run(
        strategy_path=Path(args.cli_config_file),
        working_dir=Path(args.working_dir),
        append_to_flowname=args.append_to_flowname
    )


if __name__ == "__main__":
    cli()