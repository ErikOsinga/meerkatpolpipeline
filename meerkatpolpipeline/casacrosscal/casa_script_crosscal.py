"""

CASA SCRIPT. Can be run, for example, with

casa -c casa_script_crosscal.py \
  --calms ms_with_parangcorr-cal.ms \
  --scan-xcal "12,22"


Adapted from the MeerKAT polarization pipeline script:

https://github.com/AnnalisaB/MeerKAT-polarization/blob/main/MeerKAT_pol_script.py
   -- tested on L band  -written by Annalisa Bonafede - based on Ben Hugo strategy, adapted by Erik Osinga

IMPORTANT: ASSSUMES THAT CORRECT_PARANG.PY HAS BEEN RUN BEFORE THIS SCRIPT.


TODO:  currently assumes only 1 fcal: J0408-6545
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np

# from casatools import table as tb #, logger as casalog
from casatasks import (  # type: ignore
    applycal,
    bandpass,
    clearcal,
    flagdata,
    # flagmanager,
    gaincal,
    polcal,
    setjy,
    split,
    tclean,
)

POL_CALIBRATORS = ["J0521+1638", "J1331+3030"]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Full-polarisation calibration script for MeerKAT L-band'
    )
    # required arguments
    parser.add_argument(
        '--calms', required=True,
        help="Path to calibration MS (e.g. './A754-...-cal.ms')"
    )
    parser.add_argument(
        '--fcal', required=True,
        help="str denoting fcal (e.g. 'J0408-6545' or 'J1939-6342,J0408-6545')"
    )
    parser.add_argument(
        '--bpcal', required=True,
        help="str denoting bpcal (e.g. 'J0408-6545')"
    )
    parser.add_argument(
        '--gcal', required=True,
        help="str denoting gcal (e.g. 'J0408-6545')"
    )
    parser.add_argument(
        '--xcal', required=True,
        help="str denoting xcal (e.g. 'J1331+3030')"
    )

    # optional arguments
    parser.add_argument(
        '--targetms', default=None,
        help="Path to target MS (default: derived from --calms by replacing '-cal.ms' with '-target.ms')"
    )
    parser.add_argument(
        '--leakcal', default=None,
        help="Leakage calibrator. If None, defaults to fcal."
    )    
    parser.add_argument(
        '--refant', default='m024',
        help="Reference antenna (default: m024; m002 also works well)"
    )
    parser.add_argument(
        '--scan-xcal', default='',type=str,
        help="Which scans to use for polcal, should be casa-compliant stringList, e.g. '12,22' or just '12', or empty '' for all scans"
    )
    return parser.parse_args()


def setup_logging(prefix: str):
    """ deprecated
    log_file = f"{prefix}.log"
    logging.basicConfig(
        filename=log_file,
        format='%(asctime)s %(name)s:%(funcName)s\t%(message)s',
        datefmt='%Y-%m-%d %H:%M',
        filemode='w',
        level=logging.DEBUG
    )
    root = logging.getLogger()
    root.addHandler(logging.StreamHandler(sys.stdout))
    casa_log = f"{prefix}_casa.log"
    old = casalog.logfile()
    casalog.setlogfile(filename=casa_log)
    try:
        os.remove(old)
    except OSError:
        pass
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    return logging.getLogger(__name__)
    """
    return NotImplementedError

def main():
    args = parse_args()
    calms = args.calms
    ref_ant = args.refant
    
    # derive target MS path if not given
    if args.targetms:
        targetms = args.targetms
    else:
        if calms.endswith('-cal.ms'):
            targetms = calms.replace('-cal.ms', '-target.ms')
        else:
            targetms = calms.replace('.ms', '-target.ms')

    assert os.path.exists(calms), f"Calibration MS {calms} does not exist"
    assert os.path.exists(targetms), f"Target MS {targetms} does not exist"

    # logging
    # prefix = os.path.splitext(os.path.basename(calms))[0]
    # logger = setup_logging(prefix)
    logger = logging.getLogger(__name__)

    # scan and calibrators
    scan_xcal = args.scan_xcal

    fcal = args.fcal
    bpcal = args.bpcal
    gcal = args.gcal
    xcal = args.xcal
    if args.leakcal is None:    
        # default the leakage calibrator to the fcal, often unpolarised.
        leak_cal = fcal
    else:
        leak_cal = args.leakcal

    
    # double check leakage calibrator isnt a polarised one
    assert not np.any([p in leak_cal for p in POL_CALIBRATORS]), "Leakage calibrator cannot be polarised"
    
    logger.info("Set the following calibrators for cross-calibration:")
    logger.info(f"Flux calibrator: {fcal}")
    logger.info(f"Bandpass calibrator: {bpcal}")
    logger.info(f"Gain calibrator: {gcal}")
    logger.info(f"Cross-hand delay calibrator: {xcal}")
    logger.info(f"Leakage calibrator: {leak_cal}")


    # flags & plotting flags
    do_plot = False
    do_flags = False # we flag outside the script
    selfcal_xcal = True
    model_xcal = True
    split_xcal = True
    apply_target = True

    # ensure output dir
    if not os.path.exists('CASA_Tables'):
        os.makedirs('CASA_Tables')
        logger.info('Created CASA_Tables directory')

    # table names
    ktab       = 'CASA_Tables/calib.kcal'
    gtab_p     = 'CASA_Tables/calib.gcal_p'
    gtab_a     = 'CASA_Tables/calib.gcal_a'
    btab       = 'CASA_Tables/calib.bandpass'
    ptab_df    = 'CASA_Tables/calib.df'
    gtab_sec_p = 'CASA_Tables/calib.sec_p'
    Ttab_sec   = 'CASA_Tables/calib.T'
    gtab_pol_p = 'CASA_Tables/calib.gcal_pol_p'
    kxtab      = 'CASA_Tables/calib.kcrosscal'
    ptab_xf    = 'CASA_Tables/calib.xf'

    # Good practice
    clearcal(vis=calms)

    # set flux models for flux calibrators
    for cal in fcal.split(','): # from casa stringVec to python list
        
        logger.info(f'Setting model for fcal={cal}')

        if cal == 'J1939-6342':
            setjy(vis=calms, field=cal, standard='Stevens-Reynolds 2016', usescratch=True)

        elif cal == 'J0408-6545':
            import cal_J0408
            freqs = np.linspace(0.9, 2, 200) * 1e9
            a, b, c, d = -0.9790, 3.3662, -1.1216, 0.0861
            reffreq, flux, sp0, sp1, sp2 = cal_J0408.convert_flux_model(freqs, a, b, c, d)
            setjy(vis=calms, field=cal,
                  spix=[sp0, sp1, sp2, 0],
                  fluxdensity=flux,
                  reffreq=f'{reffreq} Hz',
                  standard='manual', usescratch=True)
            
        else:
            raise NotImplementedError(f"Flux calibrator {cal} not implemented. Please add it to the script.")

    # set xcal model
    if len(xcal.split(',')) > 1:
        raise NotImplementedError(f"Cross-hand delay calibrator {xcal} cannot be a list of calibrators. Please provide a single calibrator.")

    if model_xcal:
        if xcal == 'J0521+1638':
            # example for alternate xcal
            stokesI, alpha, reffreq, polfrac, polangle, rm = \
                8.33843, [-0.4981, -0.1552, -0.0102, 0.0223], '1.47GHz', 0.078, -0.16755, 0.
        elif xcal == 'J1331+3030':
            stokesI, alpha, reffreq, polfrac, polangle, rm = \
                14.7172, [-0.4507, -0.1798, 0.0357], '1.47GHz', 0.098, 0.575959, 0.
        else:
            logger.error(f'Unknown xcal: {xcal}')
            raise NotImplementedError(f"Cross-hand delay calibrator {xcal=} is not implemented.")
        
        setjy(vis=calms, field=xcal, standard='manual',
              fluxdensity=[stokesI, 0, 0, 0], spix=alpha,
              reffreq=reffreq, polindex=polfrac,
              polangle=polangle, rotmeas=rm, usescratch=True)


    # main calibration loops
    for loop in range(2 if do_flags else 1):
        # delay (K) calibration
        # - residual, most taken out at the obs - few nsec typical 
        gaincal(vis=calms, caltable=ktab, solint='inf', field=bpcal,
                refant=ref_ant, gaintype='K', parang=False)
        # phase & amp & bandpass
        for idx, fname in enumerate(bpcal.split(',')): # currently assumes 1 bpcal. can be extended

            append = idx > 0
            # phase cal on bandpass calibrator  - phse will be twhrown away - Ben uses infinite time to wash out RFI
            gaincal(vis=calms, caltable=gtab_p, solint='60s', field=fname,
                    refant=ref_ant, gaintable=[ktab], calmode='p', append=append, parang=False)
            # amp cal on bandpass calibrator
            gaincal(vis=calms, caltable=gtab_a, solint='inf', field=fname,
                    refant=ref_ant, gaintable=[ktab, gtab_p], calmode='a', append=append, parang=False)
            # ben averages bandpass ober scans - it's more stable he says
            bandpass(vis=calms, caltable=btab, solint='inf', field=fname,
                     refant=ref_ant, gaintable=[ktab, gtab_p, gtab_a], append=append, parang=False)

        #### disabled this as well, since flagging outside script.
        # if loop == 0 and do_flags:
        #     # restore pre-bandpass flags, *disabled because we flag outside this script with aoflagger.
        #     # flagmanager(vis=calms, mode='restore', versionname='BeforeBPcal')
        #     # apply cal to bpcal,xcal,gcal
        #     applycal(vis=calms,
        #              field=','.join([bpcal, xcal, gcal]),
        #              gaintable=[ktab, gtab_p, gtab_a, btab])
            
        #     # second flagging on corrected data
        #     # os.system(
        #     #     f"singularity run -B {binding_dir} {flocs_simg} "
        #     #     f"aoflagger -strategy {aoflagger_strategy} -column CORRECTED_DATA {calms}"
        #     # )

        #     # cleanup
        #     os.system(f"rm -rf {gtab_p} {gtab_a} {btab} {ktab}")

    # leakage (Df) calibration
    # -real part of reference antenna will be set to 0 -
    polcal(vis=calms, caltable=ptab_df, solint='inf', field=leak_cal,
           combine='scan', uvrange='>150lambda', refant=ref_ant,
           poltype='Df', gaintable=[ktab, gtab_p, gtab_a, btab],
           gainfield=['', leak_cal, leak_cal, leak_cal], interp=['nearest']*4)
    # # flag of solutions as caracal is doing Ben would suggest anything above 0.1
    flagdata(vis=ptab_df, mode='clip', datacolumn='CPARAM', clipminmax=[-0.6, 0.6])
    # # Apply Df to bpcal, check that after calibration amplitude is reduced 
    # NOTE: having flagged  RFI  is critical here
    applycal(vis=calms, field=leak_cal,
             gaintable=[ktab, gtab_p, gtab_a, btab, ptab_df],
             gainfield=['', leak_cal, leak_cal, leak_cal, ''], interp=['']*5)

    # Check that amplitude of leakage cal is gone down (few %) after calibration
    if do_plot:
        os.system(f"shadems --xaxis FREQ --yaxis CORRECTED_DATA "
                  f"--field {leak_cal} --corr XY,YX --png 'Df-cal.png' {calms}")

    # secondary calibration (p & T)
    gaincal(vis=calms, caltable=gtab_sec_p, solint='inf', field=gcal,
            refant=ref_ant, gaintype='G', calmode='p', uvrange='>150lambda',
            gaintable=[ktab, gtab_a, btab, ptab_df])
    # amplitude normalized to 1
    gaincal(vis=calms, caltable=Ttab_sec, solint='inf', field=gcal,
            refant=ref_ant, gaintype='T', calmode='ap', solnorm=True,
            uvrange='>150lambda', gaintable=[ktab, gtab_sec_p, gtab_a, btab, ptab_df])
    applycal(vis=calms, field=gcal,
             gaintable=[ktab, gtab_sec_p, gtab_a, btab, Ttab_sec, ptab_df])
    # check calibration of secondary
    if do_plot:
        os.system(f"shadems --xaxis UV --yaxis CORRECTED_DATA --field {gcal} "
                  f"--corr XX,YY --png 'Gcal-amp-XX-YY.png' {calms}")
        os.system(f"shadems --xaxis UV --yaxis CORRECTED_DATA:phase --field {gcal} "
                  f"--corr XX,YY --png 'Gcal-phase-XX-YY.png' {calms}")

    # apply to xcal and refine XY phase

    #apply calibration up to  now to xcal: XY and YX will vary with time due to parang,
    # CHECK that power of XY,YX applitude is not close to 0 (not power to cailbrate
    applycal(vis=calms, field=xcal,
             gaintable=[ktab, gtab_p, gtab_a, btab, Ttab_sec, ptab_df])
    if do_plot:
        os.system(f"shadems --xaxis CORRECTED_DATA:phase --yaxis CORRECTED_DATA:amp "
                  f"--field {xcal} --corr XX,YY --png 'Xf-precal-XX-YY.png' {calms}")
        os.system(f"shadems --xaxis TIME --yaxis CORRECTED_DATA:amp "
                  f"--field {xcal} --corr XY,YX --png 'Xf-precal-XY-YX.png' {calms}")

    # Calibrate XY phase: calibrate P on 3C286 - refine the phase
    gaincal(vis=calms, caltable=gtab_pol_p, solint='inf', field=xcal,
            refant=ref_ant, gaintype='G', calmode='p', parang=False,
            gaintable=[ktab, gtab_a, btab, ptab_df])
    # apply calibration up to  now, including phase refinement to xcal - 
    # corsshands shoudl be real vaue dominated, imaginary willl give idea of induced elliptcity.
    # change in real axis due to parang
    applycal(vis=calms, field=xcal,
             gaintable=[ktab, gtab_pol_p, gtab_a, btab, Ttab_sec, ptab_df])
    if do_plot:
        os.system(f"shadems --xaxis CORRECTED_DATA:phase --yaxis CORRECTED_DATA:amp "
                  f"--field {xcal} --corr XX,YY --png 'Xf-postcal-XX-YY.png' {calms}")

    # self-cal on xcal
    # #selfcal on Xcal - TO DO:larger FOV and mask, to further improve model
    if selfcal_xcal:
        tclean(vis=calms, field=xcal, cell='0.5arcsec', imsize=512,
               niter=1000, imagename=f'{xcal}-selfcal', weighting='briggs',
               robust=-0.2, datacolumn='corrected', deconvolver='mtmfs',
               nterms=2, specmode='mfs', interactive=False)
        sc_table = f'{gtab_pol_p}-selfcal'
        gaincal(vis=calms, field=xcal, calmode='p', solint='30s',
                caltable=sc_table, refant=ref_ant,
                gaintable=[ktab, gtab_a, btab, ptab_df], parang=False)
        gtab_pol_p = sc_table

    # cross-hand delay
    # # Cross-hand delay calibration - BEN says you can skip it - very sensitive to RFI
    gaincal(vis=calms, caltable=kxtab, solint='inf', field=xcal,
            scan=scan_xcal, refant=ref_ant, gaintype='KCROSS',
            gaintable=[ktab, gtab_pol_p, gtab_a, btab, ptab_df], parang=True)

    # final Xf polarisation calibration
    # # Calibrate XY phase - combine scan to improve SNR - better add Ttab_sec
    polcal(vis=calms, caltable=ptab_xf, solint='inf,20MHz', field=xcal,
           combine='scan', scan=scan_xcal, refant=ref_ant, poltype='Xf',
           gaintable=[ktab, gtab_pol_p, gtab_a, btab, ptab_df, kxtab])
    applycal(vis=calms, scan=scan_xcal, field=xcal,
             gaintable=[ktab, gtab_pol_p, gtab_a, btab, Ttab_sec, ptab_df, kxtab, ptab_xf],
             parang=True)
    if do_plot:
        os.system(f"shadems --xaxis CORRECTED_DATA:imag --yaxis CORRECTED_DATA:real "
                  f"--corr XY,YX --field {xcal} --png 'Xf-real-im.png' {calms}")
        os.system(f"shadems --xaxis CORRECTED_DATA:phase --yaxis CORRECTED_DATA:amp "
                  f"--field {xcal} --corr XX,YY --png 'Xf-refine-XX-YY.png' {calms}")

    # split xcal
    if split_xcal:
        output = calms.replace('.ms', f'.{xcal}-cal.ms')
        split(vis=calms, scan=scan_xcal, field=xcal, outputvis=output)

    # apply to target MS
    if apply_target:
        # assumes that correct_parang.py has been run on all fields
        applycal(vis=targetms,
                 gaintable=[ktab, gtab_sec_p, gtab_a, btab, Ttab_sec, ptab_df, kxtab, ptab_xf],
                 parang=True)

    # FOR SELFCAL: Either selfcal only for scalar phase ad amplitude from here 
    # OR do images and gaincal with parang=True becasue parag correction does 
    # not commune with gain amplitude
    # Remember: Parang correction is amplitude correction, so if you selfcal 
    # on phase only tou can selfcal diagonal XX and YY separately

if __name__ == '__main__':
    main()
