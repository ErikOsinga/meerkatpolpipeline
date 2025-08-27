from __future__ import annotations

from pathlib import Path

from prefect import flow
from prefect.logging.loggers import disable_run_logger

from meerkatpolpipeline.casacrosscal.casacrosscal import run_casa_script

"""Currently assumes tests are run in the 'tests' directory """


def test_run_casasript():
    """Test running a casa script with argparse via singularity"""

    from meerkatpolpipeline.casacrosscal import (
        cal_J0408,  # cant import casa scripts
    )
    casa_script = Path(cal_J0408.__file__).parent / "test_casa_script_args.py"

    casa_container = Path("/home/osingae/OneDrive/postdoc/projects/DoradoGroup/software/flint-containers_casa.sif")
    print("TODO: make CASA test run anywhere by downloading the CASA container if it doesnt exist")
    if not casa_container.exists():
        raise FileNotFoundError(f"CASA container not found: {casa_container}")

    # run casa crosscal script
    cmd_casa = f"""casa --nologger --nogui -c {casa_script} \
        --calms calms1 \
        --targetms targetms1 \
        --fcal fcal1 \
        --bpcal bpcal1 \
        --gcal gcal1 \
        --xcal xcal1 \
        --leakcal leakcal1 \
    """

    with disable_run_logger():
        run_casa_script(
            cmd_casa=cmd_casa,
            container=casa_container,
            bind_dirs=[]
        )

    # If it did not raise an error, we are good
    return 

if __name__ == "__main__":
    test_run_casasript()