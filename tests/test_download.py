from __future__ import annotations

from pathlib import Path

from prefect.logging.loggers import disable_run_logger

from meerkatpolpipeline.download.download import DownloadOptions, download_and_extract


def test_download() -> None:
    """Test download with wget command"""
    
    opt_dict = {
        'link': 'test.com',
        'output_name': 'test.ms.tar.gz',
        'ms_name': 'test.ms'
    }
    downloadoptions = DownloadOptions(enable=True)
    # options are implemented as dicts
    downloadoptions = dict(downloadoptions.with_options(**opt_dict))

    with disable_run_logger():
        cmd = download_and_extract(downloadoptions, working_dir=Path("/path/to/workdir"), test=True)

    assert cmd == "wget --tries inf --waitretry=2 -c -O /path/to/workdir/test.ms.tar.gz test.com"

    return