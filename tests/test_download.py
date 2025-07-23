from __future__ import annotations

import sys
from pathlib import Path

sys.path.append('/home/osingae/OneDrive/postdoc/projects/MEERKAT_similarity_Bfields/meerkatpolpipeline')

from meerkatpolpipeline.download.download import DownloadOptions, start_download


def test_download() -> None:
    """Test download with wget command"""
    
    opt_dict = {'link': 'test.com', 'output_name': 'test.tar.gz'}
    downloadoptions = DownloadOptions(enable=True)
    downloadoptions = downloadoptions.with_options(**opt_dict)

    cmd = start_download(downloadoptions, working_dir=Path("/path/to/workdir"), test=True)

    assert cmd == "wget --tries inf --waitretry=2 -c -O /path/to/workdir/test.tar.gz test.com"

    return