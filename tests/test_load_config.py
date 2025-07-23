from __future__ import annotations

import sys
from pathlib import Path

sys.path.append('/home/osingae/OneDrive/postdoc/projects/MEERKAT_similarity_Bfields/meerkatpolpipeline')

from meerkatpolpipeline.configuration import load_and_copy_strategy

"""Currently assumes tests are run in the 'tests' directory """


def test_load_sample_config():
    """Test loading config file and creating timestamped copy"""
    
    strategy_file = Path("./temp_sample_configuration.yaml")
    output_path = Path("./")
    strategy = load_and_copy_strategy(strategy_file, output_path)

    assert strategy['defaults']['download']['tries'] == 'inf'

if __name__ == "__main__":
    test_load_sample_config()