from importlib import resources
import json


def test_package():
    """
    Test the package imports
    """
    from higgs_dna import tools
    from higgs_dna import utils
    from higgs_dna import workflows


def test_metaconditions():
    """
    Test the metaconditions
    1. Check that by opening the metacondition files with this procedure we get dictionaries
    """
    from higgs_dna.metaconditions import metaconditions

    for json_file in metaconditions.values():
        with resources.open_text("higgs_dna.metaconditions", json_file) as f:
            dct = json.load(f)
            assert isinstance(dct, dict)
