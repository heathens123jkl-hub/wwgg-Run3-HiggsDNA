import numpy as np
import uproot

from higgs_dna.tools.photonid_mva import calculate_photonid_mva
from higgs_dna.tools.xgb_loader import load_bdt


def test_photonid():
    """Compute PhotonID MVA on a simple ROOT file with a leading photon which contains
    only the input variables to the BDT
    """
    f = uproot.open("tests/samples/phoid_mva.root")
    t = f["Events"]
    photons = t.arrays()
    labels = photons.fields
    model_path = "tests/models/PhoID_barrel_UL2017.json.gz"
    model = load_bdt(model_path)
    mva = calculate_photonid_mva(
        (model, labels),
        photons,
        sigmoid=True
    )

    assert np.median(mva) > 0.9
