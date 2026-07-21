import awkward as ak
import onnxruntime as ort
import numpy as np
import copy
from higgs_dna.selections.HHbbgg_selections import Cxx
from higgs_dna.selections.HHbbgg_selections import DeltaR, DeltaPhi


def apply_VBFHH_pairing(model, dijets_org, diphotons):

    copy_columns = ["first_jet", "second_jet", "mass", "pt", "eta"]
    dijets = copy.copy(dijets_org[copy_columns])

    n_jets = ak.firsts(diphotons['n_jets'])
    Hgg_eta = ak.firsts(diphotons['eta'])
    eta_sum = dijets["first_jet"].eta + dijets["second_jet"].eta
    eta_diff = dijets["first_jet"].eta - dijets["second_jet"].eta

    dijets["n_jets"] = n_jets
    dijets["Hgg_eta"] = Hgg_eta

    pair_DeltaPhi = DeltaPhi(dijets["first_jet"], dijets["second_jet"])
    pair_DeltaR = DeltaR(dijets["first_jet"], dijets["second_jet"])
    pair_Cgg = Cxx(eta_diff, eta_sum, Hgg_eta)

    input_features = ak.zip({
        "pair1_ptOverM": dijets["first_jet"].pt / dijets.mass,
        "pair2_ptOverM": dijets["second_jet"].pt / dijets.mass,
        "pair1_eta": dijets["first_jet"].eta,
        "pair2_eta": dijets["second_jet"].eta,
        "pair1_btagPNetB": dijets["first_jet"].btagPNetB,
        "pair2_btagPNetB": dijets["second_jet"].btagPNetB,
        "pair1_btagPNetQvG": dijets["first_jet"].btagPNetQvG,
        "pair2_btagPNetQvG": dijets["second_jet"].btagPNetQvG,
        "n_jets": dijets["n_jets"],
        "pair_ptOverM": dijets.pt / dijets.mass,
        "pair_eta": dijets.eta,
        "pair_DeltaR": pair_DeltaR,
        "pair_DeltaPhi": pair_DeltaPhi,
        "pair_eta_prod": dijets["first_jet"].eta * dijets["second_jet"].eta,
        "pair_eta_diff": eta_diff,
        "pair_Cgg": pair_Cgg,
        "pair1_phi": dijets["first_jet"].phi,
        "pair2_phi": dijets["second_jet"].phi
    })

    # Process the format for prediction

    counts = ak.num(input_features)

    flat_features = ak.to_numpy(ak.flatten(input_features, axis=1))
    plain_array = np.stack([flat_features[field] for field in flat_features.dtype.names], axis=-1).astype(np.float32)

    sess = ort.InferenceSession(model)
    input_name = sess.get_inputs()[0].name

    outputs = sess.run(None, {input_name: plain_array})
    pred = ak.unflatten(outputs[0], counts)

    dijets_org["vbfpair_Class"] = ak.argmax(pred, axis=-1, mask_identity=False)

    dijets_org["vbfpair_Score_bb"] = pred[..., 0]
    dijets_org["vbfpair_Score_jj"] = pred[..., 1]

    return dijets_org
