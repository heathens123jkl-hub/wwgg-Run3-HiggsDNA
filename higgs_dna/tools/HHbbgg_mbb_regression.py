import awkward as ak
import logging
import numpy as np
import onnxruntime as ort
logger = logging.getLogger(__name__)


def calculate_mbb_regression(
    model_file,
    dijets: ak.Array,
) -> ak.Array:
    """
    Calculate dijet regressed mass for HHbbgg workflow.
    """

    mbb_features = [
        "puppiMET_pt_over_mjj",
        "puppiMET_sumEt_over_mjj",
        "puppiMET_phi",
        "lead_bjet_pt_over_mjj",
        "lead_bjet_mass_over_mjj",
        "lead_bjet_eta_sign",
        "lead_bjet_phi",
        "lead_bjet_btagPNetB",
        "sublead_bjet_pt_over_mjj",
        "sublead_bjet_mass_over_mjj",
        "sublead_bjet_eta_sign",
        "sublead_bjet_phi",
        "sublead_bjet_btagPNetB",
        "lead_bjet_PNetRegPtRawRes",
        "sublead_bjet_PNetRegPtRawRes",
        "projectphi2M",
        "projectphi1M"]

    dijets_input = dijets[["puppiMET_phi","puppiMET_pt", "puppiMET_sumEt"]]

    # calculate input variables:
    for feature in mbb_features:
        if "sublead_bjet" in feature:
            dijets_input["sublead_bjet_" + feature.split("_")[2]] = dijets["second_jet"][feature.split("_")[2]]
        elif "lead_bjet" in feature:
            dijets_input["lead_bjet_" + feature.split("_")[2]] = dijets["first_jet"][feature.split("_")[2]]
        if "over_mjj" in feature:
            dijets_input[feature] = dijets_input[feature.replace("_over_mjj" , '')] / dijets["mass"]
        if "sign" in feature:
            dijets_input[feature] = dijets_input[feature.replace("_sign" , '')] * np.sign(dijets_input["lead_bjet_eta"])
        if "projectphi1M" in feature:
            dijets_input[feature] = np.cos(((dijets_input["lead_bjet_phi"] - dijets_input["puppiMET_phi"]) + np.pi) % (2 * np.pi) - np.pi)
        if "projectphi2M" in feature:
            dijets_input[feature] = np.cos(((dijets_input["sublead_bjet_phi"] - dijets_input["puppiMET_phi"]) + np.pi) % (2 * np.pi) - np.pi)
    # flatten dijets for input into DNN

    dijets_input = dijets_input[mbb_features]

    flat_fields = [ak.to_numpy(ak.flatten(dijets_input[field], axis=1)) for field in mbb_features]
    flat_input = np.stack(flat_fields, axis=-1).astype(np.float32)

    session = ort.InferenceSession(model_file)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: flat_input})
    preds_array = outputs[0]
    counts = ak.num(dijets_input, axis=1)
    preds_array = preds_array.reshape(-1)
    mbb_correction = ak.unflatten(preds_array, counts)

    dijets["mass_DNNreg"] = (mbb_correction * dijets["mass"]) + dijets["mass"]

    return dijets
