import numpy as np
import awkward as ak
import vector
import logging
import pandas as pd
import onnxruntime as ort

vector.register_awkward()

logger = logging.getLogger(__name__)


def CreatePairs(j1, j2, diphotons_notsort, nano_version):

    diphotons = ak.firsts(diphotons_notsort)

    photon_lead = ak.zip(
        {
            "pt": diphotons['pho_lead'].pt,
            "eta": diphotons['pho_lead'].eta,
            "phi": diphotons['pho_lead'].phi,
            "mass": ak.zeros_like(diphotons['pho_lead'].pt),
        }
    )
    photon_lead = ak.with_name(photon_lead, "Momentum4D")

    photon_sublead = ak.zip(
        {
            "pt": diphotons['pho_sublead'].pt,
            "eta": diphotons['pho_sublead'].eta,
            "phi": diphotons['pho_sublead'].phi,
            "mass": ak.zeros_like(diphotons['pho_sublead'].pt),
        }
    )
    photon_sublead = ak.with_name(photon_sublead, "Momentum4D")

    diphoton = photon_lead + photon_sublead
    ak.with_name(diphoton, "Momentum4D")

    jet1 = ak.zip(
        {
            "pt": ak.flatten(ak.pad_none(j1.pt, 1, axis=1)),
            "eta":ak.flatten(ak.pad_none(j1.eta, 1, axis=1)),
            "phi": ak.flatten(ak.pad_none(j1.phi, 1, axis=1)),
            "mass": ak.flatten(ak.pad_none(j1.mass, 1, axis=1)),
        }
    )
    jet1 = ak.with_name(jet1, "Momentum4D")

    jet2 = ak.zip(
        {
            "pt": ak.flatten(ak.pad_none(j2.pt, 1, axis=1)),
            "eta": ak.flatten(ak.pad_none(j2.eta, 1, axis=1)),
            "phi": ak.flatten(ak.pad_none(j2.phi, 1, axis=1)),
            "mass": ak.flatten(ak.pad_none(j2.mass, 1, axis=1)),
        }
    )
    jet2 = ak.with_name(jet2, "Momentum4D")

    dijet = jet1 + jet2
    dijet = ak.with_name(dijet, "Momentum4D")

    DeltaR_j1g1 = vector.Spatial.deltaR(photon_lead, jet1)
    DeltaR_j2g1 = vector.Spatial.deltaR(photon_lead, jet2)
    DeltaR_j1g2 = vector.Spatial.deltaR(photon_sublead, jet1)
    DeltaR_j2g2 = vector.Spatial.deltaR(photon_sublead, jet2)
    DeltaRj1j2 = vector.Spatial.deltaR(jet1, jet2)
    DeltaRlist = ak.Array([DeltaR_j1g1, DeltaR_j2g2, DeltaR_j1g2, DeltaR_j2g1])

    DeltaR_jg_min = ak.min(DeltaRlist, axis=0)
    index_min = ak.argmin(DeltaRlist, axis=0)

    index_min = ak.fill_none(index_min, 0)
    index_map = np.array([1, 0, 3, 2])
    DeltaR_jg_notmin = DeltaRlist[index_map[index_min], np.arange(len(index_min))]

    HH_pair = {
        "lead_bjet_pt": ak.fill_none(jet1.pt, -999),
        "lead_bjet_eta": ak.fill_none(jet1.eta, -999),
        "lead_bjet_phi": ak.fill_none(jet1.phi, -999),
        "lead_bjet_mass": ak.fill_none(jet1.mass, - 999),
        "lead_bjet_ptOvermass": ak.fill_none(jet1.pt / dijet.mass, -999),
        "lead_bjet_btagPNetB": ak.fill_none(ak.flatten(ak.pad_none(j1.btagPNetB, 1, axis=1)), -999),
        "lead_bjet_btagRobustParTAK4B": ak.fill_none(ak.flatten(ak.pad_none(j1.btagRobustParTAK4B if nano_version <= 13 else j1.btagUParTAK4B, 1, axis=1)), -999),
        "lead_bjet_btagUParTAK4B": ak.fill_none(ak.flatten(ak.pad_none(j1.btagRobustParTAK4B if nano_version <= 13 else j1.btagUParTAK4B, 1, axis=1)), -999),
        "lead_bjet_btagDeepFlav_B": ak.fill_none(ak.flatten(ak.pad_none(j1.btagDeepFlav_B, 1, axis=1)), -999),
        "sublead_bjet_pt": ak.fill_none(jet2.pt, -999),
        "sublead_bjet_eta": ak.fill_none(jet2.eta, -999),
        "sublead_bjet_phi": ak.fill_none(jet2.phi, -999),
        "sublead_bjet_mass": ak.fill_none(jet2.mass, -999),
        "sublead_bjet_ptOvermass": ak.fill_none(jet2.pt / dijet.mass, -999),
        "sublead_bjet_btagPNetB": ak.fill_none(ak.flatten(ak.pad_none(j2.btagPNetB, 1, axis=1)), -999),
        "sublead_bjet_btagRobustParTAK4B": ak.fill_none(ak.flatten(ak.pad_none(j2.btagRobustParTAK4B if nano_version <= 13 else j2.btagUParTAK4B, 1, axis=1)), -999),
        "sublead_bjet_btagUParTAK4B": ak.fill_none(ak.flatten(ak.pad_none(j2.btagRobustParTAK4B if nano_version <= 13 else j2.btagUParTAK4B, 1, axis=1)), -999),
        "sublead_bjet_btagDeepFlav_B": ak.fill_none(ak.flatten(ak.pad_none(j2.btagDeepFlav_B, 1, axis=1)), -999),
        "DeltaRj1j2": ak.fill_none(DeltaRj1j2, -999),
        "absCosThetaStar_CS": ak.fill_none(getCosThetaStar_CS(dijet,diphoton), -999),
        "absCosThetaStar_jj": ak.fill_none(getCosThetaStar_jj(dijet, jet1), -999),
        "DeltaR_jg_min": ak.fill_none(DeltaR_jg_min, -999),
        "DeltaR_jg_notmin": ak.fill_none(DeltaR_jg_notmin, -999),
        'n_jets': ak.fill_none(diphotons['n_jets'], -999),
    }

    return HH_pair


def getCosThetaStar_jj(dijet, jet1):

    hjjforboost = ak.zip({"px": -dijet.px, "py": -dijet.py, "pz": -dijet.pz, "E": dijet.E})
    hjjforboost = ak.with_name(hjjforboost, "Momentum4D")

    Hjj_jet_boosted = jet1.boost(hjjforboost)

    return Hjj_jet_boosted.costheta


def getCosThetaStar_CS(dijet, diphoton, ebeam=6800):

    """
    cos theta star angle in the Collins Soper frame
    Copied directly from here: https://github.com/ResonantHbbHgg/Selection/blob/master/selection.h#L3367-L3385
    """

    p1 = ak.zip(
        {
            "px": 0,
            "py": 0,
            "pz": ebeam,
            "E": ebeam,
        },
        with_name="Momentum4D",
    )

    p2 = ak.zip(
        {
            "px": 0,
            "py": 0,
            "pz": -ebeam,
            "E": ebeam,
        },
        with_name="Momentum4D",
    )

    HH = dijet + diphoton
    HH = ak.with_name(HH,"Momentum4D")

    hhforboost = ak.zip({"px": -HH.px, "py": -HH.py, "pz":-HH.pz, "E": HH.E})
    hhforboost = ak.with_name(hhforboost, "Momentum4D")

    p1 = p1.boost(hhforboost)
    p2 = p2.boost(hhforboost)
    diphotonBoosted = diphoton.boost(hhforboost)

    CSaxis = (p1 - p2)

    return np.cos(CSaxis.deltaangle(diphotonBoosted))


def Compute_DNN_bpairing(dijets, diphotons, keras_model, nano_version):

    original_count = ak.num(dijets, axis=1)
    dijet = dijets[ak.local_index(dijets, axis=1) < 10]

    if "Run2" in keras_model:
        var = [
            'lead_bjet_ptOvermass',
            'sublead_bjet_ptOvermass',
            'lead_bjet_btagPNetB',
            'lead_bjet_btagUParTAK4B',
            'lead_bjet_btagDeepFlav_B',
            'sublead_bjet_btagPNetB',
            'sublead_bjet_btagUParTAK4B',
            'sublead_bjet_btagDeepFlav_B',
            'absCosThetaStar_CS',
            'absCosThetaStar_jj',
            'DeltaRj1j2',
            'DeltaR_jg_min'
        ]
    else:
        var = [
            'lead_bjet_ptOvermass',
            'sublead_bjet_ptOvermass',
            'lead_bjet_btagPNetB',
            'lead_bjet_btagDeepFlav_B',
            'sublead_bjet_btagPNetB',
            'sublead_bjet_btagDeepFlav_B',
            'absCosThetaStar_CS',
            'absCosThetaStar_jj',
            'DeltaRj1j2',
            'DeltaR_jg_min'
        ]

    session = ort.InferenceSession(keras_model)
    input_name = session.get_inputs()[0].name

    prediction = None
    count1 = ak.num(dijet, axis=1)

    Isin_dijet = None

    if ak.max(count1) is None :
        return ak.zeros_like(dijets["pt"])

    for i in range(ak.max(count1)) :
        if Isin_dijet is None :
            Isin_dijet = ak.singletons(ak.where(count1 > i , 1, 0))
        else :
            Isin_dijet = ak.concatenate([Isin_dijet, ak.singletons(ak.where(count1 > i , 1, 0))], axis=1)

        j1j2 = CreatePairs(dijet[ak.local_index(dijet, axis=1) == i]["first_jet"], dijet[ak.local_index(dijet, axis=1) == i]["second_jet"], diphotons, nano_version)

        j1j2_pd = pd.DataFrame(j1j2)
        input_data = j1j2_pd[var].to_numpy().astype(np.float32)
        Isin_dijeti = ak.flatten(Isin_dijet[ak.local_index(Isin_dijet, axis=1) == i])
        if prediction is None :
            prediction = ak.singletons(ak.where(Isin_dijeti == 1, ak.flatten(session.run(None, {input_name: input_data})[0]), -10.0))

        else :
            prediction = ak.concatenate([prediction, ak.singletons(ak.where(Isin_dijeti == 1, ak.flatten(session.run(None, {input_name: input_data})[0]), -10.0))], axis=1)

    prediction_slimmed = prediction[prediction > -1]

    count_difference = original_count - ak.num(prediction_slimmed, axis=1)

    Array_of_Nothing = ak.Array([[-999.0] * i for i in count_difference])
    prediction_padded = ak.concatenate([prediction_slimmed, Array_of_Nothing], axis=1)

    return prediction_padded
