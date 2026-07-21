import gzip
import awkward as ak
import numpy as np
import correctionlib
from correctionlib.schemav2 import Correction, CorrectionSet

import os
import logging

logger = logging.getLogger(__name__)


def get_jer_correction_set(jer_json, jer_ptres_tag, jer_sf_tag):
    # learned from: https://github.com/cms-nanoAOD/correctionlib/issues/130
    with gzip.open(jer_json) as fin:
        cset = CorrectionSet.parse_raw(fin.read())

    cset.corrections = [
        c
        for c in cset.corrections
        if c.name
        in (
            jer_ptres_tag,
            jer_sf_tag,
        )
    ]
    cset.compound_corrections = []

    res = Correction.parse_obj(
        {
            "name": "JERSmear",
            "description": "Jet smearing tool",
            "inputs": [
                {"name": "JetPt", "type": "real"},
                {"name": "JetEta", "type": "real"},
                {
                    "name": "GenPt",
                    "type": "real",
                    "description": "matched GenJet pt, or -1 if no match",
                },
                {"name": "Rho", "type": "real", "description": "entropy source"},
                {"name": "EventID", "type": "int", "description": "entropy source"},
                {
                    "name": "JER",
                    "type": "real",
                    "description": "Jet energy resolution",
                },
                {
                    "name": "JERsf",
                    "type": "real",
                    "description": "Jet energy resolution scale factor",
                },
            ],
            "output": {"name": "smear", "type": "real"},
            "version": 1,
            "data": {
                "nodetype": "binning",
                "input": "GenPt",
                "edges": [-1, 0, 1],
                "flow": "clamp",
                "content": [
                    # stochastic
                    {
                        # rewrite gen_pt with a random gaussian
                        "nodetype": "transform",
                        "input": "GenPt",
                        "rule": {
                            "nodetype": "hashprng",
                            "inputs": ["JetPt", "JetEta", "Rho", "EventID"],
                            "distribution": "normal",
                        },
                        "content": {
                            "nodetype": "formula",
                            # TODO min jet pt?
                            "expression": "1+sqrt(max(x*x - 1, 0)) * y * z",
                            "parser": "TFormula",
                            # now gen_pt is actually the output of hashprng
                            "variables": ["JERsf", "JER", "GenPt"],
                        },
                    },
                    # deterministic
                    {
                        "nodetype": "formula",
                        # TODO min jet pt?
                        "expression": "1+(x-1)*(y-z)/y",
                        "parser": "TFormula",
                        "variables": ["JERsf", "JetPt", "GenPt"],
                    },
                ],
            },
        }
    )
    cset.corrections.append(res)
    ceval = cset.to_evaluator()
    return ceval


def get_jersmear(_eval_dict, _ceval, _jer_sf_tag, _syst="nom"):
    _eval_dict.update({"systematic": _syst})
    _inputs_jer_sf = [_eval_dict[input.name] for input in _ceval[_jer_sf_tag].inputs]
    _jer_sf = _ceval[_jer_sf_tag].evaluate(*_inputs_jer_sf)
    _eval_dict.update({"JERsf": _jer_sf})
    _inputs = [_eval_dict[input.name] for input in _ceval["JERSmear"].inputs]
    _jersmear = _ceval["JERSmear"].evaluate(*_inputs)
    return _eval_dict, _jersmear


def apply_split_jec_variations(jec_syst_map, jec, algo, cset, year, era, eval_dict, jets, AK8):
    for i in jec_syst_map:
        # get the total uncertainty
        tag_jec_syst = "_".join([jec, jec_syst_map[i], algo])
        try:
            sf = cset[tag_jec_syst]
        except BaseException:
            logger.error(
                f"[ jerc_jet ] No JEC systematic: {tag_jec_syst} - Year: {year} - Era: {era}"
            )
            exit(-1)
        # systematics
        inputs = [eval_dict[input.name] for input in sf.inputs]
        sf_delta = sf.evaluate(*inputs)

        # divide by correction since it is already applied before
        corr_up_variation = 1 + sf_delta
        corr_down_variation = 1 - sf_delta

        i_name = i
        if AK8:
            i_name = i.replace("jec_", "jec_AK8_")
        jets[f"pt_{i_name}_up"] = jets.pt * corr_up_variation
        jets[f"pt_{i_name}_down"] = jets.pt * corr_down_variation
        jets[f"mass_{i_name}_up"] = jets.mass * corr_up_variation
        jets[f"mass_{i_name}_down"] = jets.mass * corr_down_variation


def jerc_jet(
    pt,
    events,
    year="2022postEE",
    era="MC",
    level="L1L2L3Res",
    apply_jec=True,
    jec_syst=False,
    split_jec_syst=False,
    apply_jer=False,
    jer_syst=False,
    AK8=False,
    reg="",
    is_Run2_v15=False,
    clipping_24=False,
):
    if year in ("2024", "2025"):
        logger.warning("Current 2024 and 2025 JER are preliminary, 2023PostBPix is used! These ntuples should not be used for a final physics result!")
    # first, check if it's data or MC
    if era == "MC" and hasattr(events, "GenPart"):
        logger.debug(
            f"[ jerc_jet ] - JERC for simulation - Year: {year} - Era: {era} - JEC: {apply_jec}, systematics: {jec_syst}, splitting: {split_jec_syst}, AK8: {AK8}, Regression: {reg} - JER: {apply_jer}, systematics: {jer_syst}"
        )
    elif (("Run" in era) or ("Data" in era)) and (not hasattr(events, "GenPart")):
        apply_jec = True
        jec_syst = False
        split_jec_syst = False
        apply_jer = False
        jer_syst = False
        logger.debug(
            f"[ jerc_jet ] - JERC for Data - Year: {year} - Era: {era} - Only JEC to be applied - JEC: {apply_jec}, systematics: {jec_syst}, splitting: {split_jec_syst}, AK8: {AK8}, Regression: {reg} - JER: {apply_jer}, systematics: {jer_syst}"
        )
    else:
        logger.error(f"[ jerc_jet ] - Era: {era} doesn't match the input dataset")
        exit(-1)
    # run2: AK4PFchs - run3: AK4PFPuppi
    # If some pT regression is used, the proper corrections will be picked up
    if int(year[:4]) > 2018:
        algo = "AK4PFPuppi" + reg
        # Workaround until 2024 & 2025 regressed JECs are ready
        if year == "2024" or year == "2025":
            algo = "AK4PFPuppi"
    else:
        if is_Run2_v15:
            # To be made "algo = "AK4PFPuppi" + reg" when Run 2 v15 JECs are ready
            algo = "AK4PFPuppi"
        else:
            algo = "AK4PFchs"

    if AK8:
        algo = "AK8PFPuppi"

    # Workaround until Run 2 v15, 2024 & 2025 regressed JECs are ready
    if is_Run2_v15 or year == "2024" or year == "2025":
        regFlag = ""
    else:
        if reg == "":
            regFlag = ""
        elif "PNet" in reg:
            regFlag = "_PNet"
        elif "UParT" in reg:
            regFlag = "_UParT"
        else:
            logger.error(f"Unknown regression algorithm: {reg}")
            exit(-1)

    jetType = "jet"
    if AK8:
        jetType = "fatJet"
    Run2_PUPPI_json = ""
    if is_Run2_v15:
        Run2_PUPPI_json = "_v15"
    # jec json file
    jerc_json = {
        "2016preVFP": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2016preVFP_UL/" + jetType + "_jerc" + Run2_PUPPI_json + ".json.gz",
        ),
        "2016postVFP": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2016postVFP_UL/" + jetType + "_jerc" + Run2_PUPPI_json + ".json.gz",
        ),
        "2017": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2017_UL/" + jetType + "_jerc" + Run2_PUPPI_json + ".json.gz",
        ),
        "2018": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2018_UL/" + jetType + "_jerc" + Run2_PUPPI_json + ".json.gz",
        ),
        "2022preEE": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2022_Summer22/" + jetType + "_jerc" + regFlag + ".json.gz",
        ),
        "2022postEE": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2022_Summer22EE/" + jetType + "_jerc" + regFlag + ".json.gz",
        ),
        "2023preBPix": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2023_Summer23/" + jetType + "_jerc" + regFlag + ".json.gz",
        ),
        "2023postBPix": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2023_Summer23BPix/" + jetType + "_jerc" + regFlag + ".json.gz",
        ),
        "2024": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2024_Summer24/" + jetType + "_jerc" + regFlag + ".json.gz",
        ),
        "2025": os.path.join(
            os.path.dirname(__file__),
            "../systematics/JSONs/POG/JME/2025_Winter25/" + jetType + "_jerc" + regFlag + ".json.gz",
        ),
    }
    jec_version = {
        "2016preVFP": {
            "RunB": f"Summer{'20' if is_Run2_v15 else '19'}UL16APV_RunBCD_{'V1' if is_Run2_v15 else 'V7'}_DATA",
            "RunC": f"Summer{'20' if is_Run2_v15 else '19'}UL16APV_RunBCD_{'V1' if is_Run2_v15 else 'V7'}_DATA",
            "RunD": f"Summer{'20' if is_Run2_v15 else '19'}UL16APV_RunBCD_{'V1' if is_Run2_v15 else 'V7'}_DATA",
            "RunE": f"Summer{'20' if is_Run2_v15 else '19'}UL16APV_RunEF_{'V1' if is_Run2_v15 else 'V7'}_DATA",
            "RunF": f"Summer{'20' if is_Run2_v15 else '19'}UL16APV_RunEF_{'V1' if is_Run2_v15 else 'V7'}_DATA",
            "MC": f"Summer{'20' if is_Run2_v15 else '19'}UL16APV_{'V1' if is_Run2_v15 else 'V7'}_MC",
        },
        "2016postVFP": {
            "RunF": f"Summer{'20' if is_Run2_v15 else '19'}UL16_RunFGH_{'V1' if is_Run2_v15 else 'V7'}_DATA",
            "RunG": f"Summer{'20' if is_Run2_v15 else '19'}UL16_RunFGH_{'V1' if is_Run2_v15 else 'V7'}_DATA",
            "RunH": f"Summer{'20' if is_Run2_v15 else '19'}UL16_RunFGH_{'V1' if is_Run2_v15 else 'V7'}_DATA",
            "MC": f"Summer{'20' if is_Run2_v15 else '19'}UL16_{'V1' if is_Run2_v15 else 'V7'}_MC",
        },
        "2017": {
            "RunB": f"Summer{'20' if is_Run2_v15 else '19'}UL17_RunB_{'V1' if is_Run2_v15 else 'V5'}_DATA",
            "RunC": f"Summer{'20' if is_Run2_v15 else '19'}UL17_RunC_{'V1' if is_Run2_v15 else 'V5'}_DATA",
            "RunD": f"Summer{'20' if is_Run2_v15 else '19'}UL17_RunD_{'V1' if is_Run2_v15 else 'V5'}_DATA",
            "RunE": f"Summer{'20' if is_Run2_v15 else '19'}UL17_RunE_{'V1' if is_Run2_v15 else 'V5'}_DATA",
            "RunF": f"Summer{'20' if is_Run2_v15 else '19'}UL17_RunF_{'V1' if is_Run2_v15 else 'V5'}_DATA",
            "MC": f"Summer{'20' if is_Run2_v15 else '19'}UL17_{'V1' if is_Run2_v15 else 'V5'}_MC",
        },
        "2018": {
            "RunA": f"Summer{'20' if is_Run2_v15 else '19'}UL18_RunA_{'V1' if is_Run2_v15 else 'V5'}_DATA",
            "RunB": f"Summer{'20' if is_Run2_v15 else '19'}UL18_RunB_{'V1' if is_Run2_v15 else 'V5'}_DATA",
            "RunC": f"Summer{'20' if is_Run2_v15 else '19'}UL18_RunC_{'V1' if is_Run2_v15 else 'V5'}_DATA",
            "RunD": f"Summer{'20' if is_Run2_v15 else '19'}UL18_RunD_{'V1' if is_Run2_v15 else 'V5'}_DATA",
            "MC": f"Summer{'20' if is_Run2_v15 else '19'}UL18_{'V1' if is_Run2_v15 else 'V5'}_MC",
        },
        "2022preEE": {
            "RunC": f"Summer22_22Sep2023_RunCD_{'V3' if reg == '' else 'V2'}_DATA",
            "RunD": f"Summer22_22Sep2023_RunCD_{'V3' if reg == '' else 'V2'}_DATA",
            "MC": f"Summer22_22Sep2023_{'V3' if reg == '' else 'V2'}_MC",
        },
        "2022postEE": {
            "RunE": f"Summer22EE_22Sep2023_RunE_{'V3' if reg == '' else 'V2'}_DATA",
            "RunF": f"Summer22EE_22Sep2023_RunF_{'V3' if reg == '' else 'V2'}_DATA",
            "RunG": f"Summer22EE_22Sep2023_RunG_{'V3' if reg == '' else 'V2'}_DATA",
            "MC": f"Summer22EE_22Sep2023_{'V3' if reg == '' else 'V2'}_MC",
        },
        # For 2023, the correct era is chosen based on the run the event is in.
        # Details: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/merge_requests/118
        "2023preBPix": {
            "Data": f"Summer23Prompt23_{'V2' if reg == '' else 'V1'}_DATA",
            "RunCv123": f"Summer23Prompt23_RunCv123_{'V2' if reg == '' else 'V1'}_DATA",
            "RunCv4": f"Summer23Prompt23_RunCv4_{'V2' if reg == '' else 'V1'}_DATA",
            "MC": f"Summer23Prompt23_{'V2' if reg == '' else 'V1'}_MC",
        },
        "2023postBPix": {
            "Data": f"Summer23BPixPrompt23_{'V3' if reg == '' else 'V1'}_DATA",
            "RunD": f"Summer23BPixPrompt23_RunD_{'V3' if reg == '' else 'V1'}_DATA",
            "MC": f"Summer23BPixPrompt23_{'V3' if reg == '' else 'V1'}_MC",
        },
        "2024": {
            "Data": "Summer24Prompt24_V2_DATA",
            "MC": "Summer24Prompt24_V2_MC"
        },
        "2025": {
            "Data": "Winter25Prompt25_V3_DATA",
            "MC": "Winter25Prompt25_V3_MC"
        },
    }
    jec = jec_version[year][era]
    tag_jec = "_".join([jec, level, algo])
    if clipping_24:
        tag_L1 = "_".join([jec, "L1FastJet", algo])
        tag_L2 = "_".join([jec, "L2Relative", algo])
        tag_L3 = "_".join([jec, "L3Absolute", algo])
        tag_L2L3 = "_".join([jec, "L2L3Residual", algo])

    # get the correction sets
    cset = correctionlib.CorrectionSet.from_file(jerc_json[year])

    # prepare inputs
    if AK8:
        jets_jagged = events.FatJet
    else:
        jets_jagged = events.Jet

    counts = ak.num(jets_jagged)
    if ("run" not in jets_jagged.fields) and (era == "Data"):
        jets_jagged["run"] = events.run
    # backup of the original nanoaod jet pt, only for once
    if "pt_nano" not in jets_jagged.fields:
        jets_jagged["pt_nano"] = jets_jagged.pt
        jets_jagged["mass_nano"] = jets_jagged.mass
    # store the raw jet pt, only for once
    if "pt_raw" not in jets_jagged.fields:
        # Workaround until Run 2 v15, 2024 & 2025 regressed JECs are ready
        if is_Run2_v15 or year == "2024" or year == "2025":
            jets_jagged["pt_raw"] = jets_jagged.pt * (1 - jets_jagged.rawFactor)
        else:
            if reg == "":
                regFactor = 1.0
            elif reg == "PNetRegression":
                regFactor = jets_jagged.PNetRegPtRawCorr
            elif reg == "PNetRegressionPlusNeutrino":
                regFactor = (
                    jets_jagged.PNetRegPtRawCorr * jets_jagged.PNetRegPtRawCorrNeutrino
                )
            elif reg == "UParTRegression":
                regFactor = jets_jagged.UParTAK4RegPtRawCorr
            elif reg == "UParTRegressionPlusNeutrino":
                # for UParT the neutrino correction is not cumulative
                regFactor = jets_jagged.UParTAK4RegPtRawCorrNeutrino
            else:
                logger.error(f"Unknown regression algorithm: {reg}")
                exit(-1)
            jets_jagged["pt_raw"] = (
                jets_jagged.pt * (1 - jets_jagged.rawFactor) * regFactor
            )
        jets_jagged["mass_raw"] = jets_jagged.mass * (1 - jets_jagged.rawFactor)
    # avoid using hasattr(jets_jagged, "rho"). Same name as the coffea vector
    # property of rho:
    # https://github.com/CoffeaTeam/coffea/blob/0e43daf8e40ccec44efb2622777354ebd0424b84/src/coffea/nanoevents/methods/vector.py#L482
    if "rho_value" not in jets_jagged.fields:
        try:
            jets_jagged["rho_value"] = (
                ak.ones_like(jets_jagged.pt) * events.Rho.fixedGridRhoFastjetAll
            )
        except BaseException:
            # UL datasets have different naming convention
            jets_jagged["rho_value"] = (
                ak.ones_like(jets_jagged.pt) * events.fixedGridRhoFastjetAll
            )
    # create the gen_matched pt, only for once
    if ("pt_gen" not in jets_jagged.fields) and (apply_jer or jer_syst):
        # TODO: finalize the gen-matching algorithms
        # current follow coffea example:
        # https://github.com/CoffeaTeam/coffea/blob/16db8f663e40dafd2399d32862c20e3faa5542be/binder/applying_corrections.ipynb#L423
        jets_jagged["pt_gen"] = ak.fill_none(jets_jagged.matched_gen.pt, -99999)
    # create the eventid, only for once
    if ("event_id" not in jets_jagged.fields) and (apply_jer or jer_syst):
        jets_jagged["event_id"] = ak.ones_like(jets_jagged.pt) * events.event

    # flatten
    jets = ak.flatten(jets_jagged)
    # evaluate dictionary
    eval_dict = {
        "JetPt": jets.pt_raw,
        "JetEta": jets.eta,
        "JetPhi": jets.phi,
        "Rho": jets.rho_value,
        "JetA": jets.area,
        **({"run": jets.run} if (era == "Data") else {}),
    }

    # jec central
    if clipping_24:
        logger.info("[ jerc_jet ] Applying JEC corrections with L2L3Residual clipping")
        if tag_L1 in list(cset.compound.keys()):
            sf_L1 = cset.compound[tag_L1]
        elif tag_L1 in list(cset.keys()):
            sf_L1 = cset[tag_L1]
        else:
            logger.error(
                f"[ jerc_jet ] No JEC correction: {tag_L1} - Year: {year} - Era: {era} - Level: L1FastJet"
            )
            exit(-1)

        if tag_L2 in list(cset.compound.keys()):
            sf_L2 = cset.compound[tag_L2]
        elif tag_L2 in list(cset.keys()):
            sf_L2 = cset[tag_L2]
        else:
            logger.error(
                f"[ jerc_jet ] No JEC correction: {tag_L2} - Year: {year} - Era: {era} - Level: L2Relative"
            )
            exit(-1)

        if tag_L3 in list(cset.compound.keys()):
            sf_L3 = cset.compound[tag_L3]
        elif tag_L3 in list(cset.keys()):
            sf_L3 = cset[tag_L3]
        else:
            logger.error(
                f"[ jerc_jet ] No JEC correction: {tag_L3} - Year: {year} - Era: {era} - Level: L3Absolute"
            )
            exit(-1)

        if tag_L2L3 in list(cset.compound.keys()):
            sf_L2L3 = cset.compound[tag_L2L3]
        elif tag_L2L3 in list(cset.keys()):
            sf_L2L3 = cset[tag_L2L3]
        else:
            logger.error(
                f"[ jerc_jet ] No JEC correction: {tag_L2L3} - Year: {year} - Era: {era} - Level: L2L3Residual"
            )
            exit(-1)

        eval_dict_L1 = {
            "JetPt": jets.pt_raw,
            "JetEta": jets.eta,
            "Rho": jets.rho_value,
            "JetA": jets.area,
            **({"run": jets.run} if (era == "Data") else {}),
        }

        inputs_L1 = [eval_dict_L1[input.name] for input in sf_L1.inputs]
        sf_L1_value = sf_L1.evaluate(*inputs_L1)
        jets["pt_L1"] = sf_L1_value * jets["pt_raw"]
        jets["mass_L1"] = sf_L1_value * jets["mass_raw"]

        eval_dict_L2 = {
            "JetPt": jets.pt_L1,
            "JetEta": jets.eta,
            "JetPhi": jets.phi,
            **({"run": jets.run} if (era == "Data") else {}),
        }
        inputs_L2 = [eval_dict_L2[input.name] for input in sf_L2.inputs]
        sf_L2_value = sf_L2.evaluate(*inputs_L2)
        jets["pt_L2"] = sf_L2_value * jets["pt_L1"]
        jets["mass_L2"] = sf_L2_value * jets["mass_L1"]

        eval_dict_L3 = {
            "JetPt": jets.pt_L2,
            "JetEta": jets.eta,
            **({"run": jets.run} if (era == "Data") else {}),
        }
        inputs_L3 = [eval_dict_L3[input.name] for input in sf_L3.inputs]
        sf_L3_value = sf_L3.evaluate(*inputs_L3)
        jets["pt_L3"] = sf_L3_value * jets["pt_L2"]
        jets["mass_L3"] = sf_L3_value * jets["mass_L2"]

        # Clip the correction to 50 GeV if the pT of the jet is less than 50 GeV and the eta is between 2.0 and 2.5
        eval_dict_L2L3 = {
            "JetPt": ak.where(
                (jets.pt_L3 < 50) & (abs(jets.eta) > 2.0) & (abs(jets.eta) < 2.5),
                50,
                jets.pt_L3
            ),
            "JetEta": jets.eta,
            "JetPhi": jets.phi,
            **({"run": jets.run} if (era == "Data") else {}),
        }
        inputs_L2L3 = [eval_dict_L2L3[input.name] for input in sf_L2L3.inputs]
        sf_L2L3_value = sf_L2L3.evaluate(*inputs_L2L3)
        jets["pt_L2L3"] = sf_L2L3_value * jets["pt_L3"]
        jets["mass_L2L3"] = sf_L2L3_value * jets["mass_L3"]
        # update the nominal pt and mass
        jets["pt"] = jets["pt_L2L3"]
        jets["mass"] = jets["mass_L2L3"]
    elif apply_jec:
        # get the correction
        if tag_jec in list(cset.compound.keys()):
            sf = cset.compound[tag_jec]
        elif tag_jec in list(cset.keys()):
            sf = cset[tag_jec]
        else:
            logger.error(
                f"[ jerc_jet ] No JEC correction: {tag_jec} - Year: {year} - Era: {era} - Level: {level}"
            )
            exit(-1)
        inputs = [eval_dict[input.name] for input in sf.inputs]
        sf_value = sf.evaluate(*inputs)
        jets["pt_jec"] = sf_value * jets["pt_raw"]
        jets["mass_jec"] = sf_value * jets["mass_raw"]
        # update the nominal pt and mass
        jets["pt"] = jets["pt_jec"]
        jets["mass"] = jets["mass_jec"]

    # jer central and systematics
    if apply_jer or jer_syst:
        # learned from: https://github.com/cms-nanoAOD/correctionlib/issues/130
        jer_version = {
            "2016preVFP": "Summer20UL16APV_JRV3_MC",
            "2016postVFP": "Summer20UL16_JRV3_MC",
            "2017": f"Summer19UL17_JR{'V3' if is_Run2_v15 else 'V2'}_MC",
            "2018": "Summer19UL18_JRV2_MC",
            "2022preEE": "Summer22_22Sep2023_JRV1_MC",
            "2022postEE": "Summer22EE_22Sep2023_JRV1_MC",
            "2023preBPix": "Summer23Prompt23_RunCv1234_JRV1_MC",
            "2023postBPix": "Summer23BPixPrompt23_RunD_JRV1_MC",
            # This is preliminary, should be changed once files with 2024 and 2025 JER are available
            "2024": "Summer23BPixPrompt23_RunD_JRV1_MC",
            "2025": "Summer23BPixPrompt23_RunD_JRV1_MC",
        }
        jer = jer_version[year]
        jer_ptres_tag = f"{jer}_PtResolution_{algo}"
        jer_sf_tag = f"{jer}_ScaleFactor_{algo}"

        # this is a hack to make sure the JER corrections aren't applied for unmatched jets with 2.5 < |eta| < 3.0
        # by setting the gen pT to the reco pT no JER shift will be applied since this is based on the pT difference
        # TODO should be removed once a proper fix is implemented at the json level
        # see https://gitlab.cern.ch/cms-jetmet/coordination/coordination/-/issues/113
        if year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]:
            logger.warning("You are removing the JER corrections for unmatched jets with 2.5 < |eta| < 3.5. This is half of the recipe to reduce the impact of the jet horns, please make sure you are also applying the eta-dependent pT cuts in your processor.")
            pt_gen_orig = jets["pt_gen"]
            jets["pt_gen"] = ak.where((jets.pt_gen < 0) & (abs(jets.eta) > 2.5) & (abs(jets.eta) < 3.0), jets.pt, jets.pt_gen)

        ceval_jer = get_jer_correction_set(jerc_json[year], jer_ptres_tag, jer_sf_tag)

        # update evaluate dictionary
        eval_dict.update(
            {
                # JER SFs for regressed jets run on standard jets
                "JetPt": jets.pt if reg == "" else jets.pt_nano,
                "GenPt": jets.pt_gen,
                "EventID": jets.event_id,
            }
        )
        # get jer pt resolution
        inputs_jer_ptres = [
            eval_dict[input.name] for input in ceval_jer[jer_ptres_tag].inputs
        ]
        jer_ptres = ceval_jer[jer_ptres_tag].evaluate(*inputs_jer_ptres)
        # update evaluate dictionary
        eval_dict.update({"JER": jer_ptres})
        # addjust pt gen
        eval_dict.update(
            {
                "GenPt": np.where(
                    np.abs(eval_dict["JetPt"] - eval_dict["GenPt"])
                    < 3 * eval_dict["JetPt"] * eval_dict["JER"],
                    eval_dict["GenPt"],
                    -1.0,
                ),
            }
        )
        if apply_jer:
            eval_dict, jersmear = get_jersmear(eval_dict, ceval_jer, jer_sf_tag, "nom")
            jets["pt_jer"] = jets.pt * jersmear
            jets["mass_jer"] = jets.mass * jersmear
        if jer_syst:
            jetTag = "jer"
            if AK8:
                jetTag = "jer_AK8"
            # jer up
            eval_dict, jersmear = get_jersmear(eval_dict, ceval_jer, jer_sf_tag, "up")
            jets[f"pt_{jetTag}_syst_up"] = jets.pt * jersmear
            jets[f"mass_{jetTag}_syst_up"] = jets.mass * jersmear
            # jer down
            eval_dict, jersmear = get_jersmear(eval_dict, ceval_jer, jer_sf_tag, "down")
            jets[f"pt_{jetTag}_syst_down"] = jets.pt * jersmear
            jets[f"mass_{jetTag}_syst_down"] = jets.mass * jersmear
        if apply_jer:
            # to avoid the sf: jer*jer_up or jer*jer_down, update the jer
            # pt/mass after calculation of the jer up/down
            jets["pt"] = jets["pt_jer"]
            jets["mass"] = jets["mass_jer"]

        # now that the JER shift has been applied, we can restore the original gen pt
        # TODO should be removed once a proper fix is implemented at the json level, see above
        if year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]:
            jets["pt_gen"] = pt_gen_orig

    # jec systematics
    if jec_syst:
        # update evaluate dictionary
        eval_dict.update({"JetPt": jets.pt})
        jetTag = "jec"
        if AK8:
            jetTag = "jec_AK8"
        if split_jec_syst == "regrouped":
            # regrouped JEC systematics
            jec_syst_regrouped = {
                "2016preVFP": {
                    # regrouped jec uncertainty
                    "jec_syst_Absolute_2016": "Regrouped_Absolute_2016",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2016": "Regrouped_BBEC1_2016",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2016": "Regrouped_EC2_2016",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2016": "Regrouped_HF_2016",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2016": "Regrouped_RelativeSample_2016",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2016postVFP": {
                    # regrouped jec uncertainty
                    "jec_syst_Absolute_2016": "Regrouped_Absolute_2016",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2016": "Regrouped_BBEC1_2016",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2016": "Regrouped_EC2_2016",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2016": "Regrouped_HF_2016",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2016": "Regrouped_RelativeSample_2016",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2017": {
                    # regrouped jec uncertainty
                    "jec_syst_Absolute_2017": "Regrouped_Absolute_2017",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2017": "Regrouped_BBEC1_2017",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2017": "Regrouped_EC2_2017",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2017": "Regrouped_HF_2017",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2017": "Regrouped_RelativeSample_2017",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2018": {
                    # regrouped jec uncertainty
                    "jec_syst_Absolute_2018": "Regrouped_Absolute_2018",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2018": "Regrouped_BBEC1_2018",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2018": "Regrouped_EC2_2018",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2018": "Regrouped_HF_2018",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2018": "Regrouped_RelativeSample_2018",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2022preEE": {
                    "jec_syst_Absolute_2022": "Regrouped_Absolute_2022",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2022": "Regrouped_BBEC1_2022",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2022": "Regrouped_EC2_2022",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2022": "Regrouped_HF_2022",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2022": "Regrouped_RelativeSample_2022",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2022postEE": {
                    "jec_syst_Absolute_2022EE": "Regrouped_Absolute_2022EE",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2022EE": "Regrouped_BBEC1_2022EE",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2022EE": "Regrouped_EC2_2022EE",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2022EE": "Regrouped_HF_2022EE",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2022EE": "Regrouped_RelativeSample_2022EE",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2023preBPix": {
                    "jec_syst_Absolute_2023": "Regrouped_Absolute_2023",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2023": "Regrouped_BBEC1_2023",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2023": "Regrouped_EC2_2023",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2023": "Regrouped_HF_2023",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2023": "Regrouped_RelativeSample_2023",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2023postBPix": {
                    "jec_syst_Absolute_2023BPix": "Regrouped_Absolute_2023BPix",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2023BPix": "Regrouped_BBEC1_2023BPix",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2023BPix": "Regrouped_EC2_2023BPix",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2023BPix": "Regrouped_HF_2023BPix",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2023BPix": "Regrouped_RelativeSample_2023BPix",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
                "2024": {
                    "jec_syst_Absolute_2024": "Regrouped_Absolute_2024",
                    "jec_syst_Absolute": "Regrouped_Absolute",
                    "jec_syst_BBEC1_2024": "Regrouped_BBEC1_2024",
                    "jec_syst_BBEC1": "Regrouped_BBEC1",
                    "jec_syst_EC2_2024": "Regrouped_EC2_2024",
                    "jec_syst_EC2": "Regrouped_EC2",
                    "jec_syst_FlavorQCD": "Regrouped_FlavorQCD",
                    "jec_syst_HF_2024": "Regrouped_HF_2024",
                    "jec_syst_HF": "Regrouped_HF",
                    "jec_syst_RelativeBal": "Regrouped_Absolute",
                    "jec_syst_RelativeSample_2024": "Regrouped_RelativeSample_2024",
                    # total regrouped jec uncertainty
                    "jec_syst_Regrouped_Total": "Regrouped_Total",
                },
            }
            apply_split_jec_variations(
                jec_syst_regrouped[year], jec, algo, cset, year, era, eval_dict, jets, AK8
            )
        elif split_jec_syst == "full":
            # full splitting of JEC systematics
            jec_syst_full = {
                "jec_syst_AbsoluteMPFBias": "AbsoluteMPFBias",
                "jec_syst_AbsoluteScale": "AbsoluteScale",
                "jec_syst_AbsoluteStat": "AbsoluteStat",
                "jec_syst_FlavorQCD": "FlavorQCD",
                "jec_syst_Fragmentation": "Fragmentation",
                "jec_syst_PileUpDataMC": "PileUpDataMC",
                "jec_syst_PileUpPtBB": "PileUpPtBB",
                "jec_syst_PileUpPtEC1": "PileUpPtEC1",
                "jec_syst_PileUpPtEC2": "PileUpPtEC2",
                "jec_syst_PileUpPtHF": "PileUpPtHF",
                "jec_syst_PileUpPtRef": "PileUpPtRef",
                "jec_syst_RelativeFSR": "RelativeFSR",
                "jec_syst_RelativeJEREC1": "RelativeJEREC1",
                "jec_syst_RelativeJEREC2": "RelativeJEREC2",
                "jec_syst_RelativeJERHF": "RelativeJERHF",
                "jec_syst_RelativePtBB": "RelativePtBB",
                "jec_syst_RelativePtEC1": "RelativePtEC1",
                "jec_syst_RelativePtEC2": "RelativePtEC2",
                "jec_syst_RelativePtHF": "RelativePtHF",
                "jec_syst_RelativeBal": "RelativeBal",
                "jec_syst_RelativeSample": "RelativeSample",
                "jec_syst_RelativeStatEC": "RelativeStatEC",
                "jec_syst_RelativeStatFSR": "RelativeStatFSR",
                "jec_syst_RelativeStatHF": "RelativeStatHF",
                "jec_syst_SinglePionECAL": "SinglePionECAL",
                "jec_syst_SinglePionHCAL": "SinglePionHCAL",
                "jec_syst_TimePtEta": "TimePtEta",
                "jec_syst_Total": "Total",
            }
            apply_split_jec_variations(
                jec_syst_full, jec, algo, cset, year, era, eval_dict, jets, AK8
            )
        else:
            # get the total uncertainty
            tag_jec_syst = "_".join([jec, "Total", algo])
            try:
                sf = cset[tag_jec_syst]
            except BaseException:
                logger.error(
                    f"[ jerc_jet ] No JEC systematic: {tag_jec_syst} - Year: {year} - Era: {era}"
                )
                exit(-1)
            # systematics
            inputs = [eval_dict[input.name] for input in sf.inputs]
            sf_delta = sf.evaluate(*inputs)

            # divide by correction since it is already applied before
            corr_up_variation = 1 + sf_delta
            corr_down_variation = 1 - sf_delta

            jets[f"pt_{jetTag}_syst_Total_up"] = jets.pt * corr_up_variation
            jets[f"pt_{jetTag}_syst_Total_down"] = jets.pt * corr_down_variation
            jets[f"mass_{jetTag}_syst_Total_up"] = jets.mass * corr_up_variation
            jets[f"mass_{jetTag}_syst_Total_down"] = jets.mass * corr_down_variation

    jets_jagged = ak.unflatten(jets, counts)

    # Workaround until Run 2 v15, 2024 & 2025 regressed JECs are ready
    if is_Run2_v15 or year == "2024" or year == "2025":
        if reg == "":
            regFactor = 1.0
        elif reg == "PNetRegression":
            regFactor = jets_jagged.PNetRegPtRawCorr
        elif reg == "PNetRegressionPlusNeutrino":
            regFactor = (
                jets_jagged.PNetRegPtRawCorr * jets_jagged.PNetRegPtRawCorrNeutrino
            )
        elif reg == "UParTRegression":
            regFactor = jets_jagged.UParTAK4RegPtRawCorr
        elif reg == "UParTRegressionPlusNeutrino":
            # for UParT the neutrino correction is not cumulative
            regFactor = jets_jagged.UParTAK4RegPtRawCorrNeutrino
        else:
            logger.error(f"Unknown regression algorithm: {reg}")
            exit(-1)
        jets_jagged["pt"] = (
            jets_jagged.pt * (1 - jets_jagged.rawFactor) * regFactor
        )

    if AK8:
        events["FatJet"] = jets_jagged
    else:
        events["Jet"] = jets_jagged
    return events
