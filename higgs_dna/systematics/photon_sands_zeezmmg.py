"""Hgg Zee+Zmmg photon calibration, vendored without algorithm changes.

Source: cms-analysis/general/HiggsDNA, commit
3d9afad9017880c3034d7144adafdc78521550f5, EGM_SS_systematics.py.
Only the three required functions are copied, to retain legacy calibrations.
WWgg currently uses the nominal corrections; uncertainty prescriptions here
are upstream code and have not been validated for our likelihood model.
"""
import numpy as np
import awkward as ak
import correctionlib
import os
import sys
import logging
logger = logging.getLogger(__name__)

def Scale_ZeeZmmg(pt, events, year="2022postEE", is_correction=True, gaussians="2G", restriction=None, is_Zee=False):
    """
    Applies the IJazZ photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py.
    The IJazZ corrections are independent and detached from the Egamma corrections.

    Due to remaining non-closure for photons with abs(eta) > 2.1 uncertainties were increased in this regions. (2x smear and 3x scale)
    This is just a preliminary solution while the non-closure is being further investigated.
    """
    egm_object = events.Photon

    # for later unflattening:
    counts = ak.num(egm_object.pt)

    run = ak.flatten(ak.broadcast_arrays(events.run, egm_object.pt)[0])
    gain = ak.flatten(egm_object.seedGain)
    eta = ak.flatten(egm_object.ScEta)
    r9 = ak.flatten(egm_object.r9)
    # scale uncertainties are applied on the smeared pt but computed from the raw pt
    _pt = ak.flatten(egm_object.pt)

    if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        gaussians = "1G"
    elif year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]:
        gaussians = "2G"
    else:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2016preVFP\", \"2016postVFP\", \"2017\", \"2018\", \"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\", \"2024\", \"2025\"] \n Exiting. \n")
        sys.exit(1)

    valid_years_paths = {
        "2016preVFP": "EGMScalesSmearing_ZeeZmmg_RunII2016preVFP.v1.json",
        "2016postVFP": "EGMScalesSmearing_ZeeZmmg_RunII2016postVFP.v1.json",
        "2017": "EGMScalesSmearing_ZeeZmmg_RunII2017.v1.json",
        "2018": "EGMScalesSmearing_ZeeZmmg_RunII2018.v1.json",
        "2022preEE": "EGMScalesSmearing_ZeeZmmg_22232022preEE.v1.json",
        "2022postEE": "EGMScalesSmearing_ZeeZmmg_22232022postEE.v1.json",
        "2023preBPix": "EGMScalesSmearing_ZeeZmmg_22232023preBPIX.v1.json",
        "2023postBPix": "EGMScalesSmearing_ZeeZmmg_22232023postBPIX.v1.json",
        "2024": "EGMScalesSmearing_ZeeZmmg_2024.v1.json" ,
        "2025": "EGMScalesSmearing_ZeeZmmg_2025.v1.json",
    }

    path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/Hgg', valid_years_paths[year])
    try:
        cset = correctionlib.CorrectionSet.from_file(path_json)
    except OSError:
        logger.error(f"WARNING: the JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
        sys.exit(1)
    if is_Zee:
        scale_evaluator = cset.compound["Scale_Zee"]
        smear_and_syst_evaluator = cset["SmearAndSyst_Zee"]
    else:
        scale_evaluator = cset.compound["Scale"]
        smear_and_syst_evaluator = cset["SmearAndSyst_Zmmg"]

    if is_correction:
        # scale is a residual correction on data to match MC calibration. Check if is MC, throw error in this case.
        if hasattr(events, "GenPart"):
            raise ValueError("Scale corrections should only be applied to data!")

        # copy the pt branch
        egm_object['pt_raw'] = egm_object.pt
        pt_raw = ak.flatten(egm_object.pt_raw)
        correction = scale_evaluator.evaluate("scale", run, eta, r9, pt_raw, gain)

        pt_corr = pt_raw * correction
        corrected_egm_object = egm_object
        pt_corr = ak.unflatten(pt_corr, counts)
        corrected_egm_object["pt"] = pt_corr

        # saving the smearing term for data as well, needed to recompute the sigma_E/E term
        if gaussians == "2G":
            smear_and_syst_evaluator_for_rho_corr = cset["Smear1G"]
            rho_corr = ak.unflatten(smear_and_syst_evaluator_for_rho_corr.evaluate('smear', ak.flatten(pt_corr), r9, eta), counts)
        else:
            smearing = smear_and_syst_evaluator.evaluate('smear', ak.flatten(pt_corr), r9, eta)
            rho_corr = ak.unflatten(smearing, counts)

        corrected_egm_object["rho_smear"] = rho_corr

        events["Photon"] = corrected_egm_object
        return events

    else:
        pt_raw = ak.flatten(egm_object.pt_raw)
        # Note the conventions in the JSON, both `scale_up`/`scale_down` and `escale` are available.
        # scale_up = 1 + escale
        if not hasattr(events, "GenPart"):
            raise ValueError("Scale uncertainties should only be applied to MC!")

        if is_Zee:
            escale = smear_and_syst_evaluator.evaluate('escale', pt_raw, r9, eta)
        else:
            # Zee and Zmmg inputs are reversed in the JSONs
            escale = smear_and_syst_evaluator.evaluate('escale', eta, r9, pt_raw)

            # muon momentum scale uncertainties
            muon_syst = 0.0005
            escale = np.sqrt(escale**2 + muon_syst**2)

            # non-linearity uncertainties for high pt photons
            nl_EB_syst = 0.0015
            nl_EE_syst = 0.0025
            isEB_mask = ak.flatten(egm_object.isScEtaEB)
            nl_syst = np.where(isEB_mask, nl_EB_syst, nl_EE_syst)
            pt80_mask = ak.flatten(egm_object.pt) > 80
            nl_syst = np.where(pt80_mask, nl_syst, 0)
            escale = np.sqrt(escale**2 + nl_syst**2)

        corr_up_variation = 1 + escale
        corr_down_variation = 1 - escale

        if restriction == "EB":
            isEE_mask = ak.flatten(egm_object.isScEtaEE)
            corr_up_variation = np.where(isEE_mask, 1.0, corr_up_variation)
            corr_down_variation = np.where(isEE_mask, 1.0, corr_down_variation)
        elif restriction == "EE":
            isEB_mask = ak.flatten(egm_object.isScEtaEB)
            corr_up_variation = np.where(isEB_mask, 1.0, corr_up_variation)
            corr_down_variation = np.where(isEB_mask, 1.0, corr_down_variation)
        elif restriction is not None:
            logger.error("The restriction is not implemented yet! Valid options are [\"EB\", \"EE\"] \n Exiting. \n")
            sys.exit(1)

        # Coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        # scale uncertainties are applied on the smeared pt
        return np.concatenate((corr_up_variation[:, None], corr_down_variation[:, None]), axis=1) * _pt[:, None]

def double_smearing(std_normal, std_flat, mu, sigma, sigma_scale, frac, old_convention=True):
    """
    Function to compute the double Gaussian smearing

    Args:
        std_normal (np.ndarray): Standard normal distribution
        std_flat (np.ndarray): Standard flat distribution
        mu (np.ndarray): Mean of the central Gaussian
        sigma (np.ndarray): Sigma of the central Gaussian
        sigma_scale (np.ndarray): Relative sigma of the tail Gaussian ie sigma_tail = sigma_scale * sigma_central
        frac (np.ndarray): Fraction of the tail Gaussian
    Returns:
        np.ndarray: Smearing value
    """
    # Compute the two possible scale values
    scale1 = 1 + sigma * std_normal
    scale2 = mu * (1 + sigma_scale * sigma * std_normal)

    # Compute binomial selection
    binom = std_flat > frac

    return np.where(binom, scale2, scale1)

def Smearing_ZeeZmmg(pt, events, year="2022postEE", is_correction=True, gaussians="1G"):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py

    Due to remaining non-closure for photons with abs(eta) > 2.1 uncertainties were increased in this regions. (2x smear and 3x scale)
    This is just a preliminary solution while the non-closure is being further investigated.
    """
    egm_object = events.Photon
    # for later unflattening:
    counts = ak.num(egm_object.pt)

    eta = ak.flatten(egm_object.ScEta)
    r9 = ak.flatten(egm_object.r9)
    seediEtaOriX = ak.flatten(egm_object.seediEtaOriX)
    seediPhiOriY = ak.flatten(egm_object.seediPhiOriY)
    # Need some broadcasting to make the event numbers match
    event_number = ak.flatten(ak.broadcast_arrays(events.event, egm_object.pt)[0])

    if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        gaussians = "1G"
    elif year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024", "2025"]:
        gaussians = "2G"
    else:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2016preVFP\", \"2016postVFP\", \"2017\", \"2018\", \"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\", \"2024\", \"2025\"] \n Exiting. \n")
        sys.exit(1)

    valid_years_paths = {
        "2016preVFP": "EGMScalesSmearing_ZeeZmmg_RunII2016preVFP.v1.json",
        "2016postVFP": "EGMScalesSmearing_ZeeZmmg_RunII2016postVFP.v1.json",
        "2017": "EGMScalesSmearing_ZeeZmmg_RunII2017.v1.json",
        "2018": "EGMScalesSmearing_ZeeZmmg_RunII2018.v1.json",
        "2022preEE": "EGMScalesSmearing_ZeeZmmg_22232022preEE.v1.json",
        "2022postEE": "EGMScalesSmearing_ZeeZmmg_22232022postEE.v1.json",
        "2023preBPix": "EGMScalesSmearing_ZeeZmmg_22232023preBPIX.v1.json",
        "2023postBPix": "EGMScalesSmearing_ZeeZmmg_22232023postBPIX.v1.json",
        "2024": "EGMScalesSmearing_ZeeZmmg_2024.v1.json" ,
        "2025": "EGMScalesSmearing_ZeeZmmg_2025.v1.json",
    }

    path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing/Hgg', valid_years_paths[year])
    try:
        cset = correctionlib.CorrectionSet.from_file(path_json)
    except OSError:
        logger.error(f"The JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
        sys.exit(1)
    smear_and_syst_evaluator = cset["SmearAndSyst_Zee"]
    random_generator = cset['EGMRandomGenerator']

    # In theory, the energy should be smeared and not the pT, see: https://mattermost.web.cern.ch/cmseg/channels/egm-ss/6mmucnn8rjdgt8x9k5zaxbzqyh
    # However, there is a linear proportionality between pT and E: E = pT * cosh(eta)
    # Because of that, applying the correction to pT and E is equivalent (since eta does not change)
    # Energy is provided as a LorentzVector mixin, so we choose to correct pT
    # Also holds true for the scale part

    # Calculate upfront since it is needed for both correction and uncertainty
    random_numbers = random_generator.evaluate('stdnormal', event_number, seediEtaOriX, seediPhiOriY)

    if is_correction:
        # copy the pt branch
        egm_object['pt_raw'] = egm_object.pt
        pt_raw = ak.flatten(egm_object.pt_raw)
        smearing = smear_and_syst_evaluator.evaluate('smear', pt_raw, r9, eta)

        if gaussians == "1G":
            correction = (1 + smearing * random_numbers)
            rho_corr = ak.unflatten(smearing, counts)

        # Else can only be "2G" due to the checks above
        # Have to use else here to satisfy that correction is always defined in all possible branches of the code
        else:
            correction = double_smearing(
                random_numbers,
                random_generator.evaluate('stdflat', event_number, seediEtaOriX, seediPhiOriY),
                smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, eta),
                smearing,
                smear_and_syst_evaluator.evaluate('reso_scale', pt_raw, r9, eta),
                smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, eta),
            )
            # For the 2G case, also take the rho_corr from the 1G case as advised by Fabrice
            smear_and_syst_evaluator_for_rho_corr = cset["Smear1G"]
            rho_corr = ak.unflatten(smear_and_syst_evaluator_for_rho_corr.evaluate('smear', pt_raw, r9, eta), counts)

        pt_corr = pt_raw * correction
        corrected_egm_object = egm_object
        pt_corr = ak.unflatten(pt_corr, counts)

        # If it is data, dont perform the pt smearing, only save the std of the gaussian for each event!
        if hasattr(events, "GenPart"):  # this operation is here because if there is no "events.GenPart" field on data, an error will be thrown and we go to the except - so we dont smear the data pt spectrum
            corrected_egm_object["pt"] = pt_corr
            corrected_egm_object["rho_smear"] = rho_corr
            events["Photon"] = corrected_egm_object
            return events
        else:
            logger.error("Smearing corrections should only be applied to data! \n Exiting. \n")

    else:
        pt_raw = ak.flatten(egm_object.pt_raw)
        # Note the conventions in the JSON, both `smear_up`/`smear_down` and `esmear` are available.
        # smear_up = smear + esmear
        if gaussians == "1G":
            corr_up_variation = 1 + smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, eta) * random_numbers
            corr_down_variation = 1 + np.maximum(0.0, smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, eta)) * random_numbers

        else:
            # For 2G, need to recompute the smearing with the varied parameters
            def scale_smear_unc(factor_ee, pt_raw, r9, eta):

                # Region masks, for the current 2024 SAS there is an non-clousure for photons in this region (eta > 2.1)
                # So uncertainties are artificially enlarged to cover remaining mismodeling
                mask_high_ee = (eta > 2.1)

                # evaluate once
                smear = smear_and_syst_evaluator.evaluate('smear', pt_raw, r9, eta)
                esmear = smear_and_syst_evaluator.evaluate('esmear', pt_raw, r9, eta)

                # default factors: EB=1, gap=1, EE=factor_ee
                fac = 1.0 * esmear
                smear_up = smear + fac
                smear_down = smear - fac

                # apply EE inflation only where mask_ee is True
                smear_up = np.where(mask_high_ee, smear + factor_ee * esmear, smear_up)
                smear_down = np.where(mask_high_ee, smear - factor_ee * esmear, smear_down)

                return smear_up, smear_down

            if year in ["2024", "2025"]:
                smear_up, smear_down = scale_smear_unc(2.0, pt_raw, r9, eta)
            else:
                smear_up = smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, eta)
                smear_down = smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, eta)

            corr_up_variation = double_smearing(
                random_numbers,
                random_generator.evaluate('stdflat', event_number, seediEtaOriX, seediPhiOriY),
                smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, eta),
                smear_up,
                smear_and_syst_evaluator.evaluate('reso_scale', pt_raw, r9, eta),
                smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, eta)
            )

            corr_down_variation = double_smearing(
                random_numbers,
                random_generator.evaluate('stdflat', event_number, seediEtaOriX, seediPhiOriY),
                smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, eta),
                np.maximum(0.0, smear_down),
                smear_and_syst_evaluator.evaluate('reso_scale', pt_raw, r9, eta),
                smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, eta)
            )

        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        # smearing uncertainties are applied on the raw pt because the smearing is redone from scratch
        return np.concatenate((corr_up_variation[:, None], corr_down_variation[:, None]), axis=1) * pt_raw[:, None]
