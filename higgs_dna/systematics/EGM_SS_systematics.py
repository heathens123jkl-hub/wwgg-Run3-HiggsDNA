import numpy as np
import awkward as ak
import correctionlib
import os
import sys
import logging

logger = logging.getLogger(__name__)


# Not nice but working: if the functions are called in the base processor by Photon.add_systematic(... "what"="pt"...), the pt is passed to the function as first argument.
# I need the full events here, so I pass in addition the events. Seems to only work if it is explicitly a function of pt, but I might be missing something. Open for better solutions.
def EGM_Scale_Trad(pt, events, year="2022postEE", is_correction=True, restriction=None, is_electron=False):
    """
    Applies the photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py
    """
    if year in ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]:
        if is_electron:
            json_name = f"electronSS_EtDependent_{year}.json"
            egm_object = events.Electron
        else:
            json_name = f"photonSS_EtDependent_{year}.json"
            egm_object = events.Photon

        if hasattr(egm_object, "eCorr"):
            logger.error("The correction is no longer compatible with NanoAODv9 Run2UL samples \n Exiting. \n")
            sys.exit(1)
    else:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2016preVFP\", \"2016postVFP\", \"2017\", \"2018\", \"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\", \"2024\"] \n Exiting. \n")
        sys.exit(1)

    # for later unflattening:
    counts = ak.num(egm_object.pt)

    run = ak.flatten(ak.broadcast_arrays(events.run, egm_object.pt)[0])
    gain = ak.flatten(egm_object.seedGain)
    SCeta = ak.flatten(egm_object.ScEta)
    r9 = ak.flatten(egm_object.r9)
    pt_raw = ak.flatten(egm_object.pt_raw)

    path_json = os.path.join(os.path.dirname(__file__), f'JSONs/scaleAndSmearing/{json_name}')
    try:
        cset = correctionlib.CorrectionSet.from_file(path_json)
        scale_evaluator = cset.compound["Scale"]
        smear_and_syst_evaluator = cset["SmearAndSyst"]
    except:
        logger.error(f"WARNING: the JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
        sys.exit(1)

    if is_correction:
        # scale is a residual correction on data to match MC calibration. Check if is MC, throw error in this case.
        if hasattr(events, "GenPart"):
            raise ValueError("Scale corrections should only be applied to data!")

        correction = scale_evaluator.evaluate("scale", run, SCeta, r9, pt_raw, gain)
        pt_corr = pt_raw * correction

        corrected_egm_object = egm_object
        pt_corr = ak.unflatten(pt_corr, counts)
        corrected_egm_object["pt"] = pt_corr

        if is_electron:
            events["Electron"] = corrected_egm_object
        else:
            events["Photon"] = corrected_egm_object

        return events

    else:
        if not hasattr(events, "GenPart"):
            raise ValueError("Scale uncertainties should only be applied to MC!")

        if is_electron:
            uncertainty_up = smear_and_syst_evaluator.evaluate("scale_up", pt_raw, r9, SCeta)
            uncertainty_down = smear_and_syst_evaluator.evaluate("scale_down", pt_raw, r9, SCeta)
        else:
            # Conservative scale uncertainties without Zmmg corrections
            if year in ["2016preVFP", "2016postVFP", "2017", "2018"]:
                uncertainty_up = 1.005 * np.ones_like(ak.to_numpy(pt_raw))
                uncertainty_down = 0.995 * np.ones_like(ak.to_numpy(pt_raw))
                logger.warning("Using conservative scale uncertainties of 0.5% to cover electron/photon energy scale discrepancies for Run2 samples \n")
            else:
                uncertainty_up = 1.01 * np.ones_like(ak.to_numpy(pt_raw))
                uncertainty_down = 0.99 * np.ones_like(ak.to_numpy(pt_raw))
                logger.warning("Using conservative scale uncertainties of 1% to cover electron/photon energy scale discrepancies for Run3 samples \n")

        # Apply restriction if needed
        if restriction is not None:
            if restriction == "EB":
                uncMask = ak.to_numpy(ak.flatten(egm_object.isScEtaEB))

            elif restriction == "EE":
                uncMask = ak.to_numpy(ak.flatten(egm_object.isScEtaEE))

            uncertainty_up = np.where(uncMask, uncertainty_up, np.zeros_like(uncertainty_up))
            uncertainty_down = np.where(uncMask, uncertainty_down, np.zeros_like(uncertainty_down))

        corr_up_variation = uncertainty_up
        corr_down_variation = uncertainty_down

        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return np.concatenate((corr_up_variation[:, None], corr_down_variation[:, None]), axis=1) * pt_raw[:, None]


def EGM_Smearing_Trad(pt, events, year="2022postEE", is_correction=True, is_electron=False):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py
    """
    if year in ["2016preVFP", "2016postVFP", "2017", "2018", "2022preEE", "2022postEE", "2023preBPix", "2023postBPix", "2024"]:
        if is_electron:
            json_name = f"electronSS_EtDependent_{year}.json"
            egm_object = events.Electron
        else:
            json_name = f"photonSS_EtDependent_{year}.json"
            egm_object = events.Photon

        if hasattr(egm_object, "eCorr"):
            logger.error("The correction is no longer compatible with NanoAODv9 Run2UL samples \n Exiting. \n")
            sys.exit(1)

    else:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2016preVFP\", \"2016postVFP\", \"2017\", \"2018\", \"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\", \"2024\"] \n Exiting. \n")
        sys.exit(1)

    # for later unflattening:
    counts = ak.num(egm_object.pt)

    SCeta = ak.flatten(egm_object.ScEta)
    r9 = ak.flatten(egm_object.r9)
    pt_raw = ak.flatten(egm_object.pt_raw)

    path_json = os.path.join(os.path.dirname(__file__), f'JSONs/scaleAndSmearing/{json_name}')
    try:
        cset = correctionlib.CorrectionSet.from_file(path_json)
        smear_and_syst_evaluator = cset["SmearAndSyst"]
    except:
        logger.error(f"WARNING: the JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
        sys.exit(1)

    # we need reproducible random numbers since in the systematics call, the previous correction needs to be cancelled out
    if len(SCeta) > 0:
        seed = abs(np.float32(SCeta[0]).view("int32"))
    else:
        seed = 42
    rng = np.random.default_rng(seed=seed)

    smearing = smear_and_syst_evaluator.evaluate('smear', pt_raw, r9, SCeta)

    if is_correction:
        smearing_factor = rng.normal(loc=1., scale=smearing)
        pt_corr = pt_raw * smearing_factor
        corrected_egm_object = egm_object
        pt_corr = ak.unflatten(pt_corr, counts)
        rho_corr = ak.unflatten(smearing, counts)

        # If it is data, dont perform the pt smearing, only save the std of the gaussian for each event!
        if hasattr(events, "GenPart"):  # this operation is here because if there is no "events.GenPart" field on data, an error will be thrown and we go to the except - so we dont smear the data pt spectrum
            corrected_egm_object["pt"] = pt_corr

        corrected_egm_object["rho_smear"] = rho_corr

        if is_electron:
            events["Electron"] = corrected_egm_object
        else:
            events["Photon"] = corrected_egm_object
        return events

    else:

        smear_up = smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, SCeta)
        smear_down = smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, SCeta)

        corr_up_variation = rng.normal(loc=1., scale=smear_up)
        corr_down_variation = rng.normal(loc=1., scale=smear_down)

        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        return np.concatenate((corr_up_variation[:, None], corr_down_variation[:, None]), axis=1) * pt_raw[:, None]


def EGM_Scale_IJazZ(pt, events, year="2022postEE", is_correction=True, gaussians="1G", restriction=None, is_electron=False):
    """
    Applies the IJazZ photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py.
    The IJazZ corrections are independent and detached from the Egamma corrections.

    Due to remaining non-closure for photons with abs(eta) > 2.1 uncertainties were increased in this regions. (2x smear and 3x scale)
    This is just a preliminary solution while the non-closure is being further investigated.
    """
    use_mvaID = False
    if is_electron:
        object_type = "Ele"
        egm_object = events.Electron
    else:
        object_type = "Pho"
        egm_object = events.Photon
        # adding mvaID dependence for 2024 https://indico.cern.ch/event/1499928/contributions/6503638/attachments/3067375/5426071/Hgg_250515_SaS2024.pdf
        if year in ["2024"]:
            use_mvaID = True

    use_absScEta = year != "2025"
    # for later unflattening:
    counts = ak.num(egm_object.pt)

    run = ak.flatten(ak.broadcast_arrays(events.run, egm_object.pt)[0])
    gain = ak.flatten(egm_object.seedGain)
    eta = ak.flatten(egm_object.ScEta)
    AbsScEta = abs(eta)
    r9 = ak.flatten(egm_object.r9)
    # 2024 photon-SaS are derived as function of mvaID
    if use_mvaID:
        mvaID = ak.flatten(egm_object.mvaID)
    # scale uncertainties are applied on the smeared pt but computed from the raw pt
    pt_raw = ak.flatten(egm_object.pt_raw)
    _pt = ak.flatten(egm_object.pt)

    if gaussians == "1G":
        gaussian_postfix = ""
    elif gaussians == "2G":
        gaussian_postfix = "2G"
    else:
        logger.error("The selected number of gaussians is not implemented yet! Valid options are [\"1G\", \"2G\"] \n Exiting. \n")
        sys.exit(1)

    valid_years_paths = {
        "2022preEE": f"EGMScalesSmearing_{object_type}_2022preEE",
        "2022postEE": f"EGMScalesSmearing_{object_type}_2022postEE",
        "2023preBPix": f"EGMScalesSmearing_{object_type}_2023preBPIX",
        "2023postBPix": f"EGMScalesSmearing_{object_type}_2023postBPIX",
        "2024": f"EGMScalesSmearing_{object_type}_2024" + ("_mvaID" if use_mvaID else ""),
        "2025": f"EGMScalesSmearing_{object_type}_2025",
    }

    ending = ".v1.json"

    if year not in valid_years_paths:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\", \"2024\", \"2025\"] \n Exiting. \n")
        sys.exit(1)

    else:
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing', valid_years_paths[year] + gaussian_postfix + ending)
        try:
            cset = correctionlib.CorrectionSet.from_file(path_json)
        except:
            logger.error(f"WARNING: the JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
            sys.exit(1)
        # Convention of Fabrice and Paul: Capitalise IX (for some reason)
        if "BPix" in year:
            year = year.replace("BPix", "BPIX")
        scale_evaluator = cset.compound[f"EGMScale_Compound_{object_type}_{year}{gaussian_postfix}"]
        smear_and_syst_evaluator = cset[f"EGMSmearAndSyst_{object_type}PTsplit_{year}{gaussian_postfix}"]

    if is_correction:
        # scale is a residual correction on data to match MC calibration. Check if is MC, throw error in this case.
        if hasattr(events, "GenPart"):
            raise ValueError("Scale corrections should only be applied to data!")
        if use_mvaID:
            correction = scale_evaluator.evaluate("scale", run, eta, r9, AbsScEta, mvaID, pt_raw, gain)
        elif use_absScEta:
            correction = scale_evaluator.evaluate("scale", run, eta, r9, AbsScEta, pt_raw, gain)
        else:
            correction = scale_evaluator.evaluate("scale", run, eta, r9, pt_raw, gain)
        pt_corr = pt_raw * correction
        corrected_egm_object = egm_object
        pt_corr = ak.unflatten(pt_corr, counts)
        corrected_egm_object["pt"] = pt_corr

        if is_electron:
            events["Electron"] = corrected_egm_object
        else:
            events["Photon"] = corrected_egm_object
        return events

    else:
        # Note the conventions in the JSON, both `scale_up`/`scale_down` and `escale` are available.
        # scale_up = 1 + escale
        if not hasattr(events, "GenPart"):
            raise ValueError("Scale uncertainties should only be applied to MC!")

        if is_electron:
            escale = smear_and_syst_evaluator.evaluate('escale', pt_raw, r9, AbsScEta)
        else:
            escale = 0.01 * np.ones_like(ak.to_numpy(pt_raw))  # use 1% uncertainties for all photons
            logger.warning("Using conservative scale uncertainties of 1% to cover electron/photon energy scale discrepancies for Run3 samples \n")

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

        # Increasing uncertainties (by a factor 3) for 2024 in the high endcap region (|eta|>2.1) due to non-closure
        if year in ["2024", "2025"] and restriction != "EB" and is_electron:
            high_endcap_mask = ak.flatten(abs(egm_object.ScEta) > 2.1)
            corr_up_variation = np.where(high_endcap_mask, 1 + 3.0 * escale, corr_up_variation)
            corr_down_variation = np.where(high_endcap_mask, 1 - 3.0 * escale, corr_down_variation)

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
    if old_convention:
        # old convention use reso2 instead of sigma_scale
        scale2 = mu * (1 + sigma_scale * std_normal)
    else:
        scale2 = mu * (1 + sigma_scale * sigma * std_normal)

    # Compute binomial selection
    binom = std_flat > frac

    return np.where(binom, scale2, scale1)


def EGM_Smearing_IJazZ(pt, events, year="2022postEE", is_correction=True, gaussians="1G", is_electron=False):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py

    Due to remaining non-closure for photons with abs(eta) > 2.1 uncertainties were increased in this regions. (2x smear and 3x scale)
    This is just a preliminary solution while the non-closure is being further investigated.
    """
    use_mvaID = False
    if is_electron:
        object_type = "Ele"
        egm_object = events.Electron
    else:
        object_type = "Pho"
        egm_object = events.Photon
        # adding mvaID dependence for 2024 https://indico.cern.ch/event/1499928/contributions/6503638/attachments/3067375/5426071/Hgg_250515_SaS2024.pdf
        # change in the gaussian tail resolution convention
        if year in ["2024"]:
            use_mvaID = True

    old_smearing_convention = year in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]

    # for later unflattening:
    counts = ak.num(egm_object.pt)

    eta = ak.flatten(egm_object.ScEta)
    AbsScEta = abs(eta)
    r9 = ak.flatten(egm_object.r9)
    pt_raw = ak.flatten(egm_object.pt_raw)
    # Need some broadcasting to make the event numbers match
    event_number = ak.flatten(ak.broadcast_arrays(events.event, egm_object.pt)[0])

    if gaussians == "1G":
        gaussian_postfix = ""
    elif gaussians == "2G":
        gaussian_postfix = "2G"
    else:
        logger.error("The selected number of gaussians is not implemented yet! Valid options are [\"1G\", \"2G\"] \n Exiting. \n")
        sys.exit(1)

    valid_years_paths = {
        "2022preEE": f"EGMScalesSmearing_{object_type}_2022preEE",
        "2022postEE": f"EGMScalesSmearing_{object_type}_2022postEE",
        "2023preBPix": f"EGMScalesSmearing_{object_type}_2023preBPIX",
        "2023postBPix": f"EGMScalesSmearing_{object_type}_2023postBPIX",
        "2024": f"EGMScalesSmearing_{object_type}_2024" + ("_mvaID" if use_mvaID else ""),
        "2025": f"EGMScalesSmearing_{object_type}_2025",
    }

    ending = ".v1.json"

    if year not in valid_years_paths:
        logger.error("The correction for the selected year is not implemented yet! Valid year tags are [\"2022preEE\", \"2022postEE\", \"2023preBPix\", \"2023postBPix\", \"2024\", \"2025\"] \n Exiting. \n")
        sys.exit(1)

    else:
        path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing', valid_years_paths[year] + gaussian_postfix + ending)
        try:
            cset = correctionlib.CorrectionSet.from_file(path_json)
        except:
            logger.error(f"The JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
            sys.exit(1)
        # Convention of Fabrice and Paul: Capitalise IX (for some reason)
        if "BPix" in year:
            year_ = year.replace("BPix", "BPIX")
        else:
            year_ = year
        smear_and_syst_evaluator = cset[f"EGMSmearAndSyst_{object_type}PTsplit_{year_}{gaussian_postfix}"]
        random_generator = cset['EGMRandomGenerator']

    # In theory, the energy should be smeared and not the pT, see: https://mattermost.web.cern.ch/cmseg/channels/egm-ss/6mmucnn8rjdgt8x9k5zaxbzqyh
    # However, there is a linear proportionality between pT and E: E = pT * cosh(eta)
    # Because of that, applying the correction to pT and E is equivalent (since eta does not change)
    # Energy is provided as a LorentzVector mixin, so we choose to correct pT
    # Also holds true for the scale part

    # Calculate upfront since it is needed for both correction and uncertainty
    smearing = smear_and_syst_evaluator.evaluate('smear', pt_raw, r9, AbsScEta)
    random_numbers = random_generator.evaluate('stdnormal', pt_raw, r9, AbsScEta, event_number)

    if gaussians == "1G":
        correction = (1 + smearing * random_numbers)
    # Else can only be "2G" due to the checks above
    # Have to use else here to satisfy that correction is always defined in all possible branches of the code
    else:
        correction = double_smearing(
            random_numbers,
            random_generator.evaluate('stdflat', pt_raw, r9, AbsScEta, event_number),
            smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, AbsScEta),
            smearing,
            smear_and_syst_evaluator.evaluate('reso2' if old_smearing_convention else 'reso_scale', pt_raw, r9, AbsScEta),
            smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, AbsScEta),
            old_convention=old_smearing_convention
        )

    if is_correction:
        pt_corr = pt_raw * correction
        corrected_egm_object = egm_object
        pt_corr = ak.unflatten(pt_corr, counts)
        # For the 2G case, also take the rho_corr from the 1G case as advised by Fabrice
        # Otherwise, the sigma_m/m will be lower on average, new CDFs will be needed etc. not worth the hassle
        if gaussians == "2G":
            path_json = os.path.join(os.path.dirname(__file__), 'JSONs/scaleAndSmearing', valid_years_paths[year] + ending)
            try:
                cset = correctionlib.CorrectionSet.from_file(path_json)
            except:
                logger.error(f"The JSON file {path_json} could not be found! \n Check if the file has been pulled \n pull_files.py -t SS-IJazZ \n")
                sys.exit(1)
            if "BPix" in year:
                year = year.replace("BPix", "BPIX")
            smear_and_syst_evaluator_for_rho_corr = cset[f"EGMSmearAndSyst_{object_type}PTsplit_{year}"]
            rho_corr = ak.unflatten(smear_and_syst_evaluator_for_rho_corr.evaluate('smear', pt_raw, r9, AbsScEta), counts)
        else:
            rho_corr = ak.unflatten(smearing, counts)

        # If it is data, dont perform the pt smearing, only save the std of the gaussian for each event!
        if hasattr(events, "GenPart"):  # this operation is here because if there is no "events.GenPart" field on data, an error will be thrown and we go to the except - so we dont smear the data pt spectrum
            corrected_egm_object["pt"] = pt_corr

        corrected_egm_object["rho_smear"] = rho_corr

        if is_electron:
            events["Electron"] = corrected_egm_object
        else:
            events["Photon"] = corrected_egm_object

        return events

    else:
        # Note the conventions in the JSON, both `smear_up`/`smear_down` and `esmear` are available.
        # smear_up = smear + esmear
        if gaussians == "1G":
            corr_up_variation = 1 + smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, AbsScEta) * random_numbers
            corr_down_variation = 1 + np.maximum(0.0, smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, AbsScEta)) * random_numbers

        else:
            # For 2G, need to recompute the smearing with the varied parameters
            def scale_smear_unc(factor_ee, pt_raw, r9, AbsScEta, min_width=1e-6):

                # Region masks, for the current 2024 SAS there is an non-clousure for photons in this region (eta > 2.1)
                # So uncertainties are artificially enlarged to cover remaining mismodeling
                mask_high_ee = (AbsScEta > 2.1)

                # evaluate once
                smear = smear_and_syst_evaluator.evaluate('smear', pt_raw, r9, AbsScEta)
                esmear = smear_and_syst_evaluator.evaluate('esmear', pt_raw, r9, AbsScEta)

                # default factors: EB=1, gap=1, EE=factor_ee
                fac = 1.0 * esmear
                smear_up = smear + fac
                smear_down = smear - fac

                # apply EE inflation only where mask_ee is True
                smear_up = np.where(mask_high_ee, smear + factor_ee * esmear, smear_up)
                smear_down = np.where(mask_high_ee, smear - factor_ee * esmear, smear_down)

                return smear_up, smear_down

            if year in ["2024", "2025"]:
                smear_up, smear_down = scale_smear_unc(2.0, pt_raw, r9, AbsScEta)
            else:
                smear_up = smear_and_syst_evaluator.evaluate('smear_up', pt_raw, r9, AbsScEta)
                smear_down = smear_and_syst_evaluator.evaluate('smear_down', pt_raw, r9, AbsScEta)

            corr_up_variation = double_smearing(
                random_numbers,
                random_generator.evaluate('stdflat', pt_raw, r9, AbsScEta, event_number),
                smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, AbsScEta),
                smear_up,
                smear_and_syst_evaluator.evaluate('reso2' if old_smearing_convention else 'reso_scale', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, AbsScEta),
                old_convention=old_smearing_convention
            )

            corr_down_variation = double_smearing(
                random_numbers,
                random_generator.evaluate('stdflat', pt_raw, r9, AbsScEta, event_number),
                smear_and_syst_evaluator.evaluate('mu', pt_raw, r9, AbsScEta),
                np.maximum(0.0, smear_down),
                smear_and_syst_evaluator.evaluate('reso2' if old_smearing_convention else 'reso_scale', pt_raw, r9, AbsScEta),
                smear_and_syst_evaluator.evaluate('frac', pt_raw, r9, AbsScEta),
                old_convention=old_smearing_convention
            )

        # coffea does the unflattenning step itself and sets this value as pt of the up/down variations
        # smearing uncertainties are applied on the raw pt because the smearing is redone from scratch
        return np.concatenate((corr_up_variation[:, None], corr_down_variation[:, None]), axis=1) * pt_raw[:, None]
