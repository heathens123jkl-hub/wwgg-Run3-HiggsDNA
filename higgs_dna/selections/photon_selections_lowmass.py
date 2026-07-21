import awkward as ak
import numpy as np

# photon preselection for Run3 -> take as input nAOD Photon collection and return the Photons that pass
# cuts (pt, eta, sieie, mvaID, iso... etc)
#


def get_corrected_pfPhoIso03(self, photon_abs_eta, pfPhoIso03, rho):
    """
    Calculate corrected pfPhoIso03 values based on photon eta regions.

    Correction formula: pfPhoIso03_corrected = pfPhoIso03 - (rho * EA1) - (rho^2 * EA2)
    """

    # Define eta regions and their corresponding EA coefficients
    eta_regions = [
        # EB regions
        {"min": -1.0, "max": 1.0, "EA1": self.EA1_EB1, "EA2": self.EA2_EB1},
        {"min": 1.0, "max": 1.4442, "EA1": self.EA1_EB2, "EA2": self.EA2_EB2},
        # EE regions
        {"min": 1.566, "max": 2.0, "EA1": self.EA1_EE1, "EA2": self.EA2_EE1},
        {"min": 2.0, "max": 2.2, "EA1": self.EA1_EE2, "EA2": self.EA2_EE2},
        {"min": 2.2, "max": 2.3, "EA1": self.EA1_EE3, "EA2": self.EA2_EE3},
        {"min": 2.3, "max": 2.4, "EA1": self.EA1_EE4, "EA2": self.EA2_EE4},
        {"min": 2.4, "max": 2.5, "EA1": self.EA1_EE5, "EA2": self.EA2_EE5},
    ]

    # Apply corrections for each eta region
    for region in eta_regions:
        mask = (photon_abs_eta > region["min"]) & (photon_abs_eta < region["max"])
        correction = (rho * region["EA1"]) + (rho * rho * region["EA2"])
        pfPhoIso03 = ak.where(mask, pfPhoIso03 - correction, pfPhoIso03)

    return pfPhoIso03


def get_corrected_ecalPFClusterIso(photon_abs_eta, ecalPFClusterIso, rho):
    """
    Calculate corrected ecalPFClusterIso values based on photon eta regions.

    Correction formula: ecalPFClusterIso_corrected = ecalPFClusterIso - (rho * EA1) - (rho^2 * EA2)
    """

    # Define eta regions and their corresponding EA coefficients
    eta_regions = [
        # EB regions
        {"min": -1.0, "max": 1.0, "EA1": 0.0866519, "EA2": -0.000229628},
        {"min": 1.0, "max": 1.469, "EA1": 0.0730397, "EA2": -0.000213401},
        # EE regions
        {"min": 1.469, "max": 2.0, "EA1": 0.0542479, "EA2": -0.000109338},
        {"min": 2.0, "max": 2.2, "EA1": 0.0486181, "EA2": -6.20977e-05},
        {"min": 2.2, "max": 2.3, "EA1": 0.0412923, "EA2": 9.63732e-06},
        {"min": 2.3, "max": 2.4, "EA1": 0.03555, "EA2": 5.79549e-05},
        {"min": 2.4, "max": 2.5, "EA1": 0.0360895, "EA2": 7.28546e-06},
    ]

    # Apply corrections for each eta region
    for region in eta_regions:
        mask = (photon_abs_eta > region["min"]) & (photon_abs_eta < region["max"])
        correction = (rho * region["EA1"]) + (rho * rho * region["EA2"])
        ecalPFClusterIso = ak.where(mask, ecalPFClusterIso - correction, ecalPFClusterIso)

    return ecalPFClusterIso


def photon_preselection_lowmass(
    self, photons: ak.Array, events: ak.Array, year="2023"
) -> ak.Array:
    """
    Apply preselection cuts to photons.
    Note that these selections are applied on each photon, it is not based on the diphoton pair.
    """
    # hlt-mimicking cuts
    rho = events.Rho.fixedGridRhoAll * ak.ones_like(photons.pt)
    photon_abs_eta = np.abs(photons.eta)
    if year in ["2016", "2016PreVFP", "2016PostVFP", "2017", "2018"]:
        # Run 2, use standard photon preselection
        photons["pfPhoIso03_rho_corrected"] = ak.where(
            photon_abs_eta < self.eta_rho_corr,
            photons.pfPhoIso03 - rho * self.low_eta_rho_corr,
            photons.pfPhoIso03 - rho * self.high_eta_rho_corr,
        )
        pass_phoIso_rho_corr_EB = (photons.isScEtaEB) & (
            photons["pfPhoIso03_rho_corrected"] < self.max_pho_iso_EB_low_r9
        )
        pass_phoIso_rho_corr_EE = (photons.isScEtaEE) & (
            photons["pfPhoIso03_rho_corrected"] < self.max_pho_iso_EE_low_r9
        )
    else:
        # quadratic EA corrections in Run3 : https://indico.cern.ch/event/1204277/contributions/5064356/attachments/2538496/4369369/CutBasedPhotonID_20221031.pdf
        photons["pfPhoIso03_rho_corrected"] = get_corrected_pfPhoIso03(
            self, photon_abs_eta, photons.pfPhoIso03, rho
        )

        pass_phoIso_rho_corr_EB = (photons.isScEtaEB) & (
            photons["pfPhoIso03_rho_corrected"] < self.max_pho_iso_EB_low_r9
        )
        pass_phoIso_rho_corr_EE = (photons.isScEtaEE) & (
            photons["pfPhoIso03_rho_corrected"] < self.max_pho_iso_EE_low_r9
        )
        # * also get rho corrected ECalIso: only make sense for run3
        photons["ecalPFClusterIso_rho_corrected"] = get_corrected_ecalPFClusterIso(
            photon_abs_eta, photons.ecalPFClusterIso, rho
        )

    trk_iso = (
        photons.trkSumPtHollowConeDR03
        if hasattr(photons, "trkSumPtHollowConeDR03")
        else photons.pfChargedIsoPFPV
    )  # photons.pfChargedIsoPFPV for v11, photons.trkSumPtHollowConeDR03 v12 and above
    rel_iso = (
        photons.pfRelIso03_chg
        if hasattr(photons, "pfRelIso03_chg")
        else photons.pfRelIso03_chg_quadratic
    )  # photons.pfRelIso03_chg for v11?, photons.pfRelIso03_chg_quadratic v12 and above

    isEB_high_r9 = (
        (photons.isScEtaEB)
        & (photons.r9 > self.min_full5x5_r9_EB_high_r9)
        & (trk_iso < self.max_trkSumPtHollowConeDR03_EB_low_r9)
        & (photons.sieie < self.max_sieie_EB_low_r9)
        & (pass_phoIso_rho_corr_EB)
    )
    isEB_low_r9 = (
        (photons.isScEtaEB)
        & (photons.r9 > self.min_full5x5_r9_EB_low_r9)
        & (photons.r9 < self.min_full5x5_r9_EB_high_r9)
        & (trk_iso < self.max_trkSumPtHollowConeDR03_EB_low_r9)
        & (photons.sieie < self.max_sieie_EB_low_r9)
        & (pass_phoIso_rho_corr_EB)
    )
    isEE_high_r9 = (
        (photons.isScEtaEE)
        & (photons.r9 > self.min_full5x5_r9_EE_high_r9)
        & (trk_iso < self.max_trkSumPtHollowConeDR03_EE_low_r9)
        & (photons.sieie < self.max_sieie_EE_low_r9)
        & (pass_phoIso_rho_corr_EE)
    )

    return photons[
        (photons.pt > self.min_pt_photon)
        & (photons.isScEtaEB | photons.isScEtaEE)
        & (photons.mvaID > self.min_mvaid)
        & (photons.hoe < self.max_hovere)
        & (
            (photons.r9 > self.min_full5x5_r9)
            | (rel_iso * photons.pt < self.max_chad_iso)
            | (rel_iso < self.max_chad_rel_iso)
        )
        & (isEB_high_r9 | isEB_low_r9 | isEE_high_r9)
    ]
