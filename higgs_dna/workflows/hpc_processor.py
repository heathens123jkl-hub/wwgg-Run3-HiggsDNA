from higgs_dna.workflows.skeleton import HggSkeletonProcessor
from higgs_dna.tools.chained_quantile import ChainedQuantileRegression
from higgs_dna.tools.diphoton_mva import calculate_multiclass_diphoton_mva
from higgs_dna.tools.hpc_mva import calculate_ggh_vs_hb_mva, fill_ggh_vs_hb_mva_vars
from higgs_dna.tools.xgb_loader import load_bdt
from higgs_dna.tools.photonid_mva import load_photonid_mva
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EELeak_region import veto_EEleak_flag
from higgs_dna.tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from higgs_dna.tools.jetID import add_jetId
from higgs_dna.tools.sigma_m_tools import compute_sigma_m
from higgs_dna.selections.photon_selections import photon_preselection
from higgs_dna.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from higgs_dna.selections.lepton_selections import select_electrons, select_muons
from higgs_dna.selections.jet_selections import select_jets, jetvetomap
from higgs_dna.selections.sv_selections import match_sv
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
)
from higgs_dna.utils.misc_utils import choose_jet, add_pnet_prob
from higgs_dna.tools.flow_corrections import apply_flow_corrections_to_photons

from higgs_dna.tools.mass_decorrelator import decorrelate_mass_resolution

# from higgs_dna.utils.dumping_utils import diphoton_list_to_pandas, dump_pandas
from higgs_dna.metaconditions import photon_id_mva_weights
from higgs_dna.metaconditions import diphoton as diphoton_mva_dir
from higgs_dna.metaconditions import hpc_bdt as hpc_mva_dir
from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections
from higgs_dna.systematics import apply_systematic_variations_object_level

import os
import warnings
from typing import Any, Dict, List, Optional
import awkward as ak
import numpy
import sys
import vector
from coffea.analysis_tools import Weights
import copy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class HplusCharmProcessor(HggSkeletonProcessor):  # type: ignore
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Optional[Dict[str, List[str]]] = None,
        corrections: Optional[Dict[str, List[str]]] = None,
        apply_trigger: bool = False,
        nano_version: int = 13,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        bTagEffFileName: Optional[str] = None,
        trigger_group: str = ".*DoubleEG.*",
        analysis: str = "mainAnalysis",
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "simple",
        doDeco: bool = False,
        Smear_sigma_m: bool = False,
        doFlow_corrections: bool = False,
        validate_with_electrons: bool = False,
        output_format: str = "parquet",
    ) -> None:
        super().__init__(
            metaconditions,
            systematics=systematics,
            corrections=corrections,
            apply_trigger=apply_trigger,
            nano_version=nano_version,
            bTagEffFileName=bTagEffFileName,
            output_location=output_location,
            taggers=taggers,
            trigger_group=trigger_group,
            analysis=analysis,
            applyCQR=applyCQR,
            skipJetVetoMap=skipJetVetoMap,
            year=year,
            fiducialCuts=fiducialCuts,
            doDeco=doDeco,
            Smear_sigma_m=Smear_sigma_m,
            doFlow_corrections=doFlow_corrections,
            validate_with_electrons=validate_with_electrons,
            output_format=output_format
        )

        # muon selection cuts
        self.muon_pt_threshold = 10
        self.muon_max_eta = 2.4
        self.mu_id_wp = "medium"
        self.global_muon = False

        # electron selection cuts
        self.electron_pt_threshold = 15
        self.electron_max_eta = 2.5
        self.el_id_wp = "WP80"

        # jet selection cuts
        self.jet_jetId = "tightLepVeto"  # can be "tightLepVeto" or "tight": https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
        self.jet_dipho_min_dr = 0.4
        self.jet_pho_min_dr = 0.4
        self.jet_ele_min_dr = 0.4
        self.jet_muo_min_dr = 0.4
        self.jet_pt_threshold = 20
        self.jet_max_eta = 2.5

        self.clean_jet_dipho = True
        self.clean_jet_pho = True
        self.clean_jet_ele = False
        self.clean_jet_muo = False

        # diphoton preselection cuts
        self.min_pt_photon = 25.0
        self.min_pt_lead_photon = 35.0
        self.min_pt_mgg_lead_photon = 1. / 3.
        self.min_pt_mgg_sublead_photon = 0.25
        self.min_mass_range = 100
        self.max_mass_range = 180
        self.min_mvaid = -0.7
        self.max_sc_eta = 2.5
        self.gap_barrel_eta = 1.4442
        self.gap_endcap_eta = 1.566
        self.max_hovere = 0.08
        self.min_full5x5_r9 = 0.8
        self.max_chad_iso = 20.0
        self.max_chad_rel_iso = 0.3

        self.min_full5x5_r9_EB_high_r9 = 0.85
        self.min_full5x5_r9_EE_high_r9 = 0.9
        self.min_full5x5_r9_EB_low_r9 = 0.5
        self.min_full5x5_r9_EE_low_r9 = 0.8
        self.max_trkSumPtHollowConeDR03_EB_low_r9 = 6.0  # for v11, we cut on Photon_pfChargedIsoPFPV
        self.max_trkSumPtHollowConeDR03_EE_low_r9 = 6.0  # Leaving the names of the preselection cut variables the same to change as little as possible
        self.max_sieie_EB_low_r9 = 0.015
        self.max_sieie_EE_low_r9 = 0.035
        self.max_pho_iso_EB_low_r9 = 4.0
        self.max_pho_iso_EE_low_r9 = 4.0

        self.eta_rho_corr = 1.5
        self.low_eta_rho_corr = 0.16544
        self.high_eta_rho_corr = 0.13212
        # EA values for Run3 from Egamma
        self.EA1_EB1 = 0.102056
        self.EA2_EB1 = -0.000398112
        self.EA1_EB2 = 0.0820317
        self.EA2_EB2 = -0.000286224
        self.EA1_EE1 = 0.0564915
        self.EA2_EE1 = -0.000248591
        self.EA1_EE2 = 0.0428606
        self.EA2_EE2 = -0.000171541
        self.EA1_EE3 = 0.0395282
        self.EA2_EE3 = -0.000121398
        self.EA1_EE4 = 0.0369761
        self.EA2_EE4 = -8.10369e-05
        self.EA1_EE5 = 0.0369417
        self.EA2_EE5 = -2.76885e-05

        logger.debug(f"Setting up processor with metaconditions: {self.meta}")
        logger.debug(f"self.meta = {self.meta}")
        logger.debug(f"self.year = {self.year}")
        logger.debug(f"self.corrections = {self.corrections}")
        logger.debug(f"self.systematics = {self.systematics}")
        logger.debug(f"self.output_location = {self.output_location}")
        logger.debug(f"self.trigger_group = {self.trigger_group}")

        self.taggers = []
        if taggers is not None:
            self.taggers = taggers
            self.taggers.sort(key=lambda x: x.priority)

        self.prefixes = {"pho_lead": "lead", "pho_sublead": "sublead"}

        if not self.doDeco:
            logger.info("Skipping Mass resolution decorrelation as required")
        else:
            logger.info("Performing Mass resolution decorrelation as required")

        # build the chained quantile regressions
        if self.applyCQR:
            try:
                self.chained_quantile: Optional[
                    ChainedQuantileRegression
                ] = ChainedQuantileRegression(**self.meta["PhoIdInputCorrections"])
            except Exception as e:
                warnings.warn(f"Could not instantiate ChainedQuantileRegression: {e}")
                self.chained_quantile = None
        else:
            logger.info("Skipping CQR as required")
            self.chained_quantile = None

        # initialize photonid_mva
        photon_id_mva_dir = os.path.dirname(photon_id_mva_weights.__file__)
        try:
            logger.debug(
                f"Looking for {self.meta['flashggPhotons']['photonIdMVAweightfile_EB']} in {photon_id_mva_dir}"
            )
            self.photonid_mva_EB = load_photonid_mva(
                os.path.join(
                    photon_id_mva_dir,
                    self.meta["flashggPhotons"]["photonIdMVAweightfile_EB"],
                )
            )
            self.photonid_mva_EE = load_photonid_mva(
                os.path.join(
                    photon_id_mva_dir,
                    self.meta["flashggPhotons"]["photonIdMVAweightfile_EE"],
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate PhotonID MVA on the fly: {e}")
            self.photonid_mva_EB = None
            self.photonid_mva_EE = None

        # initialize diphoton mva
        diphoton_weights_dir = os.path.dirname(diphoton_mva_dir.__file__)
        logger.debug(
            f"Base path to look for IDMVA weight files: {diphoton_weights_dir}"
        )

        # this is needed because the new training has a k-fold approach, we have two models, one for
        # even event numbers and one for odd event numbers
        self.diphoton_mva = [None, None]
        try:
            self.diphoton_mva[0] = load_bdt(
                os.path.join(
                    diphoton_weights_dir, self.meta["HiggsDNA_DiPhotonMVA"]["weightFile"][0]
                )
            )
            self.diphoton_mva[1] = load_bdt(
                os.path.join(
                    diphoton_weights_dir, self.meta["HiggsDNA_DiPhotonMVA"]["weightFile"][1]
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate diphoton MVA: {e}")
            self.diphoton_mva = None

        self.multiclass_diphoton_mva = [None, None]
        try:
            self.multiclass_diphoton_mva[0] = load_bdt(
                os.path.join(
                    diphoton_weights_dir, self.meta["HiggsDNA_DiPhotonMVA_multiclass"]["weightFile"][0]
                )
            )
            self.multiclass_diphoton_mva[1] = load_bdt(
                os.path.join(
                    diphoton_weights_dir, self.meta["HiggsDNA_DiPhotonMVA_multiclass"]["weightFile"][1]
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate multiclass diphoton MVA: {e}")
            self.multiclass_diphoton_mva = None

        # initialize ch vs ggh mva, only useful for run 2 cH yukawa analysis
        hpc_weight_dir = os.path.dirname(hpc_mva_dir.__file__)
        logger.debug(
            f"Base path to look for cH vs ggH MVA weight files: {hpc_weight_dir}"
        )
        self.ch_vs_ggh_mva = [None, None]
        try:
            self.ch_vs_ggh_mva[0] = load_bdt(
                os.path.join(
                    hpc_weight_dir, self.meta["hpcMVA_ch_vs_ggh"]["weightFile"][0]
                )
            )
            self.ch_vs_ggh_mva[1] = load_bdt(
                os.path.join(
                    hpc_weight_dir, self.meta["hpcMVA_ch_vs_ggh"]["weightFile"][1]
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate hpc MVA ch vs ggh: {e}")
            self.ch_vs_ggh_mva = None

        self.ch_vs_cb_mva = [None, None]
        try:
            self.ch_vs_cb_mva[0] = load_bdt(
                os.path.join(
                    hpc_weight_dir, self.meta["hpcMVA_ch_vs_cb"]["weightFile"][0]
                )
            )
            self.ch_vs_cb_mva[1] = load_bdt(
                os.path.join(
                    hpc_weight_dir, self.meta["hpcMVA_ch_vs_cb"]["weightFile"][1]
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate hpc MVA ch vs cb: {e}")
            self.ch_vs_cb_mva = None

        # initialize ggh vs hb mva
        self.ggh_vs_hb_mva = [None, None]
        try:
            self.ggh_vs_hb_mva[0] = load_bdt(
                os.path.join(
                    hpc_weight_dir, self.meta["hpcMVA_ggh_vs_hb"]["weightFile"][0]
                )
            )
            self.ggh_vs_hb_mva[1] = load_bdt(
                os.path.join(
                    hpc_weight_dir, self.meta["hpcMVA_ggh_vs_hb"]["weightFile"][1]
                )
            )
        except Exception as e:
            warnings.warn(f"Could not instantiate hpc MVA ggh vs hb: {e}")
            self.ggh_vs_hb_mva = None

        if self.validate_with_electrons:
            logger.info("Running the analysis with electrons reconstructed as photons. Using dielectron triggers.")
            self.trigger_group = ".*DoubleEG.*"
            self.analysis = "Dielectron"
            self.min_pt_photon = 12.0
            self.min_pt_lead_photon = 23.0

    def process_extra(self, events: ak.Array) -> ak.Array:
        raise NotImplementedError

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]
        self.dataset_name = dataset_name

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        # here we start recording possible coffea accumulators
        # most likely histograms, could be counters, arrays, ...
        histos_etc = {}
        histos_etc[dataset_name] = {}
        if self.data_kind == "mc":
            histos_etc[dataset_name]["nTot"] = int(
                ak.num(events.genWeight, axis=0)
            )
            histos_etc[dataset_name]["nPos"] = int(ak.sum(events.genWeight > 0))
            histos_etc[dataset_name]["nNeg"] = int(ak.sum(events.genWeight < 0))
            histos_etc[dataset_name]["nEff"] = int(
                histos_etc[dataset_name]["nPos"] - histos_etc[dataset_name]["nNeg"]
            )
            histos_etc[dataset_name]["genWeightSum"] = float(
                numpy.sum(events.genWeight.to_numpy())
            )
        else:
            histos_etc[dataset_name]["nTot"] = int(len(events))
            histos_etc[dataset_name]["nPos"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["nNeg"] = int(0)
            histos_etc[dataset_name]["nEff"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["genWeightSum"] = float(len(events))

        # lumi mask
        if self.data_kind == "data":
            try:
                lumimask = select_lumis(self.year[dataset_name][0], events, logger)
                events = events[lumimask]
            except:
                logger.info(
                    f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}"
                )
        # apply jetvetomap
        if not self.skipJetVetoMap:
            events = jetvetomap(
                self, events, logger, dataset_name, year=self.year[dataset_name][0]
            )
        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            logger.info("processing MC dataset")
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(numpy.sum(events.genWeight.to_numpy()))
            genJets = get_genJets(events, pt_cut=10., eta_cut=2.5)
            had_gen_jets = genJets[ak.argsort(abs(genJets.hadronFlavour), ascending=False, axis=1)]
            first_had_gen_jet_hadFlav = choose_jet(had_gen_jets.hadronFlavour, 0, -999.0)

        else:
            logger.info("processing Data dataset")
            metadata["sum_genw_presel"] = "Data"

        # Store the command line used
        metadata["command_line"] = " ".join(sys.argv)

        # apply filters and triggers
        events = self.apply_filters_and_triggers(events)

        # add zero photon mass and charge
        # TODO: remove this temporary fix when https://github.com/scikit-hep/vector/issues/498 is resolved
        events["Photon"] = self.add_zero_photon_mass_and_charge(events.Photon)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
        events["Photon"] = add_photon_SC_eta(events.Photon, events.PV)

        if self.validate_with_electrons:
            # select photons with an associated electron and a pixel seed
            photons_mask = (events.Photon.electronIdx != -1) & (events.Photon.pixelSeed)
            events["Photon"] = events.Photon[photons_mask]
            events = events[ak.num(events.Photon) >= 2]

        # add veto EE leak branch for photons, could also be used for electrons
        if (
            self.year[dataset_name][0] == "2022EE"
            or self.year[dataset_name][0] == "2022postEE"
        ):
            events["Photon"] = veto_EEleak_flag(self, events.Photon)

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
        except KeyError:
            systematic_names = []

        # If --Smear-sigma_m == True and no Smearing correction in .json for MC throws an error, since the pt scpectrum need to be smeared in order to properly calculate the smeared sigma_m_m
        if (
            self.data_kind == "mc"
            and self.Smear_sigma_m
            and ("Smearing_Trad" not in correction_names and "Smearing_IJazZ" not in correction_names and "Smearing2G_IJazZ" not in correction_names)
        ):
            warnings.warn(
                "Smearing_Trad or Smearing_IJazZ or Smearing2G_IJazZ should be specified in the corrections field in .json in order to smear the mass!"
            )
            sys.exit(0)

        # save raw pt if we use scale/smearing corrections
        # These needs to be before the smearing of the mass resolution in order to have the raw pt for the function
        s_or_s_applied = False
        s_or_s_ele_applied = False
        for correction in correction_names:
            if "scale" or "smearing" in correction.lower():
                if "Electron" in correction:
                    s_or_s_ele_applied = True
                else:
                    s_or_s_applied = True
        if s_or_s_applied:
            events["Photon"] = ak.with_field(events.Photon, events.Photon.pt, "pt_raw")
        if s_or_s_ele_applied:
            events["Electron"] = ak.with_field(events.Electron, events.Electron.pt, "pt_raw")

        # Since now we are applying Smearing term to the sigma_m_over_m i added this portion of code
        # specially for the estimation of smearing terms for the data events [data pt/energy] are not smeared!
        if self.data_kind == "data" and self.Smear_sigma_m:
            if "Scale_Trad" in correction_names:
                correction_name = "Smearing_Trad"
            elif "Scale_IJazZ" in correction_names:
                correction_name = "Smearing_IJazZ"
            elif "Scale2G_IJazZ" in correction_names:
                correction_name = "Smearing2G_IJazZ"
            else:
                logger.info('Specify a scale correction for the data in the corrections field in .json in order to smear the mass!')
                sys.exit(0)

        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"Applying correction {correction_name} to dataset {dataset_name}"
                )
                varying_function = available_object_corrections[correction_name]
                events = varying_function(
                    events=events, year=self.year[dataset_name][0]
                )
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        # apply jetvetomap: only retain events that without any jets in the veto region
        if not self.skipJetVetoMap:
            events = jetvetomap(
                self, events, logger, dataset_name, year=self.year[dataset_name][0]
            )

        original_photons = events.Photon
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet

        # Computing the normalizing flow correction
        if self.data_kind == "mc" and self.doFlow_corrections:
            flows_photonid_mva = self.resolve_flows_photonid_mva(events)
            original_photons = apply_flow_corrections_to_photons(
                original_photons,
                events,
                self.meta,
                self.year[dataset_name][0],
                flows_photonid_mva,
                logger
            )

        # Add additional collections if object systematics should be applied
        collections = {
            "Photon": original_photons,
        }

        # Apply the systematic variations.
        collections = apply_systematic_variations_object_level(
            systematic_names,
            events,
            self.year[dataset_name][0],
            logger,
            available_object_systematics,
            available_weight_systematics,
            collections
        )

        original_photons = collections["Photon"]

        # Write systematic variations to dicts
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                photons_dct[f"{systematic}_{variation}"] = original_photons.systematics[systematic][variation]

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        # object systematics dictionary
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
        # NOTE: jet jerc systematics are not added with add_systematics
        variations_combined.append(jerc_syst_list)
        # Flatten
        variations_flattened = sum(variations_combined, [])  # Begin with empty list and keep concatenating
        # Attach _down and _up
        variations = [item + suffix for item in variations_flattened for suffix in ['_down', '_up']]
        # Add nominal to the list
        variations.append('nominal')
        logger.debug(f"[systematics variations] {variations}")

        for variation in variations:
            photons, jets = photons_dct["nominal"], events.Jet
            if variation == "nominal":
                pass  # Do nothing since we already get the unvaried, but nominally corrected objets above
            elif variation in [*photons_dct]:  # [*dict] gets the keys of the dict since Python >= 3.5
                photons = photons_dct[variation]
            elif variation in [*jets_dct]:
                jets = jets_dct[variation]
            do_variation = variation  # We can also simplify this a bit but for now it works

            if self.chained_quantile is not None and self.data_kind == "mc":
                logger.warning("Applying CQR")
                photons = self.chained_quantile.apply(photons, events)

            # recompute photonid_mva on the fly
            # NOTE: this does not work properly at the moment, need to fix and in the meantime we use the EGamma ID
            if (self.photonid_mva_EB and self.photonid_mva_EE and self.applyCQR) or (self.photonid_mva_EB and self.photonid_mva_EE and self.data_kind == "data"):
                if self.year[dataset_name][0] in ["2017"]:
                    logger.warning("Applying photon ID MVA")
                    photons = self.add_photonid_mva(photons, events)
                elif self.year[dataset_name][0] in ["2016", "2016pre", "2016post", "2018"]:
                    logger.warning("Missing input variables correction with either CQR or NF, we will NOT re-evaluate photon ID MVA")

            # photon preselection
            if self.validate_with_electrons:
                photons = photon_preselection(self, photons, events, year=self.year[dataset_name][0], electron_veto=False, revert_electron_veto=True)
            else:
                photons = photon_preselection(self, photons, events, year=self.year[dataset_name][0])

            diphotons = build_diphoton_candidates(photons, self.min_pt_lead_photon)

            # Apply the fiducial cut at detector level with helper function
            diphotons = apply_fiducial_cut_det_level(self, diphotons)

            # preselection, require at least one candidate
            dipho_presel_cut = ak.num(diphotons["mass"]) >= 1
            dipho_num = ak.num(diphotons["mass"])

            if self.data_kind == "mc":

                # Add the fiducial flags for particle level
                diphotons['fiducialClassicalFlag'] = get_fiducial_flag(events, flavour='Classical')
                diphotons['fiducialGeometricFlag'] = get_fiducial_flag(events, flavour='Geometric')

                GenPTH, GenYH, GenPhiH, _, _ = get_higgs_gen_attributes(events)
                GenPTH = ak.fill_none(GenPTH, -999.0)
                diphotons['GenPTH'] = GenPTH

                genJets = get_genJets(
                    events,
                    pt_cut=20.,
                    eta_cut=2.5,
                    jet_pho_min_dr=self.jet_pho_min_dr,
                    jet_ele_min_dr=self.jet_ele_min_dr,
                    jet_muo_min_dr=self.jet_muo_min_dr,
                    electron_pt_threshold=self.electron_pt_threshold,
                    electron_max_eta=self.electron_max_eta,
                    muon_pt_threshold=self.muon_pt_threshold,
                    muon_max_eta=self.muon_max_eta,
                )
                diphotons['GenNJ'] = ak.num(genJets)
                GenPTJ0 = choose_jet(genJets.pt, 0, -999.0)  # Choose zero (leading) jet and pad with -999 if none
                diphotons['GenPTJ0'] = GenPTJ0

            diphotons = ak.firsts(diphotons)
            diphotons["n_dipho_cand"] = dipho_num

            # select only events with a candidate
            diphotons = diphotons[dipho_presel_cut]
            dipho_events = events[dipho_presel_cut]
            jets = jets[dipho_presel_cut]
            if self.data_kind == "mc":
                genJets = genJets[dipho_presel_cut]

            # baseline modifications to diphotons
            if self.diphoton_mva is not None:
                logger.warning("Applying diphoton MVA")
                diphotons, dipho_events = self.add_diphoton_mva(diphotons, dipho_events)

            if self.multiclass_diphoton_mva is not None:
                logger.warning("Applying multiclass diphoton MVA")
                diphotons, dipho_events = self.add_multiclass_diphoton_mva(diphotons, dipho_events)

            logger.info(f"selected number of events  after diphoton preselection {len(dipho_events)}")

            # NOTE: all this circus is needed because the PNet score name changed from one nanoAOD version to the next
            # to still catch all the possible samples I have I need to do this ugly switches
            if hasattr(jets, "particleNetAK4_B"):
                # jet_variables
                jets = ak.zip(
                    {
                        "pt": jets.pt,
                        "eta": jets.eta,
                        "phi": jets.phi,
                        "mass": jets.mass,
                        "charge": jets.chEmEF,
                        "hFlav": jets.hadronFlavour if self.data_kind == "mc" else ak.ones_like(jets.pt) * -1.,
                        "particleNetAK4_CvsL": jets.particleNetAK4_CvsL,
                        "particleNetAK4_CvsB": jets.particleNetAK4_CvsB,
                        "particleNetAK4_B": jets.particleNetAK4_B,
                        "particleNetAK4_QvsG": jets.particleNetAK4_QvsG,
                        "btagDeepFlav_B": jets.btagDeepFlavB,
                        "btagDeepFlav_CvB": jets.btagDeepFlavCvB,
                        "btagDeepFlav_CvL": jets.btagDeepFlavCvL,
                        "btagDeepFlav_QG": jets.btagDeepFlavQG,
                        "jetId": add_jetId(jets, self.nano_version, self.year[dataset_name][0], flattenUnflatten=True),  # add jet ID based on nano version
                        "n_sv": jets.nSVs if hasattr(jets, "nSVs") else ak.ones_like(jets.pt) * -1.,
                        "n_muons": jets.nMuons if hasattr(jets, "nMuons") else ak.ones_like(jets.pt) * -1.,
                        "n_electrons": jets.nElectrons if hasattr(jets, "nElectrons") else ak.ones_like(jets.pt) * -1.,
                        "nConst": jets.nConstituents if hasattr(jets, "nConstituents") else ak.ones_like(jets.pt) * -1.,
                        "neHEF": jets.neHEF if hasattr(jets, "neHEF") else ak.ones_like(jets.pt) * -1.,
                        "neEmEF": jets.neEmEF if hasattr(jets, "neEmEF") else ak.ones_like(jets.pt) * -1.,
                        "chHEF": jets.chHEF if hasattr(jets, "chHEF") else ak.ones_like(jets.pt) * -1.,
                        "chEmEF": jets.chEmEF if hasattr(jets, "chEmEF") else ak.ones_like(jets.pt) * -1.,
                        "muEF": jets.muEF if hasattr(jets, "muEF") else ak.ones_like(jets.pt) * -1.,
                        "chMultiplicity": jets.chMultiplicity if hasattr(jets, "chMultiplicity") else ak.ones_like(jets.pt) * -1.,
                        "neMultiplicity": jets.neMultiplicity if hasattr(jets, "neMultiplicity") else ak.ones_like(jets.pt) * -1.,
                    }
                )
            else:
                # jet_variables
                jets = ak.zip(
                    {
                        "pt": jets.pt,
                        "eta": jets.eta,
                        "phi": jets.phi,
                        "mass": jets.mass,
                        "charge": jets.chEmEF,
                        "hFlav": jets.hadronFlavour if self.data_kind == "mc" else ak.ones_like(jets.pt) * -1.,
                        "particleNetAK4_CvsL": jets.btagPNetCvL if hasattr(jets, "btagPNetCvL") else ak.ones_like(jets.pt) * -1.,
                        "particleNetAK4_CvsB": jets.btagPNetCvB if hasattr(jets, "btagPNetCvB") else ak.ones_like(jets.pt) * -1.,
                        "particleNetAK4_B": jets.btagPNetB if hasattr(jets, "btagPNetB") else ak.ones_like(jets.pt) * -1.,
                        "particleNetAK4_QvsG": jets.btagPNetQvG if hasattr(jets, "btagPNetQvG") else ak.ones_like(jets.pt) * -1.,
                        "btagDeepFlav_B": jets.btagDeepFlavB,
                        "btagDeepFlav_CvB": jets.btagDeepFlavCvB,
                        "btagDeepFlav_CvL": jets.btagDeepFlavCvL,
                        "btagDeepFlav_QG": jets.btagDeepFlavQG,
                        **(
                            {"jetId": jets.jetId} if self.nano_version < 15 else {}
                        ),
                        "n_sv": jets.nSVs if hasattr(jets, "nSVs") else ak.ones_like(jets.pt) * -1.,
                        "n_muons": jets.nMuons if hasattr(jets, "nMuons") else ak.ones_like(jets.pt) * -1.,
                        "n_electrons": jets.nElectrons if hasattr(jets, "nElectrons") else ak.ones_like(jets.pt) * -1.,
                        "nConst": jets.nConstituents if hasattr(jets, "nConstituents") else ak.ones_like(jets.pt) * -1.,
                        "neHEF": jets.neHEF if hasattr(jets, "neHEF") else ak.ones_like(jets.pt) * -1.,
                        "neEmEF": jets.neEmEF if hasattr(jets, "neEmEF") else ak.ones_like(jets.pt) * -1.,
                        "chHEF": jets.chHEF if hasattr(jets, "chHEF") else ak.ones_like(jets.pt) * -1.,
                        "chEmEF": jets.chEmEF if hasattr(jets, "chEmEF") else ak.ones_like(jets.pt) * -1.,
                        "muEF": jets.muEF if hasattr(jets, "muEF") else ak.ones_like(jets.pt) * -1.,
                        "chMultiplicity": jets.chMultiplicity if hasattr(jets, "chMultiplicity") else ak.ones_like(jets.pt) * -1.,
                        "neMultiplicity": jets.neMultiplicity if hasattr(jets, "neMultiplicity") else ak.ones_like(jets.pt) * -1.,
                    }
                )
            jets = ak.with_name(jets, "PtEtaPhiMCandidate")
            if not hasattr(jets, "particleNetAK4_CvsL"):
                logger.info("your jet collection doesn't have the PNet score fields")
            else:
                logger.info(f"your jet collection has the PNet score fields {jets.particleNetAK4_CvsL}")
                jets = add_pnet_prob(self, jets)

            electrons = ak.zip(
                {
                    "pt": dipho_events.Electron.pt,
                    "eta": dipho_events.Electron.eta,
                    "phi": dipho_events.Electron.phi,
                    "mass": dipho_events.Electron.mass,
                    "charge": dipho_events.Electron.charge,
                    "mvaIso_WP90": dipho_events.Electron.mvaIso_Fall17V2_WP90 if hasattr(dipho_events.Electron, "mvaIso_Fall17V2_WP90") else dipho_events.Electron.mvaIso_WP90,
                    "mvaIso_WP80": dipho_events.Electron.mvaIso_Fall17V2_WP80 if hasattr(dipho_events.Electron, "mvaIso_Fall17V2_WP80") else dipho_events.Electron.mvaIso_WP80,
                    "mvaIso_WPL": dipho_events.Electron.mvaIso_Fall17V2_WPL if hasattr(dipho_events.Electron, "mvaIso_Fall17V2_WPL") else dipho_events.Electron.mvaIso_WP80,
                }
            )
            electrons = ak.with_name(electrons, "PtEtaPhiMCandidate")

            muons = ak.zip(
                {
                    "pt": dipho_events.Muon.pt,
                    "eta": dipho_events.Muon.eta,
                    "phi": dipho_events.Muon.phi,
                    "mass": dipho_events.Muon.mass,
                    "charge": dipho_events.Muon.charge,
                    "tightId": dipho_events.Muon.tightId,
                    "mediumId": dipho_events.Muon.mediumId,
                    "looseId": dipho_events.Muon.looseId,
                    "isGlobal": dipho_events.Muon.isGlobal,
                    "pfIsoId": dipho_events.Muon.pfIsoId
                }
            )
            muons = ak.with_name(muons, "PtEtaPhiMCandidate")

            sv = ak.zip(
                {
                    "pt": dipho_events.SV.pt,
                    "eta": dipho_events.SV.eta,
                    "phi": dipho_events.SV.phi,
                    "mass": dipho_events.SV.mass,
                    "charge": dipho_events.SV.charge,
                    "dlen": dipho_events.SV.dlen,
                    "dlenSig": dipho_events.SV.dlenSig,
                    "dxy": dipho_events.SV.dxy,
                    "dxySig": dipho_events.SV.dxySig,
                    "pAngle": dipho_events.SV.pAngle,
                    "chi2": dipho_events.SV.chi2,
                    "x": dipho_events.SV.x,
                    "y": dipho_events.SV.y,
                    "z": dipho_events.SV.z,
                    "ndof": dipho_events.SV.ndof,
                    "ntracks": dipho_events.SV.ntracks,
                }
            )
            sv = ak.with_name(sv, "PtEtaPhiMCandidate")

            if self.data_kind == "mc":
                particles = ak.zip(
                    {
                        "pt": dipho_events.GenPart.pt,
                        "eta": dipho_events.GenPart.eta,
                        "phi": dipho_events.GenPart.phi,
                        "mass": dipho_events.GenPart.mass,
                        "pdgId": dipho_events.GenPart.pdgId,
                        "status": dipho_events.GenPart.status,
                        "statusFlags": dipho_events.GenPart.statusFlags,
                        "IdxMother": dipho_events.GenPart.genPartIdxMother,
                        "event": ak.ones_like(dipho_events.GenPart.pt) * dipho_events.event,
                    }
                )
                particles = ak.with_name(particles, "PtEtaPhiMCandidate")

                # here we create a flag to identify the presence of c and b hadrons in the event
                # we have to massage a bit the arrays due to some issues with old versions of ak that make complicated
                # to extract the first digit of the pdgId
                counts = ak.num(particles.pdgId)
                pdgId = ak.to_numpy(abs(ak.flatten(particles.pdgId)))
                status = ak.to_numpy(abs(ak.flatten(particles.status)))
                pt = ak.to_numpy(abs(ak.flatten(particles.pt)))
                eta = ak.to_numpy(abs(ak.flatten(particles.eta)))
                str_pdgId = numpy.array(pdgId, str)

                # masks for the different hadron types
                ME_out_c_mask = numpy.char.startswith(str_pdgId, "4") & (status == 23)
                ME_out_b_mask = numpy.char.startswith(str_pdgId, "5") & (status == 23)
                ME_out_g_mask = (str_pdgId == "21") & (status == 23)
                ME_out_l_mask = ((str_pdgId == "1") | (str_pdgId == "2") | (str_pdgId == "3")) & (status == 23)

                ME_in_c_mask = numpy.char.startswith(str_pdgId, "4") & (status == 21)
                ME_in_b_mask = numpy.char.startswith(str_pdgId, "5") & (status == 21)
                ME_in_g_mask = (str_pdgId == "21") & (status == 21)
                ME_in_l_mask = ((str_pdgId == "1") | (str_pdgId == "2") | (str_pdgId == "3")) & (status == 21)

                c_mask = numpy.char.startswith(str_pdgId, "4") & (status <= 2) & (pt > 5) & (abs(eta) < 4)
                b_mask = numpy.char.startswith(str_pdgId, "5") & (status <= 2) & (pt > 5) & (abs(eta) < 4)

                # reformat the masks
                str_pdgId = ak.unflatten(str_pdgId, counts)

                c_mask = ak.unflatten(c_mask, counts)
                b_mask = ak.unflatten(b_mask, counts)

                ME_out_c_mask = ak.unflatten(ME_out_c_mask, counts)
                ME_out_b_mask = ak.unflatten(ME_out_b_mask, counts)
                ME_out_g_mask = ak.unflatten(ME_out_g_mask, counts)
                ME_out_l_mask = ak.unflatten(ME_out_l_mask, counts)

                ME_in_c_mask = ak.unflatten(ME_in_c_mask, counts)
                ME_in_b_mask = ak.unflatten(ME_in_b_mask, counts)
                ME_in_g_mask = ak.unflatten(ME_in_g_mask, counts)
                ME_in_l_mask = ak.unflatten(ME_in_l_mask, counts)

                D_hadrons = particles[c_mask]
                B_hadrons = particles[b_mask]

                charmed_had_num = ak.num(D_hadrons)
                bottom_had_num = ak.num(B_hadrons)

                ME_in_c = particles[ME_in_c_mask]
                ME_in_b = particles[ME_in_b_mask]
                ME_in_l = particles[ME_in_l_mask]
                ME_in_g = particles[ME_in_g_mask]

                ME_out_c = particles[ME_out_c_mask]
                ME_out_b = particles[ME_out_b_mask]
                ME_out_l = particles[ME_out_l_mask]
                ME_out_g = particles[ME_out_g_mask]

                ME_in_charm_num = ak.num(ME_in_c)
                ME_in_bottom_num = ak.num(ME_in_b)
                ME_in_light_num = ak.num(ME_in_l)
                ME_in_gluon_num = ak.num(ME_in_g)

                ME_out_charm_num = ak.num(ME_out_c)
                ME_out_bottom_num = ak.num(ME_out_b)
                ME_out_light_num = ak.num(ME_out_l)
                ME_out_gluon_num = ak.num(ME_out_g)

                gen_jets = ak.zip(
                    {
                        "pt": genJets.pt,
                        "eta": genJets.eta,
                        "phi": genJets.phi,
                        "mass": genJets.mass,
                        "hFlav": genJets.hadronFlavour,
                        "event": ak.ones_like(genJets.pt) * dipho_events.event,
                    }
                )
            else:
                charmed_had_num = ak.zeros_like(diphotons.pt)
                bottom_had_num = ak.zeros_like(diphotons.pt)

                ME_in_c = ak.zeros_like(diphotons.pt)
                ME_in_b = ak.zeros_like(diphotons.pt)
                ME_in_l = ak.zeros_like(diphotons.pt)
                ME_in_g = ak.zeros_like(diphotons.pt)

                ME_out_c = ak.zeros_like(diphotons.pt)
                ME_out_b = ak.zeros_like(diphotons.pt)
                ME_out_l = ak.zeros_like(diphotons.pt)
                ME_out_g = ak.zeros_like(diphotons.pt)

                ME_in_charm_num = ak.zeros_like(diphotons.pt)
                ME_in_bottom_num = ak.zeros_like(diphotons.pt)
                ME_in_light_num = ak.zeros_like(diphotons.pt)
                ME_in_gluon_num = ak.zeros_like(diphotons.pt)

                ME_out_charm_num = ak.zeros_like(diphotons.pt)
                ME_out_bottom_num = ak.zeros_like(diphotons.pt)
                ME_out_light_num = ak.zeros_like(diphotons.pt)
                ME_out_gluon_num = ak.zeros_like(diphotons.pt)

            # lepton cleaning
            sel_electrons = electrons[
                select_electrons(self, electrons, diphotons)
            ]
            sel_muons = muons[select_muons(self, muons, diphotons)]

            # jet selection and pt ordering
            jets = jets[
                select_jets(self, jets, diphotons, sel_muons, sel_electrons)
            ]

            # sort diphotons by pT
            pt_jets = jets[ak.argsort(jets.pt, ascending=False, axis=1)]
            # sort diphotons by CvsL
            # jets = jets[ak.argsort(jets.btagDeepFlav_CvL, ascending=False, axis=1)]
            jets = jets[ak.argsort(jets.particleNetAK4_CvsL, ascending=False, axis=1)]

            n_jets = ak.num(jets)
            n_b_jets_medium = ak.num(jets[jets.btagDeepFlav_B > 0.3033])  # medium wp 2017
            n_b_jets_loose = ak.num(jets[jets.btagDeepFlav_B > 0.0521])  # loose wp 2017

            # now I create the dijet system
            dijets = ak.combinations(
                pt_jets, 2, fields=["j_lead", "j_sublead"]
            )

            # now turn the dijets into candidates with four momenta and such
            dijets_4mom = dijets["j_lead"] + dijets["j_sublead"]
            dijets["pt"] = dijets_4mom.pt
            dijets["eta"] = dijets_4mom.eta
            dijets["phi"] = dijets_4mom.phi
            dijets["mass"] = dijets_4mom.mass
            dijets["charge"] = dijets_4mom.charge

            dijets = ak.with_name(dijets, "PtEtaPhiMCandidate")

            # sort diphotons by pT
            dijets = dijets[
                ak.argsort(dijets.pt, ascending=False)
            ]

            # sv selection
            sv = sv[
                match_sv(self, jets=jets, sv=sv, lead_only=False)
            ]

            sv = sv[ak.argsort(sv.dlenSig, ascending=False, axis=1)]

            # select only events with at least one jet
            ge_1j_cut = n_jets > 0
            diphotons = diphotons[ge_1j_cut]
            dipho_events = dipho_events[ge_1j_cut]

            electrons = electrons[ge_1j_cut]
            muons = muons[ge_1j_cut]
            jets = jets[ge_1j_cut]

            if self.data_kind == "mc":
                genJets = genJets[ge_1j_cut]
                gen_jets = gen_jets[ge_1j_cut]

            pt_jets = pt_jets[ge_1j_cut]
            dijets = dijets[ge_1j_cut]
            sv = sv[ge_1j_cut]
            n_jets = n_jets[ge_1j_cut]
            n_b_jets_medium = n_b_jets_medium[ge_1j_cut]
            n_b_jets_loose = n_b_jets_loose[ge_1j_cut]
            charmed_had_num = charmed_had_num[ge_1j_cut]
            bottom_had_num = bottom_had_num[ge_1j_cut]
            ME_in_charm_num = ME_in_charm_num[ge_1j_cut]
            ME_in_bottom_num = ME_in_bottom_num[ge_1j_cut]
            ME_in_light_num = ME_in_light_num[ge_1j_cut]
            ME_in_gluon_num = ME_in_gluon_num[ge_1j_cut]
            ME_out_charm_num = ME_out_charm_num[ge_1j_cut]
            ME_out_bottom_num = ME_out_bottom_num[ge_1j_cut]
            ME_out_light_num = ME_out_light_num[ge_1j_cut]
            ME_out_gluon_num = ME_out_gluon_num[ge_1j_cut]

            dijet_pt = choose_jet(dijets.pt, 0, -999.0)
            dijet_eta = choose_jet(dijets.eta, 0, -999.0)
            dijet_phi = choose_jet(dijets.phi, 0, -999.0)
            dijet_mass = choose_jet(dijets.mass, 0, -999.0)

            first_jet_pt = choose_jet(jets.pt, 0, -999.0)
            first_jet_eta = choose_jet(jets.eta, 0, -999.0)
            first_jet_phi = choose_jet(jets.phi, 0, -999.0)
            first_jet_mass = choose_jet(jets.mass, 0, -999.0)
            first_jet_charge = choose_jet(jets.charge, 0, -999.0)
            first_jet_hFlav = choose_jet(jets.hFlav, 0, -999.0)
            first_jet_DeepFlavour_CvsL = choose_jet(jets.btagDeepFlav_CvL, 0, -999.0)
            first_jet_DeepFlavour_CvsB = choose_jet(jets.btagDeepFlav_CvB, 0, -999.0)
            first_jet_DeepFlavour_B = choose_jet(jets.btagDeepFlav_B, 0, -999.0)
            first_jet_DeepFlavour_QG = choose_jet(jets.btagDeepFlav_QG, 0, -999.0)

            first_pt_jet_pt = choose_jet(pt_jets.pt, 0, -999.0)
            first_pt_jet_eta = choose_jet(pt_jets.eta, 0, -999.0)
            first_pt_jet_phi = choose_jet(pt_jets.phi, 0, -999.0)
            first_pt_jet_mass = choose_jet(pt_jets.mass, 0, -999.0)
            first_pt_jet_charge = choose_jet(pt_jets.charge, 0, -999.0)
            first_pt_jet_hFlav = choose_jet(pt_jets.hFlav, 0, -999.0)
            first_pt_jet_DeepFlavour_CvsL = choose_jet(pt_jets.btagDeepFlav_CvL, 0, -999.0)
            first_pt_jet_DeepFlavour_CvsB = choose_jet(pt_jets.btagDeepFlav_CvB, 0, -999.0)
            first_pt_jet_DeepFlavour_B = choose_jet(pt_jets.btagDeepFlav_B, 0, -999.0)
            first_pt_jet_DeepFlavour_QG = choose_jet(pt_jets.btagDeepFlav_QG, 0, -999.0)

            first_jet_jet_pn_b = choose_jet(jets.pn_b, 0, -1.0)
            first_jet_jet_pn_c = choose_jet(jets.pn_c, 0, -1.0)
            first_jet_jet_pn_uds = choose_jet(jets.pn_uds, 0, -1.0)
            first_jet_jet_pn_g = choose_jet(jets.pn_g, 0, -1.0)
            first_jet_jet_pn_b_plus_c = choose_jet(jets.pn_b_plus_c, 0, -1.0)
            first_jet_jet_pn_b_vs_c = choose_jet(jets.pn_b_vs_c, 0, -1.0)
            first_jet_particleNetAK4_CvsL = choose_jet(jets.particleNetAK4_CvsL, 0, -999.0)
            first_jet_particleNetAK4_CvsB = choose_jet(jets.particleNetAK4_CvsB, 0, -999.0)
            first_jet_particleNetAK4_QvsG = choose_jet(jets.particleNetAK4_QvsG, 0, -999.0)
            first_jet_particleNetAK4_B = choose_jet(jets.particleNetAK4_B, 0, -999.0)
            first_jet_n_sv = choose_jet(jets.n_sv, 0, -999.0)
            first_jet_n_muons = choose_jet(jets.n_muons, 0, -999.0)
            first_jet_n_electrons = choose_jet(jets.n_electrons, 0, -999.0)

            second_jet_pt = choose_jet(jets.pt, 1, -999.0)
            second_jet_eta = choose_jet(jets.eta, 1, -999.0)
            second_jet_phi = choose_jet(jets.phi, 1, -999.0)
            second_jet_mass = choose_jet(jets.mass, 1, -999.0)
            second_jet_charge = choose_jet(jets.charge, 1, -999.0)
            second_jet_hFlav = choose_jet(jets.hFlav, 1, -999.0)
            second_jet_DeepFlavour_CvsL = choose_jet(jets.btagDeepFlav_CvL, 1, -999.0)
            second_jet_DeepFlavour_CvsB = choose_jet(jets.btagDeepFlav_CvB, 1, -999.0)
            second_jet_DeepFlavour_B = choose_jet(jets.btagDeepFlav_B, 1, -999.0)
            second_jet_DeepFlavour_QG = choose_jet(jets.btagDeepFlav_QG, 1, -999.0)
            second_jet_particleNetAK4_CvsL = choose_jet(jets.particleNetAK4_CvsL, 1, -999.0)
            second_jet_particleNetAK4_CvsB = choose_jet(jets.particleNetAK4_CvsB, 1, -999.0)
            second_jet_particleNetAK4_B = choose_jet(jets.particleNetAK4_B, 1, -999.0)
            second_jet_jet_pn_b_plus_c = choose_jet(jets.pn_b_plus_c, 1, -1.0)
            second_jet_jet_pn_b_vs_c = choose_jet(jets.pn_b_vs_c, 1, -1.0)
            second_jet_n_sv = choose_jet(jets.n_sv, 1, -999.0)

            second_pt_jet_pt = choose_jet(pt_jets.pt, 1, -999.0)
            second_pt_jet_eta = choose_jet(pt_jets.eta, 1, -999.0)
            second_pt_jet_phi = choose_jet(pt_jets.phi, 1, -999.0)
            second_pt_jet_mass = choose_jet(pt_jets.mass, 1, -999.0)
            second_pt_jet_charge = choose_jet(pt_jets.charge, 1, -999.0)
            second_pt_jet_hFlav = choose_jet(pt_jets.hFlav, 1, -999.0)
            second_pt_jet_DeepFlavour_CvsL = choose_jet(pt_jets.btagDeepFlav_CvL, 1, -999.0)
            second_pt_jet_DeepFlavour_CvsB = choose_jet(pt_jets.btagDeepFlav_CvB, 1, -999.0)
            second_pt_jet_DeepFlavour_B = choose_jet(pt_jets.btagDeepFlav_B, 1, -999.0)
            second_pt_jet_DeepFlavour_QG = choose_jet(pt_jets.btagDeepFlav_QG, 1, -999.0)

            third_jet_pt = choose_jet(jets.pt, 2, -999.0)
            third_jet_eta = choose_jet(jets.eta, 2, -999.0)
            third_jet_phi = choose_jet(jets.phi, 2, -999.0)
            third_jet_mass = choose_jet(jets.mass, 2, -999.0)
            third_jet_charge = choose_jet(jets.charge, 2, -999.0)
            third_jet_hFlav = choose_jet(jets.hFlav, 2, -999.0)
            third_jet_particleNetAK4_CvsL = choose_jet(jets.particleNetAK4_CvsL, 2, -999.0)
            third_jet_particleNetAK4_CvsB = choose_jet(jets.particleNetAK4_CvsB, 2, -999.0)
            third_jet_particleNetAK4_B = choose_jet(jets.particleNetAK4_B, 2, -999.0)
            third_jet_jet_pn_b_plus_c = choose_jet(jets.pn_b_plus_c, 2, -1.0)
            third_jet_jet_pn_b_vs_c = choose_jet(jets.pn_b_vs_c, 2, -1.0)

            fourth_jet_pt = choose_jet(jets.pt, 3, -999.0)
            fourth_jet_eta = choose_jet(jets.eta, 3, -999.0)
            fourth_jet_phi = choose_jet(jets.phi, 3, -999.0)
            fourth_jet_mass = choose_jet(jets.mass, 3, -999.0)
            fourth_jet_charge = choose_jet(jets.charge, 3, -999.0)
            fourth_jet_hFlav = choose_jet(jets.hFlav, 3, -999.0)
            fourth_jet_particleNetAK4_CvsL = choose_jet(jets.particleNetAK4_CvsL, 3, -999.0)
            fourth_jet_particleNetAK4_CvsB = choose_jet(jets.particleNetAK4_CvsB, 3, -999.0)
            fourth_jet_particleNetAK4_B = choose_jet(jets.particleNetAK4_B, 3, -999.0)
            fourth_jet_jet_pn_b_plus_c = choose_jet(jets.pn_b_plus_c, 3, -1.0)
            fourth_jet_jet_pn_b_vs_c = choose_jet(jets.pn_b_vs_c, 3, -1.0)

            # first_sv_pt = choose_jet(sv.pt, 0, -999.0)
            # first_sv_eta = choose_jet(sv.eta, 0, -999.0)
            # first_sv_phi = choose_jet(sv.phi, 0, -999.0)
            # first_sv_mass = choose_jet(sv.mass, 0, -999.0)
            # first_sv_charge = choose_jet(sv.charge, 0, -999.0)
            # first_sv_dlen = choose_jet(sv.dlen, 0, -999.0)
            # first_sv_dlenSig = choose_jet(sv.dlenSig, 0, -999.0)
            # first_sv_dxy = choose_jet(sv.dxy, 0, -999.0)
            # first_sv_dxySig = choose_jet(sv.dxySig, 0, -999.0)
            # first_sv_pAngle = choose_jet(sv.pAngle, 0, -999.0)
            # first_sv_chi2 = choose_jet(sv.chi2, 0, -999.0)
            # first_sv_x = choose_jet(sv.x, 0, -999.0)
            # first_sv_y = choose_jet(sv.y, 0, -999.0)
            # first_sv_z = choose_jet(sv.z, 0, -999.0)
            # first_sv_ndof = choose_jet(sv.ndof, 0, -999.0)
            # first_sv_ntracks = choose_jet(sv.ntracks, 0, -999.0)

            first_muon_pt = choose_jet(muons.pt, 0, -999.0)
            first_muon_eta = choose_jet(muons.eta, 0, -999.0)
            first_muon_phi = choose_jet(muons.phi, 0, -999.0)
            first_muon_mass = choose_jet(muons.mass, 0, -999.0)
            first_muon_charge = choose_jet(muons.charge, 0, -999.0)

            first_electron_pt = choose_jet(electrons.pt, 0, -999.0)
            first_electron_eta = choose_jet(electrons.eta, 0, -999.0)
            first_electron_phi = choose_jet(electrons.phi, 0, -999.0)
            first_electron_mass = choose_jet(electrons.mass, 0, -999.0)
            first_electron_charge = choose_jet(electrons.charge, 0, -999.0)

            if self.data_kind == "mc":
                first_gen_jet_pt = choose_jet(genJets.pt, 0, -999.0)
                first_gen_jet_eta = choose_jet(genJets.eta, 0, -999.0)
                first_gen_jet_mass = choose_jet(genJets.mass, 0, -999.0)
                first_gen_jet_phi = choose_jet(genJets.phi, 0, -999.0)
                first_gen_jet_hFlav = choose_jet(genJets.hadronFlavour, 0, -999.0)
                first_gen_jet_hFlav2 = choose_jet(gen_jets.hFlav, 0, -999.0)

                second_gen_jet_pt = choose_jet(genJets.pt, 1, -999.0)
                second_gen_jet_eta = choose_jet(genJets.eta, 1, -999.0)
                second_gen_jet_mass = choose_jet(genJets.mass, 1, -999.0)
                second_gen_jet_phi = choose_jet(genJets.phi, 1, -999.0)
                second_gen_jet_hFlav = choose_jet(genJets.hadronFlavour, 1, -999.0)

                third_gen_jet_pt = choose_jet(genJets.pt, 2, -999.0)
                third_gen_jet_eta = choose_jet(genJets.eta, 2, -999.0)
                third_gen_jet_mass = choose_jet(genJets.mass, 2, -999.0)
                third_gen_jet_phi = choose_jet(genJets.phi, 2, -999.0)
                third_gen_jet_hFlav = choose_jet(genJets.hadronFlavour, 2, -999.0)

                fourth_gen_jet_pt = choose_jet(genJets.pt, 3, -999.0)
                fourth_gen_jet_eta = choose_jet(genJets.eta, 3, -999.0)
                fourth_gen_jet_mass = choose_jet(genJets.mass, 3, -999.0)
                fourth_gen_jet_phi = choose_jet(genJets.phi, 3, -999.0)
                fourth_gen_jet_hFlav = choose_jet(genJets.hadronFlavour, 3, -999.0)

                had_jets = jets[ak.argsort(abs(jets.hFlav), ascending=False, axis=1)]
                had_gen_jets = genJets[ak.argsort(abs(genJets.hadronFlavour), ascending=False, axis=1)]

                first_had_jet_hadFlav = choose_jet(had_jets.hFlav, 0, -999.0)
                first_had_gen_jet_hadFlav = choose_jet(had_gen_jets.hadronFlavour, 0, -999.0)

                charmed_flag = (first_had_jet_hadFlav == 4)
                bottom_flag = (first_had_jet_hadFlav == 5)
                charmed_flag_gen = (first_had_gen_jet_hadFlav == 4)
                bottom_flag_gen = (first_had_gen_jet_hadFlav == 5)
            else:
                first_gen_jet_pt = ak.ones_like(diphotons.pt) * -999.0
                first_gen_jet_eta = ak.ones_like(diphotons.pt) * -999.0
                first_gen_jet_mass = ak.ones_like(diphotons.pt) * -999.0
                first_gen_jet_phi = ak.ones_like(diphotons.pt) * -999.0
                first_gen_jet_hFlav = ak.ones_like(diphotons.pt) * -999.0
                first_gen_jet_hFlav2 = ak.ones_like(diphotons.pt) * -999.0

                second_gen_jet_pt = ak.ones_like(diphotons.pt) * -999.0
                second_gen_jet_eta = ak.ones_like(diphotons.pt) * -999.0
                second_gen_jet_mass = ak.ones_like(diphotons.pt) * -999.0
                second_gen_jet_phi = ak.ones_like(diphotons.pt) * -999.0
                second_gen_jet_hFlav = ak.ones_like(diphotons.pt) * -999.0

                third_gen_jet_pt = ak.ones_like(diphotons.pt) * -999.0
                third_gen_jet_eta = ak.ones_like(diphotons.pt) * -999.0
                third_gen_jet_mass = ak.ones_like(diphotons.pt) * -999.0
                third_gen_jet_phi = ak.ones_like(diphotons.pt) * -999.0
                third_gen_jet_hFlav = ak.ones_like(diphotons.pt) * -999.0

                fourth_gen_jet_pt = ak.ones_like(diphotons.pt) * -999.0
                fourth_gen_jet_eta = ak.ones_like(diphotons.pt) * -999.0
                fourth_gen_jet_mass = ak.ones_like(diphotons.pt) * -999.0
                fourth_gen_jet_phi = ak.ones_like(diphotons.pt) * -999.0
                fourth_gen_jet_hFlav = ak.ones_like(diphotons.pt) * -999.0

                charmed_flag = ak.ones_like(diphotons.pt) * -999.0
                bottom_flag = ak.ones_like(diphotons.pt) * -999.0
                charmed_flag_gen = ak.ones_like(diphotons.pt) * -999.0
                bottom_flag_gen = ak.ones_like(diphotons.pt) * -999.0

            dipho_events["first_jet_jet_pn_b_plus_c"] = first_jet_jet_pn_b_plus_c
            dipho_events["first_jet_jet_pn_b_vs_c"] = first_jet_jet_pn_b_vs_c
            dipho_events["second_jet_jet_pn_b_plus_c"] = second_jet_jet_pn_b_plus_c
            dipho_events["second_jet_jet_pn_b_vs_c"] = second_jet_jet_pn_b_vs_c
            dipho_events["third_jet_jet_pn_b_plus_c"] = third_jet_jet_pn_b_plus_c
            dipho_events["third_jet_jet_pn_b_vs_c"] = third_jet_jet_pn_b_vs_c

            diphotons["first_jet_pt"] = first_jet_pt
            diphotons["first_jet_eta"] = first_jet_eta
            diphotons["first_jet_phi"] = first_jet_phi
            diphotons["first_jet_mass"] = first_jet_mass
            diphotons["first_jet_charge"] = first_jet_charge
            diphotons["first_jet_hFlav"] = first_jet_hFlav
            diphotons["first_jet_DeepFlavour_CvsL"] = first_jet_DeepFlavour_CvsL
            diphotons["first_jet_DeepFlavour_CvsB"] = first_jet_DeepFlavour_CvsB
            diphotons["first_jet_DeepFlavour_B"] = first_jet_DeepFlavour_B
            diphotons["first_jet_DeepFlavour_QG"] = first_jet_DeepFlavour_QG
            diphotons["first_jet_particleNetAK4_CvsL"] = first_jet_particleNetAK4_CvsL
            diphotons["first_jet_particleNetAK4_CvsB"] = first_jet_particleNetAK4_CvsB
            diphotons["first_jet_particleNetAK4_QvsG"] = first_jet_particleNetAK4_QvsG
            diphotons["first_jet_particleNetAK4_B"] = first_jet_particleNetAK4_B
            diphotons["first_jet_jet_pn_b"] = first_jet_jet_pn_b
            diphotons["first_jet_jet_pn_c"] = first_jet_jet_pn_c
            diphotons["first_jet_jet_pn_uds"] = first_jet_jet_pn_uds
            diphotons["first_jet_jet_pn_g"] = first_jet_jet_pn_g
            diphotons["first_jet_jet_pn_b_plus_c"] = first_jet_jet_pn_b_plus_c
            diphotons["first_jet_jet_pn_b_vs_c"] = first_jet_jet_pn_b_vs_c
            diphotons["first_jet_n_sv"] = first_jet_n_sv
            diphotons["first_jet_n_muons"] = first_jet_n_muons
            diphotons["first_jet_n_electrons"] = first_jet_n_electrons

            diphotons["first_pt_jet_pt"] = first_pt_jet_pt
            diphotons["first_pt_jet_eta"] = first_pt_jet_eta
            diphotons["first_pt_jet_phi"] = first_pt_jet_phi
            diphotons["first_pt_jet_mass"] = first_pt_jet_mass
            diphotons["first_pt_jet_charge"] = first_pt_jet_charge
            diphotons["first_pt_jet_hFlav"] = first_pt_jet_hFlav
            diphotons["first_pt_jet_DeepFlavour_CvsL"] = first_pt_jet_DeepFlavour_CvsL
            diphotons["first_pt_jet_DeepFlavour_CvsB"] = first_pt_jet_DeepFlavour_CvsB
            diphotons["first_pt_jet_DeepFlavour_B"] = first_pt_jet_DeepFlavour_B
            diphotons["first_pt_jet_DeepFlavour_QG"] = first_pt_jet_DeepFlavour_QG

            diphotons["second_jet_pt"] = second_jet_pt
            diphotons["second_jet_eta"] = second_jet_eta
            diphotons["second_jet_phi"] = second_jet_phi
            diphotons["second_jet_mass"] = second_jet_mass
            diphotons["second_jet_charge"] = second_jet_charge
            diphotons["second_jet_hFlav"] = second_jet_hFlav
            diphotons["second_jet_DeepFlavour_CvsL"] = second_jet_DeepFlavour_CvsL
            diphotons["second_jet_DeepFlavour_CvsB"] = second_jet_DeepFlavour_CvsB
            diphotons["second_jet_DeepFlavour_B"] = second_jet_DeepFlavour_B
            diphotons["second_jet_DeepFlavour_QG"] = second_jet_DeepFlavour_QG
            diphotons["second_jet_particleNetAK4_CvsL"] = second_jet_particleNetAK4_CvsL
            diphotons["second_jet_particleNetAK4_CvsB"] = second_jet_particleNetAK4_CvsB
            diphotons["second_jet_particleNetAK4_B"] = second_jet_particleNetAK4_B
            diphotons["second_jet_jet_pn_b_plus_c"] = second_jet_jet_pn_b_plus_c
            diphotons["second_jet_jet_pn_b_vs_c"] = second_jet_jet_pn_b_vs_c
            diphotons["second_jet_n_sv"] = second_jet_n_sv

            diphotons["second_pt_jet_pt"] = second_pt_jet_pt
            diphotons["second_pt_jet_eta"] = second_pt_jet_eta
            diphotons["second_pt_jet_phi"] = second_pt_jet_phi
            diphotons["second_pt_jet_mass"] = second_pt_jet_mass
            diphotons["second_pt_jet_charge"] = second_pt_jet_charge
            diphotons["second_pt_jet_hFlav"] = second_pt_jet_hFlav
            diphotons["second_pt_jet_DeepFlavour_CvsL"] = second_pt_jet_DeepFlavour_CvsL
            diphotons["second_pt_jet_DeepFlavour_CvsB"] = second_pt_jet_DeepFlavour_CvsB
            diphotons["second_pt_jet_DeepFlavour_B"] = second_pt_jet_DeepFlavour_B
            diphotons["second_pt_jet_DeepFlavour_QG"] = second_pt_jet_DeepFlavour_QG

            diphotons["third_jet_pt"] = third_jet_pt
            diphotons["third_jet_eta"] = third_jet_eta
            diphotons["third_jet_phi"] = third_jet_phi
            diphotons["third_jet_mass"] = third_jet_mass
            diphotons["third_jet_charge"] = third_jet_charge
            diphotons["third_jet_hFlav"] = third_jet_hFlav
            diphotons["third_jet_particleNetAK4_CvsL"] = third_jet_particleNetAK4_CvsL
            diphotons["third_jet_particleNetAK4_CvsB"] = third_jet_particleNetAK4_CvsB
            diphotons["third_jet_particleNetAK4_B"] = third_jet_particleNetAK4_B
            diphotons["third_jet_jet_pn_b_plus_c"] = third_jet_jet_pn_b_plus_c
            diphotons["third_jet_jet_pn_b_vs_c"] = third_jet_jet_pn_b_vs_c

            diphotons["fourth_jet_pt"] = fourth_jet_pt
            diphotons["fourth_jet_eta"] = fourth_jet_eta
            diphotons["fourth_jet_phi"] = fourth_jet_phi
            diphotons["fourth_jet_mass"] = fourth_jet_mass
            diphotons["fourth_jet_charge"] = fourth_jet_charge
            diphotons["fourth_jet_hFlav"] = fourth_jet_hFlav
            diphotons["fourth_jet_particleNetAK4_CvsL"] = fourth_jet_particleNetAK4_CvsL
            diphotons["fourth_jet_particleNetAK4_CvsB"] = fourth_jet_particleNetAK4_CvsB
            diphotons["fourth_jet_particleNetAK4_B"] = fourth_jet_particleNetAK4_B
            diphotons["fourth_jet_jet_pn_b_plus_c"] = fourth_jet_jet_pn_b_plus_c
            diphotons["fourth_jet_jet_pn_b_vs_c"] = fourth_jet_jet_pn_b_vs_c

            diphotons["first_gen_jet_pt"] = first_gen_jet_pt
            diphotons["first_gen_jet_eta"] = first_gen_jet_eta
            diphotons["first_gen_jet_mass"] = first_gen_jet_mass
            diphotons["first_gen_jet_phi"] = first_gen_jet_phi
            diphotons["first_gen_jet_hFlav"] = first_gen_jet_hFlav
            diphotons["first_gen_jet_hFlav2"] = first_gen_jet_hFlav2

            diphotons["second_gen_jet_pt"] = second_gen_jet_pt
            diphotons["second_gen_jet_eta"] = second_gen_jet_eta
            diphotons["second_gen_jet_mass"] = second_gen_jet_mass
            diphotons["second_gen_jet_phi"] = second_gen_jet_phi
            diphotons["second_gen_jet_hFlav"] = second_gen_jet_hFlav

            diphotons["third_gen_jet_pt"] = third_gen_jet_pt
            diphotons["third_gen_jet_eta"] = third_gen_jet_eta
            diphotons["third_gen_jet_mass"] = third_gen_jet_mass
            diphotons["third_gen_jet_phi"] = third_gen_jet_phi
            diphotons["third_gen_jet_hFlav"] = third_gen_jet_hFlav

            diphotons["fourth_gen_jet_pt"] = fourth_gen_jet_pt
            diphotons["fourth_gen_jet_eta"] = fourth_gen_jet_eta
            diphotons["fourth_gen_jet_mass"] = fourth_gen_jet_mass
            diphotons["fourth_gen_jet_phi"] = fourth_gen_jet_phi
            diphotons["fourth_gen_jet_hFlav"] = fourth_gen_jet_hFlav

            # diphotons["first_sv_pt"] = first_sv_pt
            # diphotons["first_sv_eta"] = first_sv_eta
            # diphotons["first_sv_phi"] = first_sv_phi
            # diphotons["first_sv_mass"] = first_sv_mass
            # diphotons["first_sv_charge"] = first_sv_charge
            # diphotons["first_sv_dlen"] = first_sv_dlen
            # diphotons["first_sv_dlenSig"] = first_sv_dlenSig
            # diphotons["first_sv_dxy"] = first_sv_dxy
            # diphotons["first_sv_dxySig"] = first_sv_dxySig
            # diphotons["first_sv_pAngle"] = first_sv_pAngle
            # diphotons["first_sv_chi2"] = first_sv_chi2
            # diphotons["first_sv_x"] = first_sv_x
            # diphotons["first_sv_y"] = first_sv_y
            # diphotons["first_sv_z"] = first_sv_z
            # diphotons["first_sv_ndof"] = first_sv_ndof
            # diphotons["first_sv_ntracks"] = first_sv_ntracks

            diphotons["first_muon_pt"] = first_muon_pt
            diphotons["first_muon_eta"] = first_muon_eta
            diphotons["first_muon_phi"] = first_muon_phi
            diphotons["first_muon_mass"] = first_muon_mass
            diphotons["first_muon_charge"] = first_muon_charge

            diphotons["first_electron_pt"] = first_electron_pt
            diphotons["first_electron_eta"] = first_electron_eta
            diphotons["first_electron_phi"] = first_electron_phi
            diphotons["first_electron_mass"] = first_electron_mass
            diphotons["first_electron_charge"] = first_electron_charge

            diphotons["n_jets"] = n_jets
            diphotons["n_b_jets_medium"] = n_b_jets_medium
            diphotons["n_b_jets_loose"] = n_b_jets_loose
            diphotons["n_c_had"] = charmed_had_num
            diphotons["n_b_had"] = bottom_had_num
            diphotons["charmed_flag"] = charmed_flag
            diphotons["bottom_flag"] = bottom_flag
            diphotons["charmed_flag_gen"] = charmed_flag_gen
            diphotons["bottom_flag_gen"] = bottom_flag_gen

            diphotons["n_ME_c_in"] = ME_in_charm_num
            diphotons["n_ME_b_in"] = ME_in_bottom_num
            diphotons["n_ME_l_in"] = ME_in_light_num
            diphotons["n_ME_g_in"] = ME_in_gluon_num

            diphotons["n_ME_c_out"] = ME_out_charm_num
            diphotons["n_ME_b_out"] = ME_out_bottom_num
            diphotons["n_ME_l_out"] = ME_out_light_num
            diphotons["n_ME_g_out"] = ME_out_gluon_num

            diphotons["dijet_pt"] = dijet_pt
            diphotons["dijet_eta"] = dijet_eta
            diphotons["dijet_phi"] = dijet_phi
            diphotons["dijet_mass"] = dijet_mass

            diphotons["LeadPhoton_pt"] = diphotons.pho_lead.pt
            diphotons["SubleadPhoton_pt"] = diphotons.pho_sublead.pt
            diphotons["LeadPhoton_energy"] = diphotons.pho_lead.energy
            diphotons["SubleadPhoton_energy"] = diphotons.pho_sublead.energy

            diphotons["LeadPhoton_r9"] = diphotons.pho_lead.r9
            diphotons["LeadPhoton_s4"] = diphotons.pho_lead.s4
            diphotons["LeadPhoton_sieie"] = diphotons.pho_lead.sieie
            diphotons["LeadPhoton_sieip"] = diphotons.pho_lead.sieip
            diphotons["LeadPhoton_etaWidth"] = diphotons.pho_lead.etaWidth
            diphotons["LeadPhoton_phiWidth"] = diphotons.pho_lead.phiWidth
            diphotons["LeadPhoton_pfChargedIsoPFPV"] = diphotons.pho_lead.pfChargedIsoPFPV
            diphotons["LeadPhoton_pfChargedIsoWorstVtx"] = diphotons.pho_lead.pfChargedIsoWorstVtx
            diphotons["LeadPhoton_pfPhoIso03"] = diphotons.pho_lead.pfPhoIso03

            # Add genPartFlav for lead/sublead photons (MC only; fill -1 for data)
            if self.data_kind == "mc" and hasattr(diphotons.pho_lead, "genPartFlav"):
                diphotons["LeadPhoton_genPartFlav"] = ak.fill_none(diphotons.pho_lead.genPartFlav, -1)
                diphotons["LeadPhoton_genPartFlav"] = ak.values_astype(diphotons["LeadPhoton_genPartFlav"], int)

                diphotons["SubleadPhoton_genPartFlav"] = ak.fill_none(diphotons.pho_sublead.genPartFlav, -1)
                diphotons["SubleadPhoton_genPartFlav"] = ak.values_astype(diphotons["SubleadPhoton_genPartFlav"], int)

                logger.info(f"Added genPartFlav for MC: lead={diphotons['LeadPhoton_genPartFlav']}")
                logger.info(f"Added genPartFlav for MC: sublead={diphotons['SubleadPhoton_genPartFlav']}")
            else:
                diphotons["LeadPhoton_genPartFlav"] = ak.values_astype(ak.ones_like(diphotons.pt) * (-1), int)
                diphotons["SubleadPhoton_genPartFlav"] = ak.values_astype(ak.ones_like(diphotons.pt) * (-1), int)

            # run taggers on the events list with added diphotons
            # the shape here is ensured to be broadcastable
            for tagger in self.taggers:
                (
                    diphotons["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    dipho_events, diphotons
                )  # creates new column in diphotons - tagger priority, or 0, also return list of histrograms here?
                histos_etc.update(tagger_extra)

            # if there are taggers to run, arbitrate by them first
            # Deal with order of tagger priorities
            # Turn from diphoton jagged array to whether or not an event was selected
            if len(self.taggers):
                counts = ak.num(diphotons.pt, axis=1)
                flat_tags = numpy.stack(
                    (
                        ak.flatten(
                            diphotons[
                                "_".join([tagger.name, str(tagger.priority)])
                            ]
                        )
                        for tagger in self.taggers
                    ),
                    axis=1,
                )
                tags = ak.from_regular(
                    ak.unflatten(flat_tags, counts), axis=2
                )
                winner = ak.min(tags[tags != 0], axis=2)
                diphotons["best_tag"] = winner

                # lowest priority is most important (ascending sort)
                # leave in order of diphoton pT in case of ties (stable sort)
                sorted_gg = ak.argsort(diphotons.best_tag, stable=True)
                diphotons = diphotons[sorted_gg]

            original_diphotons = copy.copy(diphotons)
            # annotate diphotons with event information
            diphotons["event"] = dipho_events.event
            diphotons["lumi"] = dipho_events.luminosityBlock
            diphotons["run"] = dipho_events.run
            # nPV just for validation of pileup reweighting
            diphotons["nPV"] = dipho_events.PV.npvs if not self.data_kind else ak.ones_like(dipho_events.event)
            diphotons["nPU"] = dipho_events.Pileup.nPU if not self.data_kind else ak.ones_like(dipho_events.event)
            diphotons["fixedGridRhoAll"] = dipho_events.Rho.fixedGridRhoAll

            # global variables for ggH vs RB BDT
            diphotons["nTau"] = dipho_events.nTau if hasattr(dipho_events, "nTau") else ak.num(dipho_events.Tau)
            diphotons["nMuon"] = dipho_events.nMuon if hasattr(dipho_events, "nMuon") else ak.num(dipho_events.Muon)
            diphotons["nElectron"] = dipho_events.nElectron if hasattr(dipho_events, "nElectron") else ak.num(dipho_events.Electron)
            diphotons["MET_pt"] = dipho_events.PuppiMET.pt
            diphotons["MET_phi"] = dipho_events.PuppiMET.phi
            diphotons["MET_sumEt"] = dipho_events.PuppiMET.sumEt
            diphotons["MET_significance"] = dipho_events.MET.significance

            # SV info
            diphotons["SV_ntracks"] = dipho_events.SV.ntracks
            diphotons["SV_pt"] = dipho_events.SV.pt
            diphotons["SV_eta"] = dipho_events.SV.eta
            diphotons["SV_phi"] = dipho_events.SV.phi
            diphotons["SV_charge"] = dipho_events.SV.charge
            diphotons["SV_mass"] = dipho_events.SV.mass
            diphotons["SV_dlen"] = dipho_events.SV.dlen
            diphotons["SV_dlenSig"] = dipho_events.SV.dlenSig
            diphotons["SV_pAngle"] = dipho_events.SV.pAngle
            diphotons["SV_dxy"] = dipho_events.SV.dxy
            diphotons["SV_dxySig"] = dipho_events.SV.dxySig

            # here I add lead jet to event because I need it for cTagSF evaluation
            # this may be not needed if one change the cTagSF function in event_weight_systematics.py to use the jet collection
            dipho_events["first_jet_pt"] = first_jet_pt
            dipho_events["first_jet_eta"] = first_jet_eta
            dipho_events["first_jet_phi"] = first_jet_phi
            dipho_events["first_jet_mass"] = first_jet_mass
            dipho_events["first_jet_charge"] = first_jet_charge
            dipho_events["first_jet_hFlav"] = ak.values_astype(first_jet_hFlav, int)
            dipho_events["first_jet_DeepFlavour_CvsL"] = first_jet_DeepFlavour_CvsL
            dipho_events["first_jet_DeepFlavour_CvsB"] = first_jet_DeepFlavour_CvsB

            dipho_events["sel_jets"] = jets
            dipho_events["n_jets"] = n_jets
            diphotons, ggh_vs_hb_mva_events = self.add_ggh_vs_hb_mva_vars(diphotons, dipho_events)
            if self.ggh_vs_hb_mva is not None:
                diphotons = self.add_ggh_vs_hb_mva(diphotons, ggh_vs_hb_mva_events)

            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = dipho_events.genWeight
                diphotons["dZ"] = dipho_events.GenVtx.z - dipho_events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                diphotons["HTXS_Higgs_pt"] = dipho_events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = dipho_events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = dipho_events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                diphotons["HTXS_stage_0"] = dipho_events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                diphotons["dZ"] = ak.zeros_like(dipho_events.PV.z)

            # drop events without a preselected diphoton candidate
            # drop events without a tag, if there are tags
            if len(self.taggers):
                selection_mask = ~(
                    ak.is_none(diphotons)
                    | ak.is_none(diphotons.best_tag)
                )
                diphotons = diphotons[selection_mask]
            else:
                selection_mask = ~ak.is_none(diphotons)
                diphotons = diphotons[selection_mask]
            # return if there is no surviving events
            if len(diphotons) == 0:
                logger.info("No surviving events in this run!")
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(dipho_events[selection_mask]), storeIndividual=True)
                # set weights to generator weights
                event_weights._weight = ak.to_numpy(events["genWeight"][selection_mask])

                # corrections to event weights:
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        varying_function = available_weight_corrections[
                            correction_name
                        ]
                        event_weights = varying_function(
                            events=dipho_events[selection_mask],
                            photons=original_diphotons[selection_mask],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                            meta=self.meta,
                        )
                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(dipho_events, "LHEScaleWeight"):
                                    diphotons["nweight_LHEScale"] = ak.num(
                                        dipho_events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    diphotons[
                                        "weight_LHEScale"
                                    ] = dipho_events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(dipho_events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    diphotons["nweight_LHEPdf"] = (
                                        ak.num(
                                            dipho_events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    diphotons[
                                        "weight_LHEPdf"
                                    ] = dipho_events.LHEPdfWeight[selection_mask][
                                        :, :-2
                                    ]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                varying_function = available_weight_systematics[
                                    systematic_name
                                ]
                                event_weights = varying_function(
                                    events=dipho_events[selection_mask],
                                    photons=original_diphotons[selection_mask],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                    meta=self.meta,
                                )

                diphotons["weight"] = event_weights.weight()
                diphotons["weight_central"] = event_weights.weight() / dipho_events[selection_mask].genWeight
                diphotons["genWeight"] = dipho_events[selection_mask].genWeight

                metadata["sum_weight_central"] = str(
                    ak.sum(event_weights.weight())
                )
                metadata["sum_weight_central_wo_bTagSF"] = str(
                    ak.sum(event_weights.weight() / (event_weights.partial_weight(include=["bTagSF"])))
                )

                # Store variations with respect to central weight
                if do_variation == "nominal":
                    if len(event_weights.variations):
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        diphotons["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )
                        if ("bTagSF" in modifier):
                            metadata["sum_weight_" + modifier] = str(
                                ak.sum(event_weights.weight(modifier=modifier))
                            )

            # Add weight variables (=1) for data for consistent datasets
            else:
                diphotons["weight_central"] = ak.ones_like(
                    diphotons["event"]
                )
                diphotons["weight"] = ak.ones_like(diphotons["event"])
                diphotons["genWeight"] = ak.ones_like(diphotons["event"])

            # Compute and store the different variations of sigma_m_over_m
            diphotons = compute_sigma_m(diphotons, processor='base', flow_corrections=self.doFlow_corrections, smear=self.Smear_sigma_m, IsData=(self.data_kind == "data"))

            diphotons["CMS_hgg_mass"] = diphotons.mass
            # Decorrelating the mass resolution - Still need to supress the decorrelator noises
            if self.doDeco:

                # Decorrelate nominal sigma_m_over_m
                diphotons["sigma_m_over_m_nominal_decorr"] = decorrelate_mass_resolution(diphotons, type="nominal", year=self.year[dataset_name][0])

                # decorrelate smeared nominal sigma_m_overm_m
                if (self.Smear_sigma_m):
                    diphotons["sigma_m_over_m_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="smeared", year=self.year[dataset_name][0])

                # decorrelate flow corrected sigma_m_over_m
                if (self.doFlow_corrections):
                    diphotons["sigma_m_over_m_corr_decorr"] = decorrelate_mass_resolution(diphotons, type="corr", year=self.year[dataset_name][0])

                # decorrelate flow corrected smeared sigma_m_over_m
                if (self.doFlow_corrections and self.Smear_sigma_m):
                    if self.data_kind == "data" and ("Scale_IJazZ" in correction_names or "Scale2G_IJazZ" in correction_names):
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)
                    elif self.data_kind == "mc" and ("Smearing2G_IJazZ" in correction_names or "Smearing_IJazZ" in correction_names):
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)
                    else:
                        diphotons["sigma_m_over_m_corr_smeared_decorr"] = decorrelate_mass_resolution(diphotons, type="corr_smeared", year=self.year[dataset_name][0], IsSAS_ET_Dependent=True)

                # Instead of the nominal sigma_m_over_m, we will use the smeared version of it -> (https://indico.cern.ch/event/1319585/#169-update-on-the-run-3-mass-r)
                # else:
                #    warnings.warn("Smeamering need to be applied in order to decorrelate the (Smeared) mass resolution. -- Exiting!")
                #    sys.exit(0)

            if self.output_location is not None:
                if self.output_format == "root":
                    df = diphoton_list_to_pandas(self, diphotons)
                else:
                    akarr = diphoton_ak_array(self, diphotons)
                fname = (
                    events.attrs[
                        "@events_factory"
                    ]._partition_key.replace("/", "_")
                    + ".%s" % self.output_format
                )
                fname = (fname.replace("%2F","")).replace("%3B1","")
                subdirs = []
                if "dataset" in events.metadata:
                    subdirs.append(events.metadata["dataset"])
                subdirs.append(do_variation)
                if self.output_format == "root":
                    dump_pandas(self, df, fname, self.output_location, subdirs)
                else:
                    dump_ak_array(
                        self, akarr, fname, self.output_location, metadata, subdirs,
                    )

        return histos_etc

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass

    def add_multiclass_diphoton_mva(
        self, diphotons: ak.Array, events: ak.Array
    ) -> ak.Array:
        return calculate_multiclass_diphoton_mva(
            self,
            (self.multiclass_diphoton_mva, self.meta["HiggsDNA_DiPhotonMVA_multiclass"]["inputs"]),
            diphotons,
            events,
        )

    def add_ggh_vs_hb_mva(
        self, diphotons: ak.Array, events: ak.Array
    ) -> ak.Array:
        return calculate_ggh_vs_hb_mva(
            self,
            (self.ggh_vs_hb_mva, self.meta["hpcMVA_ggh_vs_hb"]["inputs"]),
            diphotons,
            events,
            self.year[self.dataset_name][0],
        )

    def add_ggh_vs_hb_mva_vars(
        self, diphotons: ak.Array, events: ak.Array
    ) -> ak.Array:
        return fill_ggh_vs_hb_mva_vars(
            self,
            self.meta["hpcMVA_ggh_vs_hb"]["inputs"],
            diphotons,
            events,
            self.year[self.dataset_name][0],
        )
