from higgs_dna.workflows.skeleton import HggSkeletonProcessor
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EELeak_region import veto_EEleak_flag
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.tools.gen_helpers import (
    get_fiducial_flag,
    get_genJets,
    get_higgs_gen_attributes,
    label_associated_decay,
)
from higgs_dna.tools.sigma_m_tools import compute_sigma_m
from higgs_dna.tools.jetID import add_jetId
from higgs_dna.selections.photon_selections import photon_preselection
from higgs_dna.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from higgs_dna.selections.lepton_selections import select_electrons, select_muons, select_taus
from higgs_dna.selections.jet_selections import select_jets, select_jets_eta_dependent, jetvetomap, getBTagMVACut
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
    apply_naming_convention,
)
from higgs_dna.utils.misc_utils import choose_jet
from higgs_dna.tools.flow_corrections import apply_flow_corrections_to_photons

from higgs_dna.tools.mass_decorrelator import decorrelate_mass_resolution

from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections
from higgs_dna.systematics import apply_systematic_variations_object_level
from higgs_dna.systematics.MET_systematics import apply_type1_met_correction
from higgs_dna.systematics.event_weight_systematics import calculate_NNLOPS_sf

import warnings
from typing import Any, Dict, List, Optional
import awkward as ak
import json
import numpy
import pandas as pd
import sys
import vector
from coffea.analysis_tools import Weights
import copy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class STXSProcessor(HggSkeletonProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Optional[Dict[str, List[str]]] = None,
        corrections: Optional[Dict[str, List[str]]] = None,
        apply_trigger: bool = False,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        nano_version: int = None,
        bTagEffFileName: Optional[str] = None,
        trigger_group: str = ".*DoubleEG.*",
        analysis: str = "mainAnalysis",
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Optional[Dict[str, List[str]]] = None,
        fiducialCuts: str = "classical",
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

        self.name_convention = "DAS"

        # Gen-level associated decay classification
        self.associated_decay_config = [
            {
                "name": "ZH",
                "associate": {"pdgId": 23, "multiplicity": 1},
                "decay": {"pdgId": 23, "relationship_to_associate": "self"},
                "categories": {
                    "ZH_ll": {"n_lep": 2},
                    "ZH_nunu": {"n_nu": 2},
                    "ZH_qq": {"n_q": 2},
                },
                "n_higgs": 1,
                "orthogonal_categories": True,
            },
            {
                "name": "ttH",
                "associate": {"pdgId": 6, "multiplicity": 2},
                "decay": {"pdgId": 24, "relationship_to_associate": "child"},
                "categories": {
                    "ttH_lep": {"n_lep": 2, "n_nu": 2},
                    "ttH_semilep": {"n_lep": 1, "n_nu": 1},
                    "ttH_had": {"n_q": 4},
                },
                "n_higgs": 1,
                "orthogonal_categories": True,
            },
            {
                "name": "tH",
                "associate": {"pdgId": 6, "multiplicity": 1},
                "decay": {"pdgId": 24, "relationship_to_associate": "child"},
                "categories": {
                    "tH_lep": {"n_lep": 1, "n_nu": 1},
                    "tH_had": {"n_q": 2},
                },
                "n_higgs": 1,
                "orthogonal_categories": True,
            }
        ]

        # Definitions for particle-type counters used in associated decay categories
        self.associated_decay_particle_map = {
            "n_lep": (11, 13, 15),
            "n_nu": (12, 14, 16),
            "n_q": tuple(range(1, 9)),
        }

        # For eta-dependent jet pt cuts
        self.jet_eta_thresholds = [2.5, 3.0, 4.7]

        # tau selection cuts (for the moment these are just the same as ditau)
        self.tau_pt_threshold = 18
        self.tau_max_eta = 2.3
        self.tau_max_dz = 0.2

        self.tau_photon_min_dr = 0.2

        self.jet_tau_min_dr = 0.4
        self.clean_jet_tau = True

        bjet_mva = ["deepJet"]
        if self.nano_version >= 12:
            bjet_mva.append("particleNet")
        if self.nano_version in [12, 13]:
            bjet_mva.append("robustParticleTransformer")
        if self.nano_version > 13:
            bjet_mva.append("btagUParTAK4B")
        self.bjet_mva = bjet_mva
        self.bjet_wp = ["L", "M", "T", "XT", "XXT"]

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
        except KeyError:
            systematic_names = []

        if self.data_kind == "mc":
            generator = None
            for i, correction_name in enumerate(correction_names):
                if (correction_name == "NNLOPS") or (correction_name == "NNLOPS_amcatnlo") or (correction_name == "NNLOPS_powheg"):
                    nnlops_name = correction_names.pop(i)
                    generator = "powheg" if "powheg" in nnlops_name else "mcatnlo"
                    break
            # ensure that there are no multiple NNLOPS entries
            if generator is not None:
                for correction_name in correction_names:
                    if (correction_name == "NNLOPS") or (correction_name == "NNLOPS_amcatnlo") or (correction_name == "NNLOPS_powheg"):
                        raise ValueError("Multiple NNLOPS entries found in corrections list")
                events["genWeight"] = events.genWeight * calculate_NNLOPS_sf(events, dataset_name, generator)

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
        # apply jetvetomap: only retain events that without any jets in the EE leakage region
        if not self.skipJetVetoMap:
            events = jetvetomap(
                self, events, logger, dataset_name, year=self.year[dataset_name][0]
            )
        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(numpy.sum(events.genWeight.to_numpy()))

            # Add sum of gen weights before selection for each HTXS.stage_1_2 bin
            base_dict = {
                "HTXS_stage1_2_cat_pTjet30GeV": events.HTXS.stage1_2_cat_pTjet30GeV.to_numpy(),
                "genWeight": events.genWeight.to_numpy(),
            }

            unique_htxs_categories = numpy.unique(base_dict["HTXS_stage1_2_cat_pTjet30GeV"])

            # Check if LHE scale and pdf weights are present
            has_lhe_scale = hasattr(events, "LHEScaleWeight")
            has_lhe_pdf = hasattr(events, "LHEPdfWeight")

            # Add LHE scale weights if present
            lhescale_dict = {}
            n_lhe_scale_weights = 0
            if has_lhe_scale:
                events["LHEScaleWeight"] = ak.to_regular(events.LHEScaleWeight)
                lhescaleweight = events.LHEScaleWeight.to_numpy()
                n_lhe_scale_weights = lhescaleweight.shape[1]
                lhescale_dict = {
                    f"LHEScaleWeight_{i}": lhescaleweight[:, i]
                    for i in range(n_lhe_scale_weights)
                }

            # Add LHE pdf weights if present
            lhepdf_dict = {}
            n_lhe_pdf_weights = 0
            if has_lhe_pdf:
                events["LHEPdfWeight"] = ak.to_regular(events.LHEPdfWeight)
                lhepdfweight = events.LHEPdfWeight.to_numpy()
                n_lhe_pdf_weights = lhepdfweight.shape[1]
                lhepdf_dict = {
                    f"LHEPdfWeight_{i}": lhepdfweight[:, i]
                    for i in range(n_lhe_pdf_weights)
                }

            accum_dict = base_dict | lhescale_dict | lhepdf_dict

            accum_df = pd.DataFrame(accum_dict, copy=False)
            accum_sums = accum_df.groupby("HTXS_stage1_2_cat_pTjet30GeV").sum().reindex(unique_htxs_categories, fill_value=0)

            custom_accumulator = {}
            custom_accumulator["sum_genw_presel_HTXS_stage1_2_cat_pTjet30GeV"] = {}
            if has_lhe_scale:
                custom_accumulator["sum_lhescalew_presel_HTXS_stage1_2_cat_pTjet30GeV"] = {}
            if has_lhe_pdf:
                custom_accumulator["sum_lhepdfw_presel_HTXS_stage1_2_cat_pTjet30GeV"] = {}

            for bin_val in accum_sums.index:
                custom_accumulator["sum_genw_presel_HTXS_stage1_2_cat_pTjet30GeV"][bin_val] = accum_sums["genWeight"][bin_val]

                if has_lhe_scale:
                    custom_accumulator["sum_lhescalew_presel_HTXS_stage1_2_cat_pTjet30GeV"][bin_val] = {
                        f"LHEScaleWeight_{i}": accum_sums[f"LHEScaleWeight_{i}"][bin_val]
                        for i in range(n_lhe_scale_weights)
                    }

                if has_lhe_pdf:
                    custom_accumulator["sum_lhepdfw_presel_HTXS_stage1_2_cat_pTjet30GeV"][bin_val] = {
                        f"LHEPdfWeight_{i}": accum_sums[f"LHEPdfWeight_{i}"][bin_val]
                        for i in range(n_lhe_pdf_weights)
                    }

            del base_dict, lhescale_dict, lhepdf_dict, accum_dict
            if has_lhe_scale:
                del lhescaleweight
            if has_lhe_pdf:
                del lhepdfweight
            del accum_df, accum_sums
        else:
            metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        events = self.apply_filters_and_triggers(events)

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            excluded_years = ["2018", "2017", "2016preVFP", "2016postVFP"]
            if self.year[dataset_name][0] not in excluded_years:
                events = remove_EcalBadCalibCrystal_events(events)

        # add zero photon mass and charge
        # TODO: remove this temporary fix when https://github.com/scikit-hep/vector/issues/498 is resolved
        events["Photon"] = self.add_zero_photon_mass_and_charge(events.Photon)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
        events["Photon"] = add_photon_SC_eta(events.Photon, events.PV)

        events["Electron", "ScEta"] = events.Electron.eta + events.Electron.deltaEtaSC

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
            events["Photon"] = events.Photon[events.Photon.vetoEELeak]
            events["Electron"] = veto_EEleak_flag(self, events.Electron)
            events["Electron"] = events.Electron[events.Electron.vetoEELeak]

        # If --Smear-sigma_m == True and no Smearing correction in .json for MC throws an error, since the pt spectrum need to be smeared in order to properly calculate the smeared sigma_m_m
        if (
            self.data_kind == "mc"
            and self.Smear_sigma_m
            and ("Smearing_Trad" not in correction_names and "Smearing_IJazZ" not in correction_names and "Smearing2G_IJazZ" not in correction_names)
        ):
            warnings.warn(
                "Smearing_Trad or Smearing_IJazZ or Smearing2G_IJazZ should be specified in the corrections field in .json in order to smear the mass!"
            )
            sys.exit(0)

        # save raw pt for scale/smearing corrections
        # These needs to be before the smearing of the mass resolution in order to have the raw pt for the function
        events["Photon"] = ak.with_field(events.Photon, events.Photon.pt, "pt_raw")
        events["Electron"] = ak.with_field(events.Electron, events.Electron.pt, "pt_raw")

        # we need the uncorrected pt for jets, photons, electrons and muons for the type-I MET correction
        # field pt_raw is already defined in jerc_jet in a different way, so the name should be avoided
        events["Photon"] = ak.with_field(events.Photon, events.Photon.pt, "pt_nano")
        events["Electron"] = ak.with_field(events.Electron, events.Electron.pt, "pt_nano")
        events["Muon"] = ak.with_field(events.Muon, events.Muon.pt, "pt_nano")
        events["Tau"] = ak.with_field(events.Tau, events.Tau.pt, "pt_nano")
        events["Jet"] = ak.with_field(events.Jet, events.Jet.pt, "pt_nano")

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

            logger.info(
                f"""
                \nApplying correction {correction_name} to dataset {dataset_name}\n
                This is only for the addition of the smearing term to the sigma_m_over_m in data\n
                """
            )
            varying_function = available_object_corrections[correction_name]
            events = varying_function(events=events, year=self.year[dataset_name][0])

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

        # Store original collections for objects which we later correct
        original_photons = events.Photon
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet
        original_electrons = events.Electron
        original_muons = events.Muon
        original_taus = events.Tau
        original_met = events.PuppiMET

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
            "Electron": original_electrons,
            "Muon": original_muons,
            "Tau": original_taus,
            "MET": original_met,
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

        # Pick the original collections after registering systematic info
        original_photons = collections["Photon"]
        original_electrons = collections["Electron"]
        original_muons = collections["Muon"]
        original_taus = collections["Tau"]
        original_met = collections["MET"]

        # Write systematic variations to dicts
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                photons_dct[f"{systematic}_{variation}"] = original_photons.systematics[systematic][variation]

        electrons_dct = {}
        electrons_dct["nominal"] = original_electrons
        logger.debug(original_electrons.systematics.fields)
        for systematic in original_electrons.systematics.fields:
            for variation in original_electrons.systematics[systematic].fields:
                electrons_dct[f"{systematic}_{variation}"] = original_electrons.systematics[systematic][variation]

        muons_dct = {}
        muons_dct["nominal"] = original_muons
        logger.debug(original_muons.systematics.fields)
        for systematic in original_muons.systematics.fields:
            for variation in original_muons.systematics[systematic].fields:
                muons_dct[f"{systematic}_{variation}"] = original_muons.systematics[systematic][variation]

        taus_dct = {}
        taus_dct["nominal"] = original_taus
        logger.debug(original_taus.systematics.fields)
        for systematic in original_taus.systematics.fields:
            for variation in original_taus.systematics[systematic].fields:
                taus_dct[f"{systematic}_{variation}"] = original_taus.systematics[systematic][variation]

        met_dct = {}
        met_dct = {"nominal": original_met}
        logger.debug(original_met.systematics.fields)
        for systematic in original_met.systematics.fields:
            for variation in original_met.systematics[systematic].fields:
                met_dct[f"{systematic}_{variation}"] = original_met.systematics[systematic][variation]

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        # object systematics dictionary
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
        variations_combined.append(original_electrons.systematics.fields)
        variations_combined.append(original_muons.systematics.fields)
        variations_combined.append(original_taus.systematics.fields)
        variations_combined.append(original_met.systematics.fields)
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
            photons, electrons, muons, taus, jets, MET = (
                photons_dct["nominal"],
                electrons_dct["nominal"],
                muons_dct["nominal"],
                taus_dct["nominal"],
                events.Jet,
                met_dct["nominal"],
            )

            if variation == "nominal":
                pass  # Do nothing since we already get the unvaried, but nominally corrected objets above
            elif variation in [*photons_dct]:  # [*dict] gets the keys of the dict since Python >= 3.5
                photons = photons_dct[variation]
                logger.info(f"Replacing nominal photons with variation {variation}.\n")
            elif variation in [*electrons_dct]:
                electrons = electrons_dct[variation]
                logger.info(f"Replacing nominal electrons with variation {variation}.\n")
            elif variation in [*muons_dct]:
                muons = muons_dct[variation]
                logger.info(f"Replacing nominal muons with variation {variation}.\n")
            elif variation in [*taus_dct]:
                taus = taus_dct[variation]
                logger.info(f"Replacing nominal taus with variation {variation}.\n")
            elif variation in [*jets_dct]:
                jets = jets_dct[variation]
                logger.info(f"Replacing nominal jets with variation {variation}.\n")
            elif variation in [*met_dct]:
                MET = met_dct[variation]
                logger.info(f"Replacing nominal MET with variation {variation}.\n")
            do_variation = variation  # We can also simplify this a bit but for now it works

            if self.chained_quantile is not None:
                photons = self.chained_quantile.apply(photons, events)
            # recompute photonid_mva on the fly
            if self.photonid_mva_EB and self.photonid_mva_EE:
                photons = self.add_photonid_mva(photons, events)

            # photon preselection
            if self.validate_with_electrons:
                photons = photon_preselection(self, photons, events, year=self.year[dataset_name][0], electron_veto=False, revert_electron_veto=True)
            else:
                photons = photon_preselection(self, photons, events, year=self.year[dataset_name][0])

            diphotons = build_diphoton_candidates(photons, self.min_pt_lead_photon)

            # Apply the fiducial cut at detector level with helper function
            diphotons = apply_fiducial_cut_det_level(self, diphotons)

            if self.data_kind == "mc":
                # Add the fiducial flags for particle level
                diphotons['fiducialClassicalFlag'] = get_fiducial_flag(events, flavour='Classical')
                diphotons['fiducialGeometricFlag'] = get_fiducial_flag(events, flavour='Geometric')

                GenPTH, GenYH, GenPhiH, _, _ = get_higgs_gen_attributes(events)

                GenPTH = ak.fill_none(GenPTH, -999.0)
                diphotons['GenPTH'] = GenPTH

                genJets = get_genJets(
                    events,
                    pt_cut=30.,
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

                gen_first_jet_eta = choose_jet(genJets.eta, 0, -999.0)
                gen_first_jet_mass = choose_jet(genJets.mass, 0, -999.0)
                gen_first_jet_phi = choose_jet(genJets.phi, 0, -999.0)

                with numpy.errstate(over='ignore', invalid='ignore'):
                    gen_first_jet_pz = GenPTJ0 * numpy.sinh(gen_first_jet_eta)
                    gen_first_jet_energy = numpy.sqrt((GenPTJ0**2 * numpy.cosh(gen_first_jet_eta)**2) + gen_first_jet_mass**2)

                    GenYJ0 = 0.5 * numpy.log((gen_first_jet_energy + gen_first_jet_pz) / (gen_first_jet_energy - gen_first_jet_pz))

                GenYJ0 = ak.fill_none(GenYJ0, -999)
                GenYJ0 = ak.where(numpy.isnan(GenYJ0), -999, GenYJ0)
                diphotons['GenYJ0'] = GenYJ0

                GenYH = ak.fill_none(GenYH, -999)
                GenYH = ak.where(numpy.isnan(GenYH), -999, GenYH)
                diphotons['GenYH'] = GenYH

                GenAbsPhiHJ0 = numpy.abs(gen_first_jet_phi - GenPhiH)

                # Set all entries above 2*pi to -999
                GenAbsPhiHJ0 = ak.where(
                    GenAbsPhiHJ0 > 2 * numpy.pi,
                    -999,
                    GenAbsPhiHJ0
                )
                GenAbsPhiHJ0_pi_array = ak.full_like(GenAbsPhiHJ0, 2 * numpy.pi)

                # Select the smallest angle
                GenAbsPhiHJ0 = ak.where(
                    GenAbsPhiHJ0 > numpy.pi,
                    GenAbsPhiHJ0_pi_array - GenAbsPhiHJ0,
                    GenAbsPhiHJ0
                )
                GenAbsPhiHJ0 = ak.fill_none(GenAbsPhiHJ0, -999.0)

                diphotons["GenDPhiHJ0"] = GenAbsPhiHJ0

                GenAbsYHJ0 = numpy.abs(GenYJ0 - GenYH)

                # Set all entries above 500 to -999
                GenAbsYHJ0 = ak.where(
                    GenAbsYHJ0 > 500,
                    -999,
                    GenAbsYHJ0
                )

                diphotons["GenDYHJ0"] = GenAbsYHJ0

            # baseline modifications to diphotons
            if self.diphoton_mva is not None:
                diphotons = self.add_diphoton_mva(diphotons, events)

            # workflow specific processing
            events, process_extra = self.process_extra(events)
            histos_etc.update(process_extra)

            # jet_variables
            jets = ak.zip(
                {
                    "pt": jets.pt,
                    "pt_nano": jets.pt_nano,
                    "eta": jets.eta,
                    "phi": jets.phi,
                    "mass": jets.mass,
                    "charge": ak.zeros_like(
                        jets.pt
                    ),  # added this because jet charge is not a property of photons in nanoAOD v11. We just need the charge to build jet collection.
                    "hFlav": jets.hadronFlavour if self.data_kind == "mc" else ak.zeros_like(jets.pt),
                    "nConstituents": jets.nConstituents,
                    "jetId": add_jetId(jets, self.nano_version, self.year[dataset_name][0], flattenUnflatten=True),  # add jet ID based on nano version
                    **(
                        {x: jets[x] for x in jets.fields if x.startswith("btag")}
                    ),
                    **(
                        {"neHEF": jets.neHEF, "neEmEF": jets.neEmEF, "chEmEF": jets.chEmEF, "muEF": jets.muEF} if self.nano_version == 12 else {}
                    ),
                    **(
                        {"neHEF": jets.neHEF, "neEmEF": jets.neEmEF, "chMultiplicity": jets.chMultiplicity, "neMultiplicity": jets.neMultiplicity, "chEmEF": jets.chEmEF, "chHEF": jets.chHEF, "muEF": jets.muEF} if self.nano_version >= 13 else {}
                    ),
                }
            )
            jets = ak.with_name(jets, "PtEtaPhiMCandidate")

            electrons = ak.zip(
                {
                    "pt": electrons.pt,
                    "pt_nano": electrons.pt_nano,
                    "eta": electrons.eta,
                    "phi": electrons.phi,
                    "mass": electrons.mass,
                    "charge": electrons.charge,
                    "cutBased": electrons.cutBased,
                    "mvaIso_WP90": electrons.mvaIso_WP90,
                    "mvaIso_WP80": electrons.mvaIso_WP80,
                    "leptonFlavour": ak.full_like(electrons.pt, 0),
                    "leptonID": electrons.mvaIso
                }
            )
            electrons = ak.with_name(electrons, "PtEtaPhiMCandidate")

            # Special cut for base workflow to replicate iso cut for electrons also for muons
            muons = muons[muons.pfRelIso03_all < 0.2]

            muons = ak.zip(
                {
                    "pt": muons.pt,
                    "pt_nano": muons.pt_nano,
                    "eta": muons.eta,
                    "phi": muons.phi,
                    "mass": muons.mass,
                    "charge": muons.charge,
                    "tightId": muons.tightId,
                    "mediumId": muons.mediumId,
                    "looseId": muons.looseId,
                    "isGlobal": muons.isGlobal,
                    "pfIsoId": muons.pfIsoId,
                    "leptonFlavour": ak.full_like(muons.pt, 1),
                    "leptonID": muons.mvaMuID
                }
            )
            muons = ak.with_name(muons, "PtEtaPhiMCandidate")

            taus = ak.zip(
                {
                    "pt": taus.pt,
                    "pt_nano": taus.pt_nano,
                    "eta": taus.eta,
                    "phi": taus.phi,
                    "mass": taus.mass,
                    "charge": taus.charge,
                    "decayMode": taus.decayMode,
                    "dz": taus.dz,
                    "idDeepTau2018v2p5VSe": taus.idDeepTau2018v2p5VSe,
                    "idDeepTau2018v2p5VSmu": taus.idDeepTau2018v2p5VSmu,
                    "idDeepTau2018v2p5VSjet": taus.idDeepTau2018v2p5VSjet,
                    "leptonFlavour": ak.full_like(taus.pt, 2),
                    "leptonID": ak.full_like(taus.pt, -999.0)  # TODO: we don't have a score, only WPs
                }
            )
            taus = ak.with_name(taus, "PtEtaPhiMCandidate")

            # Apply type-I MET correction before object selections and add MET to diphotons
            # This is *always* applied, so doesn't need to be listed in the runner
            met_corr = apply_type1_met_correction(MET, objects=(jets, photons, electrons, muons, taus), raw_pt_name="pt_nano")
            # Add MET
            diphotons["MET_pt"] = ak.fill_none(met_corr.pt, -999.0)
            diphotons["MET_phi"] = ak.fill_none(met_corr.phi, -999.0)

            # lepton cleaning
            sel_electrons = electrons[select_electrons(self, electrons, diphotons)]
            sel_muons = muons[select_muons(self, muons, diphotons)]
            sel_taus = taus[select_taus(self, taus, diphotons)]

            # Build pt-ordered lepton collection
            sel_leptons = ak.concatenate([sel_electrons, sel_muons, sel_taus], axis=1)
            sel_leptons = sel_leptons[ak.argsort(sel_leptons.pt, ascending=False)]

            # Add lepton variables to diphotons
            lepton_indices = [0, 1]
            choose_lepton = choose_jet  # TODO this should be renamed to e.g. choose_object
            diphotons["n_leptons"] = ak.num(sel_leptons)
            diphotons["n_electrons"] = ak.num(sel_electrons)
            diphotons["n_muons"] = ak.num(sel_muons)
            diphotons["n_taus"] = ak.num(sel_taus)
            for i in lepton_indices:
                # Add 'merged' leptons
                diphotons[f"Lep{i}_pt"] = choose_lepton(sel_leptons.pt, i, -999.0)
                diphotons[f"Lep{i}_eta"] = choose_lepton(sel_leptons.eta, i, -999.0)
                diphotons[f"Lep{i}_phi"] = choose_lepton(sel_leptons.phi, i, -999.0)
                diphotons[f"Lep{i}_mass"] = choose_lepton(sel_leptons.mass, i, -999.0)
                diphotons[f"Lep{i}_charge"] = choose_lepton(sel_leptons.charge, i, -999.0)
                diphotons[f"Lep{i}_leptonFlavour"] = choose_lepton(sel_leptons.leptonFlavour, i, -999.0)
                diphotons[f"Lep{i}_id"] = choose_lepton(sel_leptons.leptonID, i, -999.0)

                # Add individual leptons
                diphotons[f"Ele{i}_pt"] = choose_lepton(sel_electrons.pt, i, -999.0)
                diphotons[f"Ele{i}_eta"] = choose_lepton(sel_electrons.eta, i, -999.0)
                diphotons[f"Ele{i}_phi"] = choose_lepton(sel_electrons.phi, i, -999.0)
                diphotons[f"Ele{i}_mass"] = choose_lepton(sel_electrons.mass, i, -999.0)
                diphotons[f"Ele{i}_charge"] = choose_lepton(sel_electrons.charge, i, -999.0)
                diphotons[f"Ele{i}_leptonFlavour"] = choose_lepton(sel_electrons.leptonFlavour, i, -999.0)
                diphotons[f"Ele{i}_id"] = choose_lepton(sel_electrons.leptonID, i, -999.0)
                diphotons[f"Muo{i}_pt"] = choose_lepton(sel_muons.pt, i, -999.0)
                diphotons[f"Muo{i}_eta"] = choose_lepton(sel_muons.eta, i, -999.0)
                diphotons[f"Muo{i}_phi"] = choose_lepton(sel_muons.phi, i, -999.0)
                diphotons[f"Muo{i}_mass"] = choose_lepton(sel_muons.mass, i, -999.0)
                diphotons[f"Muo{i}_charge"] = choose_lepton(sel_muons.charge, i, -999.0)
                diphotons[f"Muo{i}_leptonFlavour"] = choose_lepton(sel_muons.leptonFlavour, i, -999.0)
                diphotons[f"Muo{i}_id"] = choose_lepton(sel_muons.leptonID, i, -999.0)
                diphotons[f"Tau{i}_pt"] = choose_lepton(sel_taus.pt, i, -999.0)
                diphotons[f"Tau{i}_eta"] = choose_lepton(sel_taus.eta, i, -999.0)
                diphotons[f"Tau{i}_phi"] = choose_lepton(sel_taus.phi, i, -999.0)
                diphotons[f"Tau{i}_mass"] = choose_lepton(sel_taus.mass, i, -999.0)
                diphotons[f"Tau{i}_charge"] = choose_lepton(sel_taus.charge, i, -999.0)
                diphotons[f"Tau{i}_leptonFlavour"] = choose_lepton(sel_taus.leptonFlavour, i, -999.0)
                diphotons[f"Tau{i}_id"] = choose_lepton(sel_taus.leptonID, i, -999.0)

            # jet selection and pt ordering
            # follows https://indico.cern.ch/event/1624984/contributions/6896120/
            if self.year[dataset_name][0] in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]:
                self.jet_pt_thresholds = [20, 50, 50]
                jet_selection_func = select_jets_eta_dependent
            elif self.year[dataset_name][0] in ["2024"]:
                self.jet_pt_thresholds = [20, 50, 20]
                jet_selection_func = select_jets_eta_dependent
            else:
                jet_selection_func = select_jets
            jets = jets[
                jet_selection_func(self, jets, diphotons, sel_muons, sel_electrons, sel_taus)
            ]
            jets = jets[ak.argsort(jets.pt, ascending=False)]

            # Btagged jets
            btagMVA_selection = {
                "deepJet": {"btagDeepFlavB": jets.btagDeepFlavB},  # Always available
                "particleNet": {"btagPNetB": jets.btagPNetB} if self.nano_version >= 12 else {},
                "robustParticleTransformer": {"btagRobustParTAK4B": jets.btagRobustParTAK4B} if self.nano_version in [12, 13] else {},
                "btagUParTAK4B": {"btagUParTAK4B": jets.btagUParTAK4B} if self.nano_version > 13 else {},
            }

            base_pt_eta_cut = (jets.pt > 30) & (abs(jets.eta) < 2.5)
            for bjet_mva in self.bjet_mva:
                btag_mva_column = list(btagMVA_selection[bjet_mva].keys())[0]
                for bjet_wp in self.bjet_wp:
                    btag_WP = getBTagMVACut(mva_name=bjet_mva,
                                            mva_wp=bjet_wp,
                                            year=self.year[dataset_name][0])
                    bJetCondition = base_pt_eta_cut & (jets[btag_mva_column] >= btag_WP)
                    jets[f"{bjet_mva}_is{bjet_wp}"] = bJetCondition

            # adding selected jets to events to be used in ctagging SF calculation
            events["sel_jets"] = jets
            n_jets = ak.num(jets)
            Njets2p5 = ak.num(jets[(jets.pt > 30) & (numpy.abs(jets.eta) < 2.5)])

            # Add jets
            jet_indices = [0, 1, 2, 3, 4, 5]
            jet_collection = {}
            for i in jet_indices:
                jet_collection[f"J{i}_pt"] = choose_jet(jets.pt, i, -999.0)
                jet_collection[f"J{i}_eta"] = choose_jet(jets.eta, i, -999.0)
                jet_collection[f"J{i}_phi"] = choose_jet(jets.phi, i, -999.0)
                jet_collection[f"J{i}_mass"] = choose_jet(jets.mass, i, -999.0)
                jet_collection[f"J{i}_charge"] = choose_jet(jets.charge, i, -999.0)
                jet_collection[f"J{i}_nConstituents"] = choose_jet(jets.nConstituents, i, -999.0)
                for btag_key in jets.fields:
                    if btag_key.startswith("btag"):
                        jet_collection[f"J{i}_{btag_key}"] = choose_jet(jets[btag_key], i, -999.0)
                for bjet_mva in self.bjet_mva:
                    for bjet_wp in self.bjet_wp:
                        jet_collection[f"J{i}_{bjet_mva}_is{bjet_wp}"] = choose_jet(jets[f"{bjet_mva}_is{bjet_wp}"], i, -999)

            # Add Ht (scalar sum of jet Et)
            jet_Et = numpy.sqrt(jets.pt**2 + jets.mass**2)
            jet_Ht = ak.sum(jet_Et, axis=1)
            jet_Ht = ak.fill_none(jet_Ht, -999.0)
            diphotons["HT"] = jet_Ht

            # Add jet variables to diphotons
            for i in jet_indices:
                diphotons[f"J{i}_pt"] = jet_collection[f"J{i}_pt"]
                diphotons[f"J{i}_eta"] = jet_collection[f"J{i}_eta"]
                diphotons[f"J{i}_phi"] = jet_collection[f"J{i}_phi"]
                diphotons[f"J{i}_mass"] = jet_collection[f"J{i}_mass"]
                diphotons[f"J{i}_charge"] = jet_collection[f"J{i}_charge"]
                diphotons[f"J{i}_nConstituents"] = jet_collection[f"J{i}_nConstituents"]
                for btag_key in jet_collection.keys():
                    if btag_key.startswith(f"J{i}_btag"):
                        diphotons[btag_key] = jet_collection[btag_key]
                for bjet_mva in self.bjet_mva:
                    for bjet_wp in self.bjet_wp:
                        diphotons[f"J{i}_{bjet_mva}_is{bjet_wp}"] = jet_collection[f"J{i}_{bjet_mva}_is{bjet_wp}"]
            diphotons["n_jets"] = n_jets
            diphotons["NJ"] = Njets2p5
            if self.nano_version > 13:
                diphotons["n_bjets"] = ak.sum(jets["btagUParTAK4B_isT"], axis=1)
            else:
                diphotons["n_bjets"] = ak.sum(jets["particleNet_isT"], axis=1)

            # Extract forwardmost selected jet
            eta_sort_idxs = ak.argsort(numpy.abs(jets.eta), ascending=False)
            etasorted_jets = jets[eta_sort_idxs]
            diphotons["JFWD_pt"] = choose_jet(etasorted_jets.pt, 0, -999.0)
            diphotons["JFWD_eta"] = choose_jet(etasorted_jets.eta, 0, -999.0)
            diphotons["JFWD_phi"] = choose_jet(etasorted_jets.phi, 0, -999.0)

            with numpy.errstate(over='ignore', invalid='ignore'):
                first_jet_pz = jet_collection["J0_pt"] * numpy.sinh(jet_collection["J0_eta"])
                first_jet_energy = numpy.sqrt((jet_collection["J0_pt"]**2 * numpy.cosh(jet_collection["J0_eta"])**2) + jet_collection["J0_mass"]**2)

                first_jet_y = 0.5 * numpy.log((first_jet_energy + first_jet_pz) / (first_jet_energy - first_jet_pz))
                first_jet_y = ak.fill_none(first_jet_y, -999)
                first_jet_y = ak.where(numpy.isnan(first_jet_y), -999, first_jet_y)
            diphotons["YJ0"] = first_jet_y

            AbsPhiHJ0 = numpy.abs(jet_collection["J0_phi"] - diphotons["phi"])

            AbsPhiHJ0_pi_array = ak.full_like(AbsPhiHJ0, 2 * numpy.pi)

            # Select the smallest angle
            AbsPhiHJ0 = ak.where(
                AbsPhiHJ0 > numpy.pi,
                AbsPhiHJ0_pi_array - AbsPhiHJ0,
                AbsPhiHJ0
            )
            AbsPhiHJ0 = ak.where(
                AbsPhiHJ0 > 2 * numpy.pi,
                -999,
                ak.where(
                    AbsPhiHJ0 < 0,
                    -999,
                    AbsPhiHJ0
                )
            )
            diphotons["DPhiHJ0"] = AbsPhiHJ0

            AbsYHJ0 = numpy.abs(first_jet_y - diphotons["rapidity"])

            # Set all entries above 500 to -999
            AbsYHJ0 = ak.where(
                AbsYHJ0 > 500,
                -999,
                AbsYHJ0
            )

            diphotons["DYHJ0"] = AbsYHJ0

            # run taggers on the events list with added diphotons
            # the shape here is ensured to be broadcastable
            for tagger in self.taggers:
                (
                    diphotons["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    events, diphotons
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

            diphotons = ak.firsts(diphotons)
            original_diphotons = copy.copy(diphotons)
            # annotate diphotons with event information
            diphotons["event"] = events.event
            diphotons["lumi"] = events.luminosityBlock
            diphotons["run"] = events.run
            # nPV just for validation of pileup reweighting
            diphotons["nPV"] = events.PV.npvs
            diphotons["PVScore"] = events.PV.score
            diphotons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # Beamspot variables
            diphotons["BeamSpot_sigmaZ"] = events.BeamSpot.sigmaZ
            diphotons["BeamSpot_sigmaZError"] = events.BeamSpot.sigmaZError
            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = events.genWeight
                associated_decay_labels, _ = label_associated_decay(
                    events,
                    self.associated_decay_config,
                    default_label="unclassified",
                    raise_on_overlap=True,
                    particle_type_map=self.associated_decay_particle_map,
                )
                diphotons["AssociatedDecay"] = associated_decay_labels
                diphotons["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                diphotons["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                diphotons["HTXS_stage_0"] = events.HTXS.stage_0
                diphotons["HTXS_stage1_2_cat_pTjet30GeV"] = events.HTXS.stage1_2_cat_pTjet30GeV
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                diphotons["dZ"] = ak.zeros_like(events.PV.z)

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
                event_weights = Weights(size=len(events[selection_mask]), storeIndividual=True)
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
                            events=events[selection_mask],
                            photons=original_diphotons[selection_mask],
                            electrons=sel_electrons[selection_mask],
                            muons=sel_muons[selection_mask],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if has_lhe_scale:
                                    diphotons["nweight_LHEScale"] = ak.num(
                                        events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    diphotons[
                                        "weight_LHEScale"
                                    ] = events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if has_lhe_pdf:
                                    # two AlphaS weights are removed
                                    diphotons["nweight_LHEPdf"] = (
                                        ak.num(
                                            events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    diphotons[
                                        "weight_LHEPdf"
                                    ] = events.LHEPdfWeight[selection_mask][
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
                                    events=events[selection_mask],
                                    photons=original_diphotons[selection_mask],
                                    electrons=sel_electrons[selection_mask],
                                    muons=sel_muons[selection_mask],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                )

                diphotons["weight"] = event_weights.weight()
                diphotons["weight_central"] = event_weights.weight() / events["genWeight"][selection_mask]

                metadata["sum_weight_central"] = str(
                    ak.sum(event_weights.weight())
                )
                metadata["sum_weight_central_wo_bTagSF"] = str(
                    ak.sum(event_weights.weight() / (event_weights.partial_weight(include=["bTagSF"])))
                )

                base_dict = {
                    "HTXS_stage1_2_cat_pTjet30GeV": events.HTXS.stage1_2_cat_pTjet30GeV[selection_mask].to_numpy(),
                    "genWeight": events.genWeight[selection_mask].to_numpy(),
                }

                # Add LHE scale weights if present
                lhescale_dict = {}
                if has_lhe_scale:
                    lhescaleweight = events.LHEScaleWeight[selection_mask].to_numpy()
                    lhescale_dict = {
                        f"LHEScaleWeight_{i}": lhescaleweight[:, i]
                        for i in range(n_lhe_scale_weights)
                    }

                # Add LHE pdf weights if present
                lhepdf_dict = {}
                if has_lhe_pdf:
                    lhepdfweight = events.LHEPdfWeight[selection_mask].to_numpy()
                    lhepdf_dict = {
                        f"LHEPdfWeight_{i}": lhepdfweight[:, i]
                        for i in range(n_lhe_pdf_weights)
                    }

                accum_dict = base_dict | lhescale_dict | lhepdf_dict

                accum_df = pd.DataFrame(accum_dict, copy=False)
                accum_sums = accum_df.groupby("HTXS_stage1_2_cat_pTjet30GeV").sum().reindex(unique_htxs_categories, fill_value=0)

                custom_accumulator["sum_genw_postsel_HTXS_stage1_2_cat_pTjet30GeV"] = {}
                if has_lhe_scale:
                    custom_accumulator["sum_lhescalew_postsel_HTXS_stage1_2_cat_pTjet30GeV"] = {}
                if has_lhe_pdf:
                    custom_accumulator["sum_lhepdfw_postsel_HTXS_stage1_2_cat_pTjet30GeV"] = {}

                for bin_val in accum_sums.index:
                    custom_accumulator["sum_genw_postsel_HTXS_stage1_2_cat_pTjet30GeV"][bin_val] = accum_sums["genWeight"][bin_val]

                    if has_lhe_scale:
                        custom_accumulator["sum_lhescalew_postsel_HTXS_stage1_2_cat_pTjet30GeV"][bin_val] = {
                            f"LHEScaleWeight_{i}": accum_sums[f"LHEScaleWeight_{i}"][bin_val]
                            for i in range(n_lhe_scale_weights)
                        }

                    if has_lhe_pdf:
                        custom_accumulator["sum_lhepdfw_postsel_HTXS_stage1_2_cat_pTjet30GeV"][bin_val] = {
                            f"LHEPdfWeight_{i}": accum_sums[f"LHEPdfWeight_{i}"][bin_val]
                            for i in range(n_lhe_pdf_weights)
                        }

                del base_dict, lhescale_dict, lhepdf_dict, accum_df, accum_sums
                if has_lhe_scale:
                    del lhescaleweight
                if has_lhe_pdf:
                    del lhepdfweight

                metadata["custom_accumulator"] = json.dumps(
                    custom_accumulator,
                    default=lambda x: x.item() if isinstance(x, numpy.number) else x
                ).encode("utf-8")

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

            # Compute and store the different variations of sigma_m_over_m
            diphotons = compute_sigma_m(diphotons, processor='base', flow_corrections=self.doFlow_corrections, smear=self.Smear_sigma_m, IsData=(self.data_kind == "data"))

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
                    elif self.data_kind == "mc" and ("Smearing_IJazZ" in correction_names or "Smearing2G_IJazZ" in correction_names):
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

                    # Remove fixedGridRhoAll from photons to avoid having event-level info per photon
                    akarr = akarr[
                        [
                            field
                            for field in akarr.fields
                            if "lead_fixedGridRhoAll" not in field
                        ]
                    ]

                fname = apply_naming_convention(self, events)
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

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass
