from higgs_dna.workflows.skeleton import HggSkeletonProcessor

from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EELeak_region import veto_EEleak_flag
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.selections.photon_selections import photon_preselection
from higgs_dna.tools.jetID import add_jetId
from higgs_dna.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from higgs_dna.selections.lepton_selections import select_electrons, select_muons
from higgs_dna.selections.jet_selections import select_jets, jetvetomap, getBTagMVACut
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.utils.dumping_utils import apply_naming_convention, diphoton_ak_array, dump_ak_array, diphoton_list_to_pandas, dump_pandas, get_obj_syst_dict
from higgs_dna.utils.misc_utils import choose_jet
from higgs_dna.tools.flow_corrections import apply_flow_corrections_to_photons
from higgs_dna.tools.gen_helpers import get_fiducial_flag, get_genJets
from higgs_dna.tools.Frixione_helper import attach_frixione_isolation_flag_to_diphotons, attach_geninfo_to_photons

from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections
from higgs_dna.systematics import apply_systematic_variations_object_level
from higgs_dna.systematics.MET_systematics import apply_type1_met_correction

import warnings
from typing import Any, Dict, List, Optional
import awkward as ak
import numpy as np
import vector
from coffea.analysis_tools import Weights
import copy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()


class TopProcessor(HggSkeletonProcessor):  # type: ignore

    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        apply_trigger: bool = False,
        nano_version: int = None,
        bTagEffFileName: Optional[str] = None,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        trigger_group=".*DoubleEG.*",
        analysis="mainAnalysis",
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
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
        self.nano_version = nano_version

        self.el_id_wp = "WP90"
        self.name_convention = "DAS"
        self.min_mvaid = -0.7
        if self.nano_version == 15:
            self.bjet_mva = "btagUParTAK4B"
        else:
            self.bjet_mva = "robustParticleTransformer"

    def process_extra(self, events: ak.Array) -> ak.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass

    def process(self, events: ak.Array) -> Dict[Any, Any]:

        print("\n \t INFO: running top processor. \n")
        dataset_name = events.metadata["dataset"]
        add_frixione_info = False
        if dataset_name.startswith(("TTG", "TTto", "TGQB", "TJGG", "TBbarQ", "TbarBQ")):
            add_frixione_info = True
        if "sideband" in dataset_name.lower():
            self.min_mvaid = -0.95

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
                np.sum(events.genWeight.to_numpy())
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
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(np.sum(events.genWeight.to_numpy()))
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

        if self.validate_with_electrons:
            # select photons with an associated electron and a pixel seed
            photons_mask = (events.Photon.electronIdx != -1) & (events.Photon.pixelSeed)
            events["Photon"] = events.Photon[photons_mask]
            events = events[ak.num(events.Photon) >= 2]

        # Need to add ScEta variables to electrons for scale and smearing corrections
        electrons = events.Electron
        electrons["ScEta"] = electrons.eta + electrons.deltaEtaSC
        electrons["isScEtaEB"] = np.abs(electrons.ScEta) < 1.4442
        electrons["isScEtaEE"] = np.abs(electrons.ScEta) > 1.566
        events["Electron"] = electrons

        # add veto EE leak branch for photons, could also be used for electrons
        if self.year[dataset_name][0] == "2022postEE":
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

        # save raw pt for scale/smearing corrections
        events["Photon"] = ak.with_field(events.Photon, events.Photon.pt, "pt_raw")
        events["Electron"] = ak.with_field(events.Electron, events.Electron.pt, "pt_raw")

        # we need the uncorrected pt for jets, photons, electrons and muons for the type-I MET correction
        # field pt_raw is already defined in jerc_jet in a different way, so the name should be avoided
        events["Photon"] = ak.with_field(events.Photon, events.Photon.pt, "pt_nano")
        events["Electron"] = ak.with_field(events.Electron, events.Electron.pt, "pt_nano")
        events["Muon"] = ak.with_field(events.Muon, events.Muon.pt, "pt_nano")
        events["Jet"] = ak.with_field(events.Jet, events.Jet.pt, "pt_nano")

        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"Applying correction {correction_name} to dataset {dataset_name}"
                )
                varying_function = available_object_corrections[correction_name]
                events = varying_function(events=events, year=self.year[dataset_name][0])
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        original_photons = events.Photon
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet
        original_electrons = events.Electron
        original_muons = events.Muon
        original_met = ak.zip(
            {"pt": events.PuppiMET.pt, "phi": events.PuppiMET.phi},
            with_name="MissingET",
        )

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

        original_photons = collections["Photon"]
        original_electrons = collections["Electron"]
        original_muons = collections["Muon"]
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

        met_dct = {"nominal": original_met}
        for syst in original_met.systematics.fields:
            for var in original_met.systematics[syst].fields:
                met_dct[f"{syst}_{var}"] = original_met.systematics[syst][var]

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
        variations_combined.append(original_electrons.systematics.fields)
        variations_combined.append(original_muons.systematics.fields)
        variations_combined.append(original_met.systematics.fields)
        # NOTE: jet jerc systematics are not added with add_systematics
        variations_combined.append(jerc_syst_list)
        variations_flattened = sum(variations_combined, [])
        # Attach _down and _up
        variations = [item + suffix for item in variations_flattened for suffix in ['_down', '_up']]
        # Add nominal to the list
        variations.append('nominal')

        logger.debug(f"[systematics variations] {variations}")

        for variation in variations:
            logger.info(f"Processing {variation} samples.\n")
            photons, electrons, muons, jets, MET = photons_dct["nominal"], electrons_dct["nominal"], muons_dct["nominal"], events.Jet, met_dct["nominal"]
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

            if (self.data_kind == "mc" and add_frixione_info == True):
                photons = attach_geninfo_to_photons(photons)

            diphotons = build_diphoton_candidates(photons, self.min_pt_lead_photon)

            # Apply the fiducial cut at detector level with helper function
            diphotons = apply_fiducial_cut_det_level(self, diphotons)

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
                    "charge": ak.zeros_like(jets.pt),
                    "hFlav": jets.hadronFlavour if self.data_kind == "mc" else ak.zeros_like(jets.pt),
                    "btagPNetB": jets.btagPNetB,
                    "btagPNetCvB": jets.btagPNetCvB,
                    "btagPNetCvL": jets.btagPNetCvL,
                    "btagPNetQvG": jets.btagPNetQvG,
                    "btagPNetTauVJet": jets.btagPNetTauVJet,

                    **({
                        "btagUParTAK4B": jets.btagUParTAK4B,
                        "btagUParTAK4CvB": jets.btagUParTAK4CvB,
                        "btagUParTAK4QvG": jets.btagUParTAK4QvG,
                        "btagUParTAK4TauVJet": jets.btagUParTAK4TauVJet,
                    } if self.nano_version == 15 else {
                        "btagRobustParTAK4B": jets.btagRobustParTAK4B,
                        "btagRobustParTAK4CvB": jets.btagRobustParTAK4CvB,
                        "btagRobustParTAK4CvL": jets.btagRobustParTAK4CvL,
                        "btagRobustParTAK4QG": jets.btagRobustParTAK4QG,
                    }),

                    "jetId": add_jetId(jets, self.nano_version, self.year[dataset_name][0], flattenUnflatten=True),  # add jet ID based on nano version
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
                    "mvaIso_WP90": electrons.mvaIso_WP90,
                    "mvaIso_WP80": electrons.mvaIso_WP80,
                    **({"mvaTTH": electrons.promptMVA} if self.nano_version == 15 else {"mvaTTH": electrons.mvaTTH}),
                    "genPartFlav": electrons.genPartFlav if self.data_kind == "mc" else np.full_like(electrons.pt, -999),
                    "pfRelIso03_all": electrons.pfRelIso03_all,
                    "pfRelIso03_chg": electrons.pfRelIso03_chg,
                }
            )
            electrons = ak.with_name(electrons, "PtEtaPhiMCandidate")

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
                    **({"mvaTTH": muons.promptMVA} if self.nano_version == 15 else {"mvaTTH": muons.mvaTTH}),
                    "genPartFlav": muons.genPartFlav if self.data_kind == "mc" else np.full_like(muons.pt, -999),
                    "pfRelIso03_all": muons.pfRelIso03_all,
                    "pfRelIso03_chg": muons.pfRelIso03_chg,
                }
            )
            muons = ak.with_name(muons, "PtEtaPhiMCandidate")

            # apply type-I MET correction before selecting (i.e. removing possibly corrected) jets and leptons
            met_corr = apply_type1_met_correction(MET, objects=(jets, photons, electrons, muons), raw_pt_name="pt_nano")
            diphotons["met_pt"] = met_corr.pt
            diphotons["met_phi"] = met_corr.phi
            diphotons["met_significance"] = events.PuppiMET.significance if self.nano_version >= 15 else events.MET.significance

            # lepton cleaning
            electrons = electrons[select_electrons(self, electrons, diphotons)]
            muons = muons[select_muons(self, muons, diphotons)]
            jets = jets[select_jets(self, jets, diphotons, muons, electrons)]

            # remove "jet horns". Although there is no final recipe, applying pT > 50 GeV for jets with abs(eta) in (2.5, 3) seems to help
            # See https://gitlab.cern.ch/cms-jetmet/coordination/coordination/-/issues/113
            excluded_years = ["2018", "2017", "2016preVFP", "2016postVFP"]
            if self.year[dataset_name][0] not in excluded_years:
                jets = jets[
                    ~((jets.pt < 50) & (np.abs(jets.eta) > 2.5) & (np.abs(jets.eta) < 3))
                ]
            # ordering in pt since corrections may have changed the order
            electrons = electrons[ak.argsort(electrons.pt, ascending=False)]
            muons = muons[ak.argsort(muons.pt, ascending=False)]
            jets = jets[ak.argsort(jets.pt, ascending=False)]

            # adding selected jets, electrons and muons of the specific variation to events to be used in SF calculations
            events["sel_jets"] = jets
            events["sel_muons"] = muons
            events["sel_electrons"] = electrons

            # Standard jet and bjet counting (pt > 20 GeV)
            n_jets = ak.num(jets)
            n_jets_forward = ak.num(jets[np.abs(jets.eta) > 2.5])
            n_jets_central = ak.num(jets[np.abs(jets.eta) <= 2.5])
            btag_score = jets.btagUParTAK4B if self.nano_version == 15 else jets.btagRobustParTAK4B
            n_bjets_loose = ak.num(jets[btag_score > getBTagMVACut(mva_name=self.bjet_mva, mva_wp='L', year=self.year[dataset_name][0])])
            n_bjets_medium = ak.num(jets[btag_score > getBTagMVACut(mva_name=self.bjet_mva, mva_wp='M', year=self.year[dataset_name][0])])
            n_bjets_tight = ak.num(jets[btag_score > getBTagMVACut(mva_name=self.bjet_mva, mva_wp='T', year=self.year[dataset_name][0])])

            # Jet and bjet counting with pt > 25 GeV
            jets_25 = jets[jets.pt > 25]
            n_jets_25 = ak.num(jets_25)
            n_jets_forward_25 = ak.num(jets_25[np.abs(jets_25.eta) > 2.5])
            n_jets_central_25 = ak.num(jets_25[np.abs(jets_25.eta) <= 2.5])
            btag_score_25 = jets_25.btagUParTAK4B if self.nano_version == 15 else jets_25.btagRobustParTAK4B
            n_bjets_loose_25 = ak.num(jets_25[btag_score_25 > getBTagMVACut(mva_name=self.bjet_mva, mva_wp='L', year=self.year[dataset_name][0])])
            n_bjets_medium_25 = ak.num(jets_25[btag_score_25 > getBTagMVACut(mva_name=self.bjet_mva, mva_wp='M', year=self.year[dataset_name][0])])
            n_bjets_tight_25 = ak.num(jets_25[btag_score_25 > getBTagMVACut(mva_name=self.bjet_mva, mva_wp='T', year=self.year[dataset_name][0])])

            # Store all jet and bjet counting variables in diphotons
            diphotons["n_jets"] = n_jets
            diphotons["n_jets_forward"] = n_jets_forward
            diphotons["n_jets_central"] = n_jets_central
            diphotons["n_bjets_loose"] = n_bjets_loose
            diphotons["n_bjets_medium"] = n_bjets_medium
            diphotons["n_bjets_tight"] = n_bjets_tight
            diphotons["JetHT"] = ak.sum(jets.pt, axis=1)

            diphotons["n_jets_25"] = n_jets_25
            diphotons["n_jets_forward_25"] = n_jets_forward_25
            diphotons["n_jets_central_25"] = n_jets_central_25
            diphotons["n_bjets_loose_25"] = n_bjets_loose_25
            diphotons["n_bjets_medium_25"] = n_bjets_medium_25
            diphotons["n_bjets_tight_25"] = n_bjets_tight_25
            diphotons["JetHT_25"] = ak.sum(jets_25.pt, axis=1)

            max_btag_score = ak.max(btag_score, axis=1)
            diphotons["max_btag_score"] = ak.fill_none(max_btag_score, -999.0)
            second_btag_score = ak.where(btag_score == max_btag_score[:, None], -999.0, btag_score)
            diphotons["secondmax_bTag_score"] = ak.fill_none(ak.max(second_btag_score, axis=1), -999.0)

            # For jets with pt > 25 GeV
            max_btag_score_25 = ak.max(btag_score_25, axis=1)
            diphotons["max_btag_score_25"] = ak.fill_none(max_btag_score_25, -999.0)
            second_btag_score_25 = ak.where(btag_score_25 == max_btag_score_25[:, None], -999.0, btag_score_25)
            diphotons["secondmax_bTag_score_25"] = ak.fill_none(ak.max(second_btag_score_25, axis=1), -999.0)

            num_jets = 8
            jet_properties = [
                "pt", "eta", "phi", "mass", "charge", "btagPNetB", "btagPNetCvB", "btagPNetCvL",
                "btagPNetQvG", "btagPNetTauVJet",
                *(
                    [
                        "btagUParTAK4B","btagUParTAK4CvB","btagUParTAK4QvG","btagUParTAK4TauVJet"
                    ] if self.nano_version == 15 else [
                        "btagRobustParTAK4B", "btagRobustParTAK4CvB","btagRobustParTAK4CvL",
                        "btagRobustParTAK4QG"]
                )
            ]
            for i in range(num_jets):
                for prop in jet_properties:
                    key = f"jet{i + 1}_{prop}"
                    value = choose_jet(getattr(jets, prop), i, -999.0)
                    diphotons[key] = value

            # Adding a 'generation' field to electrons and muons
            electrons['generation'] = ak.ones_like(electrons.pt)
            electrons['ElectronMvaIso_WP80_MuonTightID'] = ak.ones_like(electrons.mvaIso_WP80)
            muons['generation'] = 2 * ak.ones_like(muons.pt)
            muons['ElectronMvaIso_WP80_MuonTightID'] = ak.ones_like(muons.tightId)

            # Combine electrons and muons into a single leptons collection
            leptons = ak.concatenate([electrons, muons], axis=1)
            leptons = ak.with_name(leptons, "PtEtaPhiMCandidate")
            leptons = leptons[ak.argsort(leptons.pt, ascending=False)]
            n_leptons = ak.num(leptons)
            leptons_tight = ak.concatenate([electrons[electrons.mvaIso_WP80], muons[muons.tightId]], axis=1)
            n_leptons_tight = ak.num(leptons_tight)

            diphotons["n_leptons_tight"] = n_leptons_tight
            diphotons["n_leptons"] = n_leptons

            # Annotate diphotons with selected leptons properties
            lepton_properties = ["pt", "eta", "phi", "mass", "charge", "generation", "ElectronMvaIso_WP80_MuonTightID", "mvaTTH", "genPartFlav", "pfRelIso03_all", "pfRelIso03_chg"]
            num_leptons = 2  # Number of leptons to select
            for i in range(num_leptons):
                for prop in lepton_properties:
                    key = f"lepton{i + 1}_{prop}"
                    # Retrieve the value using the choose_jet function (which can be used for leptons as well)
                    value = choose_jet(getattr(leptons, prop), i, -999.0)
                    # Store the value in the diphotons dictionary
                    diphotons[key] = value

            diphotons = ak.firsts(diphotons)
            if (self.data_kind == "mc" and add_frixione_info == True):
                diphotons = attach_frixione_isolation_flag_to_diphotons(events["GenPart"], diphotons, frix_cones=[0.05, 0.4])

            original_diphotons = copy.copy(diphotons)
            # annotate diphotons with event information
            diphotons["event"] = events.event
            diphotons["lumi"] = events.luminosityBlock
            diphotons["run"] = events.run
            # nPV just for validation of pileup reweighting
            diphotons["nPV"] = events.PV.npvs
            diphotons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = events.genWeight
                diphotons["dZ"] = events.GenVtx.z - events.PV.z
                diphotons["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = events.HTXS.njets30
                diphotons["HTXS_stage_0"] = events.HTXS.stage_0
            else:
                diphotons["dZ"] = ak.zeros_like(events.PV.z)

            # drop events without a preselected diphoton candidate
            selection_mask = ~ak.is_none(diphotons)
            diphotons = diphotons[selection_mask]

            # return if there is no surviving events
            if len(diphotons) == 0:
                logger.info("No surviving events in this run!")
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(events[selection_mask]),storeIndividual=True)
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
                            # adding muons and electrons because I don't want to introduce a naming obligation like e.g. "sel_muons" in the syst functions
                            muons=events["sel_muons"][selection_mask],
                            electrons=events["sel_electrons"][selection_mask],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )
                diphotons["bTagWeight"] = event_weights.partial_weight(include=["bTagSF"])

                # for fiducial phase space definitions
                diphotons['fiducialClassicalFlag'] = get_fiducial_flag(events[selection_mask], flavour='Classical')
                diphotons['fiducialGeometricFlag'] = get_fiducial_flag(events[selection_mask], flavour='Geometric')
                genJets = get_genJets(
                    events[selection_mask],
                    pt_cut=self.jet_pt_threshold,
                    eta_cut=2.5,
                    jet_pho_min_dr=self.jet_pho_min_dr,
                    jet_ele_min_dr=self.jet_ele_min_dr,
                    jet_muo_min_dr=self.jet_muo_min_dr,
                    electron_pt_threshold=self.electron_pt_threshold,
                    electron_max_eta=self.electron_max_eta,
                    muon_pt_threshold=self.muon_pt_threshold,
                    muon_max_eta=self.muon_max_eta,
                )
                num_bjets = ak.sum(
                    (genJets.hadronFlavour == 5), axis=-1
                )
                diphotons["GenNBJet"] = num_bjets

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(events, "LHEScaleWeight"):
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
                                if hasattr(events, "LHEPdfWeight"):
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
                                    # adding muons and electrons because I don't want to introduce a naming obligation like e.g. "sel_muons" in the syst functions
                                    muons=events["sel_muons"][selection_mask],
                                    electrons=events["sel_electrons"][selection_mask],
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

                if ('LHEReweightingWeight' in events.fields):
                    if ak.num(events.LHEReweightingWeight)[0] > 0:
                        diphotons["LHEReweightingWeight"] = events.LHEReweightingWeight[selection_mask]
                        diphotons["LHEWeight"] = events.LHEWeight[selection_mask]

            # Add weight variables (=1) for data for consistent datasets
            else:
                diphotons["weight_central"] = ak.ones_like(
                    diphotons["event"]
                )
                diphotons["weight"] = ak.ones_like(diphotons["event"])

            # save some disk space, as this analysis targets top quark decays, we can remove events without any jet
            diphotons = diphotons[diphotons.n_jets > 0]
            if not self.validate_with_electrons:
                # select events within standard HGG mass window only, after all corrections & systematics were applied
                diphotons = diphotons[(diphotons.mass > 100) & (diphotons.mass < 180)]
            if "sideband" in dataset_name.lower():
                # keep only events in the photon ID sideband: one photon passing, one failing
                photon_IDs = ak.concatenate(
                    [diphotons.pho_lead.mvaID[:, None], diphotons.pho_sublead.mvaID[:, None]],
                    axis=1,
                )
                diphotons = diphotons[(ak.max(photon_IDs, axis=1) > -0.7) & (ak.min(photon_IDs, axis=1) < -0.7)]

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
