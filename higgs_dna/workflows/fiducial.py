from higgs_dna.workflows.skeleton import HggSkeletonProcessor

from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EELeak_region import veto_EEleak_flag
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from higgs_dna.tools.sigma_m_tools import compute_sigma_m
from higgs_dna.tools.jetID import add_jetId
from higgs_dna.selections.photon_selections import photon_preselection
from higgs_dna.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from higgs_dna.selections.lepton_selections import select_electrons, select_muons
from higgs_dna.selections.jet_selections import select_jets, select_jets_eta_dependent, jetvetomap, getBTagMVACut
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
)
from higgs_dna.utils.misc_utils import choose_jet, DPhiV1V2, rapidity_from_pt_eta_mass
from higgs_dna.tools.flow_corrections import apply_flow_corrections_to_photons

from higgs_dna.tools.mass_decorrelator import decorrelate_mass_resolution

# from higgs_dna.utils.dumping_utils import diphoton_list_to_pandas, dump_pandas
from higgs_dna.systematics import object_systematics as available_object_systematics
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_systematics as available_weight_systematics
from higgs_dna.systematics import weight_corrections as available_weight_corrections
from higgs_dna.systematics import apply_systematic_variations_object_level

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


class HggFiducialProcessor(HggSkeletonProcessor):  # type: ignore
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        apply_trigger: bool = False,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        nano_version: int = None,
        bTagEffFileName: Optional[str] = None,
        trigger_group: str = ".*DoubleEG.*",
        analysis: str = "mainAnalysis",
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

    def process(self, events: ak.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

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
        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(numpy.sum(events.genWeight.to_numpy()))
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

        # save raw pt if we use scale/smearing corrections
        # These needs to be before the smearing of the mass resolution in order to have the raw pt for the function
        s_or_s_applied = False
        s_or_s_ele_applied = False
        for correction in correction_names:
            correction_name_lower = correction.lower()
            if any(keyword in correction_name_lower for keyword in ("scale", "smearing")):
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

            logger.info(
                f"""
                Applying correction {correction_name} to dataset {dataset_name}\n
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
                logger.warning(f"Could not process correction {correction_name}.")
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

        # Write systematic variations to dictss
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

                GenPTH, GenYH, GenPhiH, GenLeadPho, GenSubleadPho = get_higgs_gen_attributes(events)
                genJets = get_genJets(
                    events,
                    pt_cut=30.,
                    eta_cut=4.7,
                    jet_pho_min_dr=self.jet_pho_min_dr,
                    jet_ele_min_dr=self.jet_ele_min_dr,
                    jet_muo_min_dr=self.jet_muo_min_dr,
                    electron_pt_threshold=self.electron_pt_threshold,
                    electron_max_eta=self.electron_max_eta,
                    muon_pt_threshold=self.muon_pt_threshold,
                    muon_max_eta=self.muon_max_eta,
                )

                ######################
                # Diphoton Variables #
                ######################
                GenPTH = ak.fill_none(GenPTH, -999.0)
                diphotons['GenPTH'] = GenPTH

                GenYH = ak.fill_none(GenYH, -999)
                GenYH = ak.where(numpy.isnan(GenYH), -999, GenYH)
                diphotons['GenYH'] = GenYH

                GenDeltaPhoPhi = GenLeadPho.phi - GenSubleadPho.phi
                GenDeltaPhoPhi_pi_array = ak.full_like(GenDeltaPhoPhi, 2 * numpy.pi)
                # Select the smallest angle
                GenDeltaPhoPhi = ak.where(
                    GenDeltaPhoPhi > numpy.pi,
                    GenDeltaPhoPhi - GenDeltaPhoPhi_pi_array,
                    GenDeltaPhoPhi
                )
                GenDeltaPhoPhi = ak.where(
                    GenDeltaPhoPhi < -numpy.pi,
                    GenDeltaPhoPhi + GenDeltaPhoPhi_pi_array,
                    GenDeltaPhoPhi
                )
                GenAcop = ak.full_like(GenDeltaPhoPhi, numpy.pi) - GenDeltaPhoPhi
                GenThetaEtaStar = numpy.tan(GenAcop / 2) / numpy.cosh((GenLeadPho.eta - GenSubleadPho.eta) / 2)
                GenThetaEtaStar = ak.fill_none(GenThetaEtaStar, -999.0)
                diphotons['GenThetaEtaStar'] = GenThetaEtaStar

                GenDiphoton = GenLeadPho + GenSubleadPho
                GenDiPhoMass = GenDiphoton.mass
                GenDiPhoPT = GenDiphoton.pt
                GenCosThetaStarCS = 2 * (((GenLeadPho.pz * GenSubleadPho.energy) - (GenLeadPho.energy * GenSubleadPho.pz)) / (GenDiPhoMass * numpy.sqrt(GenDiPhoMass**2 + GenDiPhoPT**2)))
                GenCosThetaStarCS = ak.fill_none(GenCosThetaStarCS, -999.0)
                diphotons['GenCosThetaStarCS'] = GenCosThetaStarCS

                #########################
                # Leading Jet Variables #
                #########################
                genJets_absEta4p7 = genJets[numpy.abs(genJets.eta) < 4.7]
                genJets_absEta2p5 = genJets[numpy.abs(genJets.eta) < 2.5]
                diphotons["GenNJ"] = ak.num(genJets_absEta4p7)
                diphotons["GenNJ_pt30_absEta2p5"] = ak.num(genJets_absEta2p5)
                diphotons["GenPTJ0"] = choose_jet(genJets_absEta4p7.pt, 0, -999.0)
                diphotons["GenPTJ0_pt30_absEta2p5"] = choose_jet(genJets_absEta2p5.pt, 0, -999.0)

                # Choose zero (leading) jet and pad with -999 if none
                GenPTJ0 = choose_jet(genJets.pt, 0, -999.0)

                gen_first_jet_eta = choose_jet(genJets.eta, 0, -999.0)
                gen_first_jet_mass = choose_jet(genJets.mass, 0, -999.0)
                gen_first_jet_phi = choose_jet(genJets.phi, 0, -999.0)

                diphotons['gen_first_jet_eta'] = gen_first_jet_eta
                diphotons['gen_first_jet_mass'] = gen_first_jet_mass
                diphotons['gen_first_jet_phi'] = gen_first_jet_phi

                GenYJ0 = rapidity_from_pt_eta_mass(GenPTJ0, gen_first_jet_eta, gen_first_jet_mass, fill_value=-999.0)
                diphotons["GenYJ0"] = GenYJ0

                GenPTJ0_absEta2p5 = choose_jet(genJets_absEta2p5.pt, 0, -999.0)
                gen_first_jet_eta_absEta2p5 = choose_jet(genJets_absEta2p5.eta, 0, -999.0)
                gen_first_jet_mass_absEta2p5 = choose_jet(genJets_absEta2p5.mass, 0, -999.0)
                GenYJ0_absEta2p5 = rapidity_from_pt_eta_mass(
                    GenPTJ0_absEta2p5,
                    gen_first_jet_eta_absEta2p5,
                    gen_first_jet_mass_absEta2p5,
                    fill_value=-999.0,
                )
                diphotons["GenYJ0_pt30_absEta2p5"] = GenYJ0_absEta2p5

                GenDYHJ0 = GenYJ0 - GenYH
                # Set all entries above 500 to -999
                GenDYHJ0 = ak.where(
                    numpy.abs(GenDYHJ0) > 500,
                    -999,
                    GenDYHJ0
                )
                GenDYHJ0 = ak.fill_none(GenDYHJ0, -999.0)
                diphotons["GenDYHJ0"] = GenDYHJ0

                GenHPhi = ak.fill_none(GenPhiH, -999)

                GenDPhiHJ0 = gen_first_jet_phi - GenHPhi
                GenDPhiHJ0 = (GenDPhiHJ0 + numpy.pi) % (2 * numpy.pi) - numpy.pi
                GenDPhiHJ0 = ak.where(GenHPhi == -999, -999, GenDPhiHJ0)
                GenDPhiHJ0 = ak.where(gen_first_jet_phi == -999, -999, GenDPhiHJ0)
                diphotons["GenDPhiHJ0"] = GenDPhiHJ0

                #################################
                # Next-to-leading Jet Variables #
                #################################
                GenPTJ1 = choose_jet(genJets.pt, 1, -999.0)
                diphotons['GenPTJ1'] = GenPTJ1

                gen_second_jet_eta = choose_jet(genJets.eta, 1, -999.0)
                gen_second_jet_mass = choose_jet(genJets.mass, 1, -999.0)
                gen_second_jet_phi = choose_jet(genJets.phi, 1, -999.0)

                diphotons['gen_second_jet_eta'] = gen_second_jet_eta
                diphotons['gen_second_jet_mass'] = gen_second_jet_mass
                diphotons['gen_second_jet_phi'] = gen_second_jet_phi

                GenYJ1 = rapidity_from_pt_eta_mass(GenPTJ1, gen_second_jet_eta, gen_second_jet_mass, fill_value=-999.0)
                diphotons['GenYJ1'] = GenYJ1

                GenDYJ0J1 = GenYJ0 - GenYJ1
                # Set all entries above 500 to -999
                GenDYJ0J1 = ak.where(
                    numpy.abs(GenDYJ0J1) > 500,
                    -999,
                    GenDYJ0J1
                )
                # Set all entries which are precisely 0 to -999
                GenDYJ0J1 = ak.where(
                    GenDYJ0J1 == 0,
                    -999,
                    GenDYJ0J1
                )
                GenDYJ0J1 = ak.fill_none(GenDYJ0J1, -999.0)
                diphotons["GenDYJ0J1"] = GenDYJ0J1

                gen_first_jet_vector = ak.zip({
                    "pt": GenPTJ0,
                    "eta": gen_first_jet_eta,
                    "phi": gen_first_jet_phi,
                    "mass": gen_first_jet_mass
                }, with_name="Momentum4D")

                gen_second_jet_vector = ak.zip({
                    "pt": GenPTJ1,
                    "eta": gen_second_jet_eta,
                    "phi": gen_second_jet_phi,
                    "mass": gen_second_jet_mass
                }, with_name="Momentum4D")

                GenDPhiJ0J1 = DPhiV1V2(gen_first_jet_vector, gen_second_jet_vector)
                diphotons["GenDPhiJ0J1"] = GenDPhiJ0J1

                padded_genJets = genJets[ak.argsort(genJets.pt, ascending=False)]
                # First build the dijet system out of the leading and subleading jet (in pt)
                padded_genJets = ak.pad_none(genJets, 2)
                genDijet = padded_genJets[:, 0] + padded_genJets[:, 1]

                GenMassJ0J1 = ak.fill_none(genDijet.mass, -999.0)
                diphotons["GenMassJ0J1"] = GenMassJ0J1

                GenDijetEta = ak.fill_none(genDijet.eta, -999.0)
                GenDiphotonEta = ak.fill_none(GenDiphoton.eta, -999.0)
                GenDEtaJ0J1H = GenDijetEta - GenDiphotonEta
                # Set all entries which are precisely 0 to -999
                GenDEtaJ0J1H = ak.where(
                    GenDEtaJ0J1H == 0,
                    -999,
                    GenDEtaJ0J1H
                )
                # Set all entries which are above 500 in absolute value to -999 (come from either no diphoton or no dijet system)
                GenDEtaJ0J1H = ak.where(
                    numpy.abs(GenDEtaJ0J1H) > 500,
                    -999,
                    GenDEtaJ0J1H
                )
                GenDEtaJ0J1H = ak.fill_none(GenDEtaJ0J1H, -999.0)
                diphotons["GenDEtaJ0J1H"] = GenDEtaJ0J1H

                GenDijetPhi = ak.fill_none(genDijet.phi, -999)
                GenHPhi = ak.fill_none(GenPhiH, -999)

                GenDPhiHJ0J1 = GenDijetPhi - GenPhiH
                GenDPhiHJ0J1 = (GenDPhiHJ0J1 + numpy.pi) % (2 * numpy.pi) - numpy.pi
                GenDPhiHJ0J1 = ak.where(GenHPhi == -999, -999, GenDPhiHJ0J1)
                GenDPhiHJ0J1 = ak.where(GenDijetPhi == -999, -999, GenDPhiHJ0J1)
                diphotons["GenDPhiHJ0J1"] = GenDPhiHJ0J1

                GenEtaJ0J1 = gen_first_jet_eta - gen_second_jet_eta
                # Set all entries which are precisely 0 to -999
                GenEtaJ0J1 = ak.where(
                    GenEtaJ0J1 == 0,
                    -999,
                    GenEtaJ0J1
                )
                # Set all entries which are above 500 in absolute value to -999 (come from either no diphoton or no dijet system)
                GenEtaJ0J1 = ak.where(
                    numpy.abs(GenEtaJ0J1) > 500,
                    -999,
                    GenEtaJ0J1
                )
                GenEtaJ0J1 = ak.fill_none(GenEtaJ0J1, -999.0)
                diphotons["GenEtaJ0J1"] = GenEtaJ0J1

                ###########################
                # Event Level Observables #
                ###########################
                # B-Jets
                # Following the recommendations of https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideBTagMCTools for hadronFlavour
                # and the Run 2 recommendations for the bjets
                genJetCondition = (genJets.pt > 30) & (numpy.abs(genJets.eta) < 4.7)
                genBJetCondition = genJetCondition & (genJets.hadronFlavour == 5)
                genJets = ak.with_field(genJets, genBJetCondition, "GenIsBJet")
                num_bjets = ak.sum(genJets["GenIsBJet"], axis=-1)
                diphotons["GenNBJet"] = num_bjets

                gen_first_bjet_pt = choose_jet(genJets[genJets["GenIsBJet"] == True].pt, 0, -999.0)
                diphotons["GenPTbJ0"] = gen_first_bjet_pt

                # Jet Rapidity Observable
                # Iterate over max six largest pt jets to compute tauJC
                GenTauJC_list = []
                GenTauJC_maxJets = 10
                for i in range(GenTauJC_maxJets):
                    mass = choose_jet(genJets.mass, i, -999.0)
                    pt = choose_jet(genJets.pt, i, -999.0)
                    eta = choose_jet(genJets.eta, i, -999.0)

                    with numpy.errstate(over='ignore', invalid='ignore'):
                        cosh_eta = numpy.cosh(eta)
                        sinh_eta = numpy.sinh(eta)

                        energy = numpy.sqrt((pt**2 * cosh_eta**2) + mass**2)
                        pz = pt * sinh_eta

                        # If energy or pz is inf (cause of the hyperbolic functions), set them to -999
                        # later set every GenTauJC to -999 which has a value of precisely 0 (corresponding to a transverse momentum of exactly 0 GeV)
                        energy = ak.where(numpy.isinf(energy), -999, energy)
                        pz = ak.where(numpy.isinf(pz), -999, pz)

                        transverse_mass = numpy.sqrt(pt**2 + mass**2)

                        y = numpy.log((numpy.sqrt((mass**2 + pt**2) * cosh_eta**2) + (pt * sinh_eta)) / transverse_mass)

                        tau_jc = numpy.sqrt(energy**2 - pz**2) / (2 * numpy.cosh(y - GenYH))
                    GenTauJC_list.append(tau_jc)

                    logger.debug(f"GenTauJC: Jet {i}: Energy={energy}, Pz={pz}, TauJC={tau_jc}")

                # Convert to awkward array for proper axis manipulation
                GenTauJC_array = ak.Array(GenTauJC_list)
                flipped_GenTauJC = ak.unzip(GenTauJC_array)[0] if isinstance(GenTauJC_array, tuple) else GenTauJC_array
                GenTauJC = ak.max(flipped_GenTauJC, axis=0)
                GenTauJC = ak.where(GenTauJC == 0, -999, GenTauJC)
                GenTauJC = ak.fill_none(GenTauJC, -999.0)
                diphotons["GenTauJC"] = GenTauJC

            # baseline modifications to diphotons
            if self.diphoton_mva is not None:
                diphotons = self.add_diphoton_mva(diphotons, events)

            # workflow specific processing
            events, process_extra = self.process_extra(events)
            histos_etc.update(process_extra)

            btagMVA_selection = {
                "deepJet": {"btagDeepFlavB": jets.btagDeepFlavB},  # Always available
                "particleNet": {"btagPNetB": jets.btagPNetB} if self.nano_version >= 12 else {},
                "robustParticleTransformer": {"btagRobustParTAK4B": jets.btagRobustParTAK4B} if self.nano_version in [12, 13] else {},
                "btagUParTAK4B": {"btagUParTAK4B": jets.btagUParTAK4B} if self.nano_version >= 15 else {},
            }

            # jet_variables
            jets = ak.zip(
                {
                    "pt": jets.pt,
                    "eta": jets.eta,
                    "phi": jets.phi,
                    "mass": jets.mass,
                    "charge": ak.zeros_like(
                        jets.pt
                    ),
                    **btagMVA_selection.get(self.bjet_mva, {}),
                    "hFlav": jets.hadronFlavour if self.data_kind == "mc" else ak.zeros_like(jets.pt),
                    "btagDeepFlav_CvB": jets.btagDeepFlavCvB,
                    "btagDeepFlav_CvL": jets.btagDeepFlavCvL,
                    "btagDeepFlav_QG": jets.btagDeepFlavQG,
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
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                    "charge": events.Electron.charge,
                    "cutBased": events.Electron.cutBased,
                    "mvaIso_WP90": events.Electron.mvaIso_WP90,
                    "mvaIso_WP80": events.Electron.mvaIso_WP80,
                }
            )
            electrons = ak.with_name(electrons, "PtEtaPhiMCandidate")

            # Special cut for base workflow to replicate iso cut for electrons also for muons
            events['Muon'] = events.Muon[events.Muon.pfRelIso03_all < 0.2]

            muons = ak.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                    "charge": events.Muon.charge,
                    "tightId": events.Muon.tightId,
                    "mediumId": events.Muon.mediumId,
                    "looseId": events.Muon.looseId,
                    "isGlobal": events.Muon.isGlobal,
                    "pfIsoId": events.Muon.pfIsoId
                }
            )
            muons = ak.with_name(muons, "PtEtaPhiMCandidate")

            # lepton cleaning
            sel_electrons = electrons[
                select_electrons(self, electrons, diphotons)
            ]
            sel_muons = muons[select_muons(self, muons, diphotons)]

            # jet selection
            # Eta-dependent jet pt cuts
            # Keep at 30 in the central and far forward region (not below 20, 20-30 low pt jets have problems)
            # raised to 50 in the 2.5-3.0 absEta region due to jet spikes (JME recommendation)
            if self.year[dataset_name][0] in ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix","2024"]:
                self.jet_pt_thresholds = [30, 50, 50]
                self.jet_eta_thresholds = [2.5, 3.0, 4.7]
                jet_selection_func = select_jets_eta_dependent
            else:
                jet_selection_func = select_jets
            jets = jets[
                jet_selection_func(self, jets, diphotons, sel_muons, sel_electrons)
            ]
            jets = jets[ak.argsort(jets.pt, ascending=False)]

            # adding selected jets to events to be used in ctagging SF calculation
            events["sel_jets"] = jets

            ######################
            # Diphoton Variables #
            ######################
            LeadPho = diphotons["pho_lead"]
            SubleadPho = diphotons["pho_sublead"]

            DeltaPhoPhi = LeadPho.phi - SubleadPho.phi
            DeltaPhoPhi_pi_array = ak.full_like(DeltaPhoPhi, 2 * numpy.pi)
            # Select the smallest angle
            DeltaPhoPhi = ak.where(
                DeltaPhoPhi > numpy.pi,
                DeltaPhoPhi - DeltaPhoPhi_pi_array,
                DeltaPhoPhi
            )
            DeltaPhoPhi = ak.where(
                DeltaPhoPhi < -numpy.pi,
                DeltaPhoPhi + DeltaPhoPhi_pi_array,
                DeltaPhoPhi
            )
            Acop = ak.full_like(DeltaPhoPhi, numpy.pi) - DeltaPhoPhi
            ThetaEtaStar = numpy.tan(Acop / 2) / numpy.cosh((LeadPho.eta - SubleadPho.eta) / 2)
            ThetaEtaStar = ak.fill_none(ThetaEtaStar, -999.0)
            diphotons['ThetaEtaStar'] = ThetaEtaStar

            CosThetaStarCS = 2 * (((LeadPho.pz * SubleadPho.energy) - (LeadPho.energy * SubleadPho.pz)) / (diphotons["mass"] * numpy.sqrt(diphotons["mass"]**2 + diphotons["pt"]**2)))
            CosThetaStarCS = ak.fill_none(CosThetaStarCS, -999.0)
            diphotons['CosThetaStarCS'] = CosThetaStarCS

            #########################
            # Leading Jet Variables #
            #########################
            jets_absEta4p7 = jets[numpy.abs(jets.eta) < 4.7]
            jets_absEta2p5 = jets[numpy.abs(jets.eta) < 2.5]
            diphotons["NJ"] = ak.num(jets_absEta4p7)
            diphotons["NJ_pt30_absEta2p5"] = ak.num(jets_absEta2p5)
            diphotons["PTJ0"] = choose_jet(jets_absEta4p7.pt, 0, -999.0)
            diphotons["PTJ0_pt30_absEta2p5"] = choose_jet(jets_absEta2p5.pt, 0, -999.0)

            PTJ0 = choose_jet(jets.pt, 0, -999.0)
            first_jet_eta = choose_jet(jets.eta, 0, -999.0)
            first_jet_phi = choose_jet(jets.phi, 0, -999.0)
            first_jet_mass = choose_jet(jets.mass, 0, -999.0)
            YJ0 = rapidity_from_pt_eta_mass(PTJ0, first_jet_eta, first_jet_mass, fill_value=-999.0)
            diphotons["YJ0"] = YJ0

            PTJ0_absEta2p5 = choose_jet(jets_absEta2p5.pt, 0, -999.0)
            first_jet_eta_absEta2p5 = choose_jet(jets_absEta2p5.eta, 0, -999.0)
            first_jet_mass_absEta2p5 = choose_jet(jets_absEta2p5.mass, 0, -999.0)
            YJ0_absEta2p5 = rapidity_from_pt_eta_mass(
                PTJ0_absEta2p5,
                first_jet_eta_absEta2p5,
                first_jet_mass_absEta2p5,
                fill_value=-999.0,
            )
            diphotons["YJ0_pt30_absEta2p5"] = YJ0_absEta2p5

            diphotons["first_jet_eta"] = first_jet_eta
            diphotons["first_jet_phi"] = first_jet_phi
            diphotons["first_jet_mass"] = first_jet_mass

            DYHJ0 = YJ0 - diphotons["eta"]
            # Set all entries above 500 to -999
            DYHJ0 = ak.where(
                numpy.abs(DYHJ0) > 500,
                -999,
                DYHJ0
            )
            DYHJ0 = ak.fill_none(DYHJ0, -999.0)
            diphotons["DYHJ0"] = DYHJ0

            HPhi = ak.fill_none(diphotons.phi, -999)
            DPhiHJ0 = first_jet_phi - HPhi
            DPhiHJ0 = (DPhiHJ0 + numpy.pi) % (2 * numpy.pi) - numpy.pi
            DPhiHJ0 = ak.where(HPhi == -999, -999, DPhiHJ0)
            DPhiHJ0 = ak.where(first_jet_phi == -999, -999, DPhiHJ0)
            diphotons["DPhiHJ0"] = DPhiHJ0

            #################################
            # Next-to-leading Jet Variables #
            #################################
            PTJ1 = choose_jet(jets.pt, 1, -999.0)
            diphotons["PTJ1"] = PTJ1
            second_jet_eta = choose_jet(jets.eta, 1, -999.0)
            second_jet_phi = choose_jet(jets.phi, 1, -999.0)
            second_jet_mass = choose_jet(jets.mass, 1, -999.0)
            diphotons["second_jet_eta"] = second_jet_eta
            diphotons["second_jet_phi"] = second_jet_phi
            diphotons["second_jet_mass"] = second_jet_mass

            YJ1 = rapidity_from_pt_eta_mass(PTJ1, second_jet_eta, second_jet_mass, fill_value=-999.0)
            diphotons['YJ1'] = YJ1

            DYJ0J1 = YJ0 - YJ1
            # Set all entries above 500 to -999
            DYJ0J1 = ak.where(
                numpy.abs(DYJ0J1) > 500,
                -999,
                DYJ0J1
            )
            # Set all entries which are precisely 0 to -999
            DYJ0J1 = ak.where(
                DYJ0J1 == 0,
                -999,
                DYJ0J1
            )
            DYJ0J1 = ak.fill_none(DYJ0J1, -999.0)
            diphotons["DYJ0J1"] = DYJ0J1

            first_jet_vector = ak.zip({
                "pt": PTJ0,
                "eta": first_jet_eta,
                "phi": first_jet_phi,
                "mass": first_jet_mass
            }, with_name="Momentum4D")

            second_jet_vector = ak.zip({
                "pt": PTJ1,
                "eta": second_jet_eta,
                "phi": second_jet_phi,
                "mass": second_jet_mass
            }, with_name="Momentum4D")

            DPhiJ0J1 = DPhiV1V2(first_jet_vector, second_jet_vector)
            diphotons["DPhiJ0J1"] = DPhiJ0J1

            # First build the dijet system out of the leading and subleading jet (in pt)
            padded_jets = ak.pad_none(jets, 2)
            dijet = padded_jets[:, 0] + padded_jets[:, 1]

            MassJ0J1 = ak.fill_none(dijet.mass, -999.0)
            diphotons["MassJ0J1"] = MassJ0J1

            DijetEta = ak.fill_none(dijet.eta, -999.0)
            DijetPhi = ak.fill_none(dijet.phi, -999.0)
            DiphotonEta = ak.fill_none(diphotons["eta"], -999.0)
            DEtaJ0J1H = DijetEta - DiphotonEta
            # Set all entries which are precisely 0 to -999
            DEtaJ0J1H = ak.where(
                DEtaJ0J1H == 0,
                -999,
                DEtaJ0J1H
            )
            # Set all entries which are above 500 in absolute value to -999 (come from either no diphoton or no dijet system)
            DEtaJ0J1H = ak.where(
                numpy.abs(DEtaJ0J1H) > 500,
                -999,
                DEtaJ0J1H
            )
            DEtaJ0J1H = ak.fill_none(DEtaJ0J1H, -999.0)
            diphotons["DEtaJ0J1H"] = DEtaJ0J1H

            DPhiHJ0J1 = DijetPhi - HPhi
            DPhiHJ0J1 = (DPhiHJ0J1 + numpy.pi) % (2 * numpy.pi) - numpy.pi
            DPhiHJ0J1 = ak.where(HPhi == -999, -999, DPhiHJ0J1)
            DPhiHJ0J1 = ak.where(DijetPhi == -999, -999, DPhiHJ0J1)
            diphotons["DPhiHJ0J1"] = DPhiHJ0J1

            EtaJ0J1 = first_jet_eta - second_jet_eta
            # Set all entries which are precisely 0 to -999
            EtaJ0J1 = ak.where(
                EtaJ0J1 == 0,
                -999,
                EtaJ0J1
            )
            # Set all entries which are above 500 in absolute value to -999 (come from either no diphoton or no dijet system)
            EtaJ0J1 = ak.where(
                numpy.abs(EtaJ0J1) > 500,
                -999,
                EtaJ0J1
            )
            EtaJ0J1 = ak.fill_none(EtaJ0J1, -999.0)
            diphotons["EtaJ0J1"] = EtaJ0J1

            ###########################
            # Event Level Observables #
            ###########################

            # B-Jets
            btag_WP = getBTagMVACut(mva_name=self.bjet_mva,
                                    mva_wp=self.bjet_wp,
                                    year=self.year[dataset_name][0])

            btag_mva_column = list(btagMVA_selection[self.bjet_mva].keys())[0]

            bJetCondition = (jets.pt > 30) & (abs(jets.eta) < 4.7) & (jets[btag_mva_column] >= btag_WP)
            jets = ak.with_field(jets, bJetCondition, f"{self.bjet_mva}_IsBJet")
            num_bjets = ak.sum(jets[f"{self.bjet_mva}_IsBJet"], axis=-1)
            diphotons[f"{self.bjet_mva}_NBJet"] = num_bjets

            first_bjet_pt = choose_jet(jets[jets[f"{self.bjet_mva}_IsBJet"] == True].pt, 0, -999.0)
            diphotons[f"{self.bjet_mva}_PTbJ0"] = first_bjet_pt

            first_bjet_mva = choose_jet(jets[jets[f"{self.bjet_mva}_IsBJet"] == True][btag_mva_column], 0, -999.0)
            diphotons[f"{self.bjet_mva}_ScorebJ0"] = first_bjet_mva

            # Jet Rapidity Observable
            # Iterate over max six largest pt jets to compute tauJC
            TauJC_list = []
            TauJC_maxJets = 10
            for i in range(TauJC_maxJets):
                mass = choose_jet(jets.mass, i, -999.0)
                pt = choose_jet(jets.pt, i, -999.0)
                eta = choose_jet(jets.eta, i, -999.0)

                with numpy.errstate(over='ignore', invalid='ignore'):
                    cosh_eta = numpy.cosh(eta)
                    sinh_eta = numpy.sinh(eta)

                    energy = numpy.sqrt((pt**2 * cosh_eta**2) + mass**2)
                    pz = pt * sinh_eta

                    # If energy or pz is inf (cause of the hyperbolic functions), set them to -999
                    # later set every TauJC to -999 which has a value of precisely 0 (corresponding to a transverse momentum of exactly 0 GeV)
                    energy = ak.where(numpy.isinf(energy), -999, energy)
                    pz = ak.where(numpy.isinf(pz), -999, pz)

                    transverse_mass = numpy.sqrt(pt**2 + mass**2)

                    y = numpy.log((numpy.sqrt((mass**2 + pt**2) * cosh_eta**2) + (pt * sinh_eta)) / transverse_mass)

                    tau_jc = numpy.sqrt(energy**2 - pz**2) / (2 * numpy.cosh(y - diphotons["rapidity"]))
                TauJC_list.append(tau_jc)

                logger.debug(f"TauJC: Jet {i}: Energy={energy}, Pz={pz}, TauJC={tau_jc}")

            # Convert to awkward array for proper axis manipulation
            TauJC_array = ak.Array(TauJC_list)
            flipped_TauJC = ak.unzip(TauJC_array)[0] if isinstance(TauJC_array, tuple) else TauJC_array
            TauJC = ak.max(flipped_TauJC, axis=0)
            TauJC = ak.where(TauJC == 0, -999, TauJC)
            TauJC = ak.fill_none(TauJC, -999.0)
            diphotons["TauJC"] = TauJC

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
            diphotons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # Beamspot variables
            diphotons["BeamSpot_sigmaZ"] = events.BeamSpot.sigmaZ
            diphotons["BeamSpot_sigmaZError"] = events.BeamSpot.sigmaZError
            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = events.genWeight
                diphotons["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                diphotons["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                diphotons["HTXS_stage_0"] = events.HTXS.stage_0
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

            bTagFixedWP_present = any("bTagFixedWP" in item for item in systematic_names) + any("bTagFixedWP" in item for item in correction_names)
            PNet_present = any("bTagFixedWP_PNet" in item for item in systematic_names) + any("bTagFixedWP_PNet" in item for item in correction_names)

            if PNet_present and (self.nano_version < 12):
                logger.error("\n B-Tagging systematics and corrections using Particle Net are only available for NanoAOD v12 or higher. Exiting! \n")
                exit()

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
                        common_args = {
                            "events": events[selection_mask],
                            "photons": original_diphotons[selection_mask],
                            "weights": event_weights,
                            "dataset_name": dataset_name,
                            "year": self.year[dataset_name][0],
                        }

                        if any("bTagFixedWP" in item for item in correction_names):
                            common_args["bTagEffFileName"] = self.bTagEffFileName

                        varying_function = available_weight_corrections[correction_name]
                        event_weights = varying_function(**common_args)

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
                                    for index, elem in enumerate(ak.sum(diphotons["weight_LHEScale"], axis=0)):
                                        metadata[f"sum_weight_LHEScale_{index}"] = str(elem)
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
                                    for index, elem in enumerate(ak.sum(diphotons["weight_LHEPdf"], axis=0)):
                                        metadata[f"sum_weight_LHEPdf_{index}"] = str(elem)
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                common_args = {
                                    "events": events[selection_mask],
                                    "photons": original_diphotons[selection_mask],
                                    "weights": event_weights,
                                    "dataset_name": dataset_name,
                                    "year": self.year[dataset_name][0],
                                }

                                if any("bTagFixedWP" in item for item in systematic_names):
                                    common_args["bTagEffFileName"] = self.bTagEffFileName

                                varying_function = available_weight_systematics[systematic_name]
                                event_weights = varying_function(**common_args)

                diphotons["weight"] = event_weights.weight() / (
                    event_weights.partial_weight(include=["bTagFixedWP"])
                    if bTagFixedWP_present
                    else 1
                )
                diphotons["weight_central"] = event_weights.weight() / (
                    (event_weights.partial_weight(include=["bTagFixedWP"]) * events["genWeight"][selection_mask])
                    if bTagFixedWP_present
                    else events["genWeight"][selection_mask]
                )

                if bTagFixedWP_present:
                    diphotons["weight_bTagFixedWP"] = event_weights.partial_weight(include=["bTagFixedWP"])

                metadata["sum_weight_central"] = str(
                    ak.sum(
                        event_weights.weight()
                        / (
                            event_weights.partial_weight(include=["bTagFixedWP"])
                            if bTagFixedWP_present
                            else 1
                        )
                    )
                )
                metadata["sum_weight_central_wo_bTagSF"] = str(
                    ak.sum(
                        event_weights.weight()
                        / (
                            (event_weights.partial_weight(include=["bTagSF"]) * event_weights.partial_weight(include=["bTagFixedWP"]))
                            if bTagFixedWP_present
                            else event_weights.partial_weight(include=["bTagSF"])
                        )
                    )
                )

                # Handle variations
                if do_variation == "nominal":
                    if event_weights.variations:
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        diphotons["weight_" + modifier] = event_weights.weight(modifier=modifier)
                        if "bTagSF" in modifier:
                            metadata["sum_weight_" + modifier] = str(
                                ak.sum(event_weights.weight(modifier=modifier))
                                / (
                                    event_weights.partial_weight(include=["bTagFixedWP"])
                                    if bTagFixedWP_present
                                    else 1
                                )
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

                    # Remove fixedGridRhoAll from photons to avoid having event-level info per photon
                    akarr = akarr[
                        [
                            field
                            for field in akarr.fields
                            if "lead_fixedGridRhoAll" not in field
                        ]
                    ]

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
