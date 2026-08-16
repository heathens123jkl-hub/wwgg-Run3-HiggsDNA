# WWgg workflow for Run3 HiggsDNA — non-resonant HH→WWγγ preselection
# Selection conditions aligned with Analysis Note
# Based on HggSkeletonProcessor (same base as HHbbgg)

from typing import Any, Dict, List, Optional
from higgs_dna.workflows.skeleton import HggSkeletonProcessor
from higgs_dna.selections.photon_selections import photon_preselection
from higgs_dna.selections.diphoton_selections import build_diphoton_candidates, apply_fiducial_cut_det_level
from higgs_dna.selections.lepton_selections import select_electrons, select_muons
from higgs_dna.selections.jet_selections import select_jets, jetvetomap
from higgs_dna.tools.jetID import add_jetId
from higgs_dna.selections.object_selections import delta_r_mask
from higgs_dna.selections.lumi_selections import select_lumis
from higgs_dna.utils.dumping_utils import diphoton_ak_array, dump_ak_array, apply_naming_convention
from higgs_dna.tools.SC_eta import add_photon_SC_eta
from higgs_dna.tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from higgs_dna.systematics import object_corrections as available_object_corrections
from higgs_dna.systematics import weight_corrections as available_weight_corrections

import awkward as ak
import numpy
import vector
import logging
import warnings
from coffea.analysis_tools import Weights

logger = logging.getLogger(__name__)
vector.register_awkward()


class WWggProcessor(HggSkeletonProcessor):
    """
    HH→WWγγ non-resonant preselection processor for Run3.

    Event categories (AN Note §5.2-5.4):
      Cat 0: FH — 0 lepton + ≥4 jets  (WW→qqqq)
      Cat 1: SL — 1 lepton            (WW→qqℓν)
      Cat 2: FL — ≥2 leptons + MET>20 + pT(γγ)>91 + Z→ll veto + b-veto  (WW→ℓνℓν)
    """

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
        trigger_group: str = ".*DoubleEG.*",
        analysis: str = "mainAnalysis",
        applyCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "classical_noIso",
        doDeco: bool = False,
        Smear_sigma_m: bool = False,
        doFlow_corrections: bool = False,
        validate_with_electrons: bool = False,
        output_format: str = "parquet",
        split_mc: bool = False,
    ) -> None:
        super().__init__(
            metaconditions,
            systematics=systematics,
            corrections=corrections,
            nano_version=nano_version,
            apply_trigger=apply_trigger,
            output_location=output_location,
            bTagEffFileName=bTagEffFileName,
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
            output_format=output_format,
        )

        # === WW-specific overrides (after skeleton defaults are set) ===

        # Fiducial cuts — store_flag, same as HHbbgg
        self.fiducialCuts = "store_flag"

        # Photon preselection (aligned with bbgg preselection standards)
        self.min_pt_photon = 25.0
        self.min_mvaid = -0.9

        # Electron selection (AN2025_108 Table 13 tight)
        self.electron_pt_threshold = 10.0
        self.electron_max_eta = 2.5
        self.electron_photon_min_dr = 0.4
        self.electron_max_dxy = 0.05
        self.electron_max_dz = 0.1
        self.el_id_wp = "loose"   # Note Table 13: EGamma POG MVA > WP-loose

        # Muon selection (AN2025_108 Table 14 tight)
        self.muon_pt_threshold = 10.0
        self.muon_max_eta = 2.4
        self.muon_photon_min_dr = 0.4
        self.muon_max_dxy = 0.05
        self.muon_max_dz = 0.1
        self.mu_id_wp = "medium"
        self.mu_iso_wp = "loose"
        self.global_muon = True

        # AK4 Jet selection (aligned with bbgg)
        # jet_pt_threshold=20 and jet_max_eta=4.7 from skeleton defaults
        self.jet_pho_min_dr = 0.4
        self.clean_jet_pho = True
        self.clean_jet_ele = False   # bbgg: no dR(jet, e) cleaning
        self.clean_jet_muo = False   # bbgg: no dR(jet, mu) cleaning

        # Event categories
        self.categories = {0: "FH", 1: "SL", 2: "FL"}

        logger.info("[WWggProcessor] Initialized for non-resonant HH→WWγγ")

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass

    def process(self, events):
        dataset_name = events.metadata["dataset"]
        year = self.year.get(dataset_name, ["2022preEE"])[0]
        n_total_input = len(events)  # before any selection

        # === MC/Data info ===
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        # === Lumi mask (data only) ===
        if self.data_kind == "data":
            try:
                events = events[select_lumis(year, events, logger)]
            except Exception:
                logger.info(f"[WWgg] Lumi mask skip for {dataset_name}")

        # Sum of gen weights before any event-level selection (needed for
        # normalization: weight_norm = xsec * lumi / sum_genw_presel)
        if self.data_kind == "mc":
            sum_genw_presel = numpy.sum(events.genWeight.to_numpy())
        else:
            sum_genw_presel = None

        # Read which corrections to process (same as HHbbgg)
        try:
            correction_names = self.corrections[dataset_name]
        except (KeyError, TypeError):
            correction_names = []

        # === Filters & triggers ===
        events = self.apply_filters_and_triggers(events)

        # Remove events affected by EcalBadCalibCrystal (Run3 data only, same as HHbbgg L227-231)
        if self.data_kind == "data":
            excluded_years = ["2018", "2017", "2016preVFP", "2016postVFP"]
            if year not in excluded_years:
                events = remove_EcalBadCalibCrystal_events(events)

        # Photon preprocessing (same as HHbbgg L233-238):
        # zero mass/charge for vector ops, SC eta needed by scale/smearing corrections
        events["Photon"] = self.add_zero_photon_mass_and_charge(events.Photon)
        events["Photon"] = add_photon_SC_eta(events.Photon, events.PV)

        # Save raw pT before scale/smearing corrections (same as HHbbgg L284-297)
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

        # === Apply object corrections (before selection, same as HHbbgg) ===
        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"Applying correction {correction_name} to dataset {dataset_name}"
                )
                varying_function = available_object_corrections[correction_name]
                events = varying_function(
                    events=events, year=year
                )
            elif correction_name in available_weight_corrections:
                # event weight corrections are applied after selection
                continue
            else:
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        # =================================================================
        # === Photon preselection ===
        # photons passing pT > 25, |η| < 2.5, mvaID > -0.9,
        # pixel seed veto, H/E < 0.08, electron veto
        # =================================================================
        photons = events.Photon
        photons = photon_preselection(self, photons, events, year=year)
        if ak.sum(ak.num(photons) >= 2) == 0:
            return {}

        # =================================================================
        # === Diphoton candidates (store_flag fiducial, same as HHbbgg) ===
        # =================================================================
        diphotons = build_diphoton_candidates(photons, self.min_pt_lead_photon)
        diphotons = apply_fiducial_cut_det_level(self, diphotons)
        diphotons = diphotons[diphotons.pass_fiducial_classical]
        diphotons = diphotons[(diphotons.mass > 100) & (diphotons.mass < 180)]  # Higgs mass window

        has_dipho = ak.num(diphotons) > 0  # baseline for all post-diphoton cutflow

        # === Build gen-level truth matching for electrons and muons (MC only) ===
        if self.data_kind == "mc":
            gen_part = events.GenPart
            gen_status1 = gen_part.status == 1
            gen_prompt = (gen_part.statusFlags & 1) > 0  # isPrompt
            # Gen prompt electrons
            gen_e_mask = gen_status1 & gen_prompt & (abs(gen_part.pdgId) == 11)
            gen_e = gen_part[gen_e_mask]
            gen_e_eta = ak.pad_none(gen_e.eta, 1)  # pad to ≥1 per event
            gen_e_phi = ak.pad_none(gen_e.phi, 1)
            # Gen prompt muons
            gen_m_mask = gen_status1 & gen_prompt & (abs(gen_part.pdgId) == 13)
            gen_m = gen_part[gen_m_mask]
            gen_m_eta = ak.pad_none(gen_m.eta, 1)
            gen_m_phi = ak.pad_none(gen_m.phi, 1)
        else:
            gen_e_eta, gen_e_phi = None, None
            gen_m_eta, gen_m_phi = None, None

        # === Electron selection: manual baseline + tight (sequential) ===
        # =================================================================
        electrons = events.Electron
        if not hasattr(electrons, 'ScEta'):
            electrons['ScEta'] = electrons.eta

        # Build reco-gen matching flag for all electrons (manual ΔR, no metric_table)
        if self.data_kind == "mc":
            reco_e_eta = electrons.eta
            reco_e_phi = electrons.phi
            # Broadcast: [N_electrons] × [N_gen_e_padded]
            deta = reco_e_eta[:, :, None] - gen_e_eta[:, None, :]
            dphi = (reco_e_phi[:, :, None] - gen_e_phi[:, None, :] + numpy.pi) % (2 * numpy.pi) - numpy.pi
            e_dr = numpy.sqrt(deta**2 + dphi**2)
            e_is_real = ak.fill_none(ak.min(e_dr, axis=-1) < 0.2, False)
        else:
            e_is_real = ak.zeros_like(electrons.pt, dtype=bool)

        # raw leading electron (before cuts, for debugging)
        electrons["dR_pho"] = delta_r_mask(electrons, diphotons.pho_lead, 0.4)
        e_conept_raw = electrons.coneept if hasattr(electrons, "coneept") else electrons.pt
        raw_ele = electrons[(e_conept_raw >= 10.0) & electrons.dR_pho]
        raw_ele = raw_ele[ak.argsort(raw_ele.pt, ascending=False)]
        first_raw_ele = ak.firsts(raw_ele)

        e_cumul = ak.ones_like(electrons.pt, dtype=bool)
        ele_cf = {}

        # === Baseline cuts (matching select_electrons exactly) ===
        e_baseline_cuts = [
            ("pt", electrons.pt > self.electron_pt_threshold),
            ("eta", abs(electrons.eta) < self.electron_max_eta),
            ("transition_veto", ~((abs(electrons.ScEta) > 1.4442) & (abs(electrons.ScEta) < 1.566))),
            ("id", electrons.cutBased >= 2),  # self.el_id_wp == "loose"
            ("dr_lead", delta_r_mask(electrons, diphotons.pho_lead, self.electron_photon_min_dr)),
            ("dr_sublead", delta_r_mask(electrons, diphotons.pho_sublead, self.electron_photon_min_dr)),
            ("dxy", abs(electrons.dxy) < self.electron_max_dxy if self.electron_max_dxy is not None else ak.ones_like(electrons.pt, dtype=bool)),
            ("dz", abs(electrons.dz) < self.electron_max_dz if self.electron_max_dz is not None else ak.ones_like(electrons.pt, dtype=bool)),
        ]
        for name, mask in e_baseline_cuts:
            e_cumul = e_cumul & mask
            e_pass_any = ak.any(e_cumul, axis=-1) & has_dipho
            e_pass_real = ak.any(e_cumul & e_is_real, axis=-1) & has_dipho
            ele_cf[f"cf_ele_base_{name}"] = int(ak.sum(e_pass_any))
            ele_cf[f"cf_ele_base_{name}_real"] = int(ak.sum(e_pass_real))
            ele_cf[f"cf_ele_base_{name}_only_fake"] = int(ak.sum(e_pass_any & ~e_pass_real))

        # Verify manual baseline matches select_electrons function
        e_cumul_original = select_electrons(self, electrons, diphotons)
        assert ak.all(e_cumul == e_cumul_original), \
            f"Manual electron baseline differs from select_electrons! Differences: {ak.sum(e_cumul != e_cumul_original)}"

        # Apply baseline
        sel_ele = electrons[e_cumul]

        # === Tight cuts (sequential on top of baseline) ===
        e_conept = sel_ele.coneept if hasattr(sel_ele, "coneept") else sel_ele.pt
        e_deltaEtaSC = sel_ele.deltaEtaSC if hasattr(sel_ele, "deltaEtaSC") else ak.zeros_like(sel_ele.eta)
        scEta = abs(sel_ele.eta + e_deltaEtaSC)
        e_cumul_tight = ak.ones_like(sel_ele.pt, dtype=bool)
        e_is_real_tight = e_is_real[e_cumul]  # gen-match for baseline-passing electrons only
        for name, mask in [
            ("conept", e_conept >= 10.0),
            ("miniIso", sel_ele.miniPFRelIso_all <= 0.4),
            ("sip3d", sel_ele.sip3d < 8),
            ("lostHits", sel_ele.lostHits == 0),
            ("convVeto", sel_ele.convVeto),
            ("hoe", sel_ele.hoe <= 0.10),
            ("eInvMinusPInv", sel_ele.eInvMinusPInv >= -0.04),
            ("promptMVA", sel_ele.promptMVA >= 0.30),
            ("sieie", ((scEta <= 1.479) & (sel_ele.sieie <= 0.011)) |
                      ((scEta > 1.479) & (sel_ele.sieie <= 0.030))),
        ]:
            e_cumul_tight = e_cumul_tight & mask
            e_pass_any = ak.any(e_cumul_tight, axis=-1) & has_dipho
            e_pass_real = ak.any(e_cumul_tight & e_is_real_tight, axis=-1) & has_dipho
            ele_cf[f"cf_ele_{name}"] = int(ak.sum(e_pass_any))
            ele_cf[f"cf_ele_{name}_real"] = int(ak.sum(e_pass_real))
            ele_cf[f"cf_ele_{name}_only_fake"] = int(ak.sum(e_pass_any & ~e_pass_real))
        sel_ele = sel_ele[e_cumul_tight]
        n_ele = ak.num(sel_ele)

        # =================================================================
        # === Muon selection: manual baseline + tight (sequential) ===
        # =================================================================
        muons = events.Muon

        # Build reco-gen matching flag for all muons (manual ΔR, no metric_table)
        if self.data_kind == "mc":
            reco_m_eta = muons.eta
            reco_m_phi = muons.phi
            deta_m = reco_m_eta[:, :, None] - gen_m_eta[:, None, :]
            dphi_m = (reco_m_phi[:, :, None] - gen_m_phi[:, None, :] + numpy.pi) % (2 * numpy.pi) - numpy.pi
            m_dr = numpy.sqrt(deta_m**2 + dphi_m**2)
            m_is_real = ak.fill_none(ak.min(m_dr, axis=-1) < 0.2, False)
        else:
            m_is_real = ak.zeros_like(muons.pt, dtype=bool)

        # raw leading muon (before cuts, for debugging)
        muons["dR_pho"] = delta_r_mask(muons, diphotons.pho_lead, 0.4)
        mu_conept_raw = muons.coneept if hasattr(muons, "coneept") else muons.pt
        raw_mu = muons[(mu_conept_raw >= 10.0) & muons.dR_pho]
        raw_mu = raw_mu[ak.argsort(raw_mu.pt, ascending=False)]
        first_raw_mu = ak.firsts(raw_mu)

        m_cumul = ak.ones_like(muons.pt, dtype=bool)
        mu_cf = {}

        # === Baseline cuts (matching select_muons exactly) ===
        m_baseline_cuts = [
            ("pt", muons.pt > self.muon_pt_threshold),
            ("eta", abs(muons.eta) < self.muon_max_eta),
            ("id", muons.mediumId),  # self.mu_id_wp == "medium"
            ("iso", muons.pfIsoId >= 2),  # self.mu_iso_wp == "loose"
            ("global", muons.isGlobal if self.global_muon else ak.ones_like(muons.pt, dtype=bool)),
            ("dr_lead", delta_r_mask(muons, diphotons.pho_lead, self.muon_photon_min_dr)),
            ("dr_sublead", delta_r_mask(muons, diphotons.pho_sublead, self.muon_photon_min_dr)),
            ("dxy", abs(muons.dxy) < self.muon_max_dxy if self.muon_max_dxy is not None else ak.ones_like(muons.pt, dtype=bool)),
            ("dz", abs(muons.dz) < self.muon_max_dz if self.muon_max_dz is not None else ak.ones_like(muons.pt, dtype=bool)),
        ]
        for name, mask in m_baseline_cuts:
            m_cumul = m_cumul & mask
            m_pass_any = ak.any(m_cumul, axis=-1) & has_dipho
            m_pass_real = ak.any(m_cumul & m_is_real, axis=-1) & has_dipho
            mu_cf[f"cf_mu_base_{name}"] = int(ak.sum(m_pass_any))
            mu_cf[f"cf_mu_base_{name}_real"] = int(ak.sum(m_pass_real))
            mu_cf[f"cf_mu_base_{name}_only_fake"] = int(ak.sum(m_pass_any & ~m_pass_real))

        # Verify manual baseline matches select_muons function
        m_cumul_original = select_muons(self, muons, diphotons)
        assert ak.all(m_cumul == m_cumul_original), \
            f"Manual muon baseline differs from select_muons! Differences: {ak.sum(m_cumul != m_cumul_original)}"

        # Apply baseline
        sel_mu = muons[m_cumul]

        # === Tight cuts (sequential on top of baseline) ===
        mu_conept = sel_mu.coneept if hasattr(sel_mu, "coneept") else sel_mu.pt
        m_cumul_tight = ak.ones_like(sel_mu.pt, dtype=bool)
        m_is_real_tight = m_is_real[m_cumul]  # gen-match for baseline-passing muons only
        for name, mask in [
            ("conept", mu_conept >= 10.0),
            ("miniIso", sel_mu.miniPFRelIso_all <= 0.4),
            ("sip3d", sel_mu.sip3d < 8),
            ("promptMVA", sel_mu.promptMVA >= 0.5),
        ]:
            m_cumul_tight = m_cumul_tight & mask
            m_pass_any = ak.any(m_cumul_tight, axis=-1) & has_dipho
            m_pass_real = ak.any(m_cumul_tight & m_is_real_tight, axis=-1) & has_dipho
            mu_cf[f"cf_mu_{name}"] = int(ak.sum(m_pass_any))
            mu_cf[f"cf_mu_{name}_real"] = int(ak.sum(m_pass_real))
            mu_cf[f"cf_mu_{name}_only_fake"] = int(ak.sum(m_pass_any & ~m_pass_real))
        sel_mu = sel_mu[m_cumul_tight]
        n_mu = ak.num(sel_mu)

        # === Combined lepton multiplicity ===
        n_lep = n_ele + n_mu

        # =================================================================
        # === AK4 Jet selection ===
        # pT > 20, |η| < 4.7, tightLepVeto, dR cleaning vs photon only
        # =================================================================
        jets = events.Jet
        jets["jetId"] = add_jetId(jets, self.nano_version, year)  # v15 doesn't store jetId
        jets_clean = select_jets(self, jets, diphotons, sel_mu, sel_ele)
        sel_jets = jets[jets_clean]

        try:
            sel_jets = sel_jets[jetvetomap(self, sel_jets, year, logger)]
            n_jets = ak.num(sel_jets)
        except Exception:
            n_jets = ak.num(sel_jets)

        # =================================================================
        # === Z-veto: |m(e± + γ_lead) - 91.2| > 5 GeV ===
        # =================================================================
        Z_veto = ak.ones_like(n_jets, dtype=bool)
        has_ele = n_ele > 0
        if ak.any(has_ele):
            first_ele = ak.firsts(sel_ele)
            lead_pho = ak.firsts(diphotons.pho_lead)
            e_4p = ak.zip({"pt": first_ele.pt, "eta": first_ele.eta,
                           "phi": first_ele.phi, "mass": first_ele.mass},
                          with_name="Momentum4D")
            pho_4p = ak.zip({"pt": lead_pho.pt, "eta": lead_pho.eta,
                            "phi": lead_pho.phi, "mass": lead_pho.mass},
                           with_name="Momentum4D")
            megamma = (e_4p + pho_4p).mass
            Z_veto = ak.where(has_ele, abs(megamma - 91.2) > 5, Z_veto)

        # =================================================================
        # === Photon ID ===
        # =================================================================
        pho_id = (ak.firsts(diphotons.pho_lead.mvaID) > self.min_mvaid) & \
                 (ak.firsts(diphotons.pho_sublead.mvaID) > self.min_mvaid)

        # =================================================================
        # === Event categorization (from AN Note §5.2-5.4) ===
        #   Cat 0: FH — 0 lepton + ≥4 jets  (WW→qqqq)
        #   Cat 1: SL — 1 lepton             (WW→qqℓν)
        #   Cat 2: FL — >=2 leptons (FL-specific cuts saved as output, not applied here)
        # =================================================================
        cat = -1 * ak.ones_like(n_jets, dtype=numpy.int32)  # -1 = unassigned

        # Cat 0: FH = 0 lepton + >=4 jets
        cat_fh = (n_lep == 0) & (n_jets >= 4)
        cat = ak.where(cat_fh, 0, cat)

        # Cat 1: SL = 1 lepton
        cat_sl = (n_lep == 1)
        cat = ak.where(cat_sl, 1, cat)

        # Cat 2: FL = >=2 leptons (tight cuts applied)
        # FL-specific cuts (MET, pT(gg), Zll veto, ...) are NOT applied here;
        # they are saved as output variables for later optimization.
        cat_fl_base = (n_lep >= 2) & has_dipho
        cat_fl = cat_fl_base
        fl_cf = {}

        # Compute FL diagnostic variables for output (not used as cuts)
        fl_out = {}
        if ak.any(cat_fl_base):
            all_lep = ak.concatenate([sel_ele, sel_mu], axis=1)
            all_lep = all_lep[ak.argsort(all_lep.pt, ascending=False)]
            lead_lep = ak.firsts(all_lep)
            sublead_lep = ak.firsts(all_lep[..., 1:])
            lead_4v = ak.zip({"pt": lead_lep.pt, "eta": lead_lep.eta,
                              "phi": lead_lep.phi, "mass": lead_lep.mass},
                             with_name="Momentum4D")
            sublead_4v = ak.zip({"pt": sublead_lep.pt, "eta": sublead_lep.eta,
                                 "phi": sublead_lep.phi, "mass": sublead_lep.mass},
                                with_name="Momentum4D")

            btag_medium_wp = {"2022preEE": 0.3040, "2022postEE": 0.3040,
                              "2023preBPix": 0.3040, "2023postBPix": 0.3040}
            btag_cut = btag_medium_wp.get(year, 0.3040)

            fl_out["fl_met_pt"] = events.PuppiMET.pt
            fl_out["fl_dipho_pt"] = ak.fill_none(ak.firsts(diphotons.pt), -999)
            fl_out["fl_mll"] = ak.fill_none((lead_lep + sublead_lep).mass, -999)
            fl_out["fl_lead_lep_pt"] = ak.fill_none(lead_lep.pt, -999)
            fl_out["fl_sublead_lep_pt"] = ak.fill_none(sublead_lep.pt, -999)
            fl_out["fl_drll"] = ak.fill_none(lead_4v.deltaR(sublead_4v), -999)
            fl_out["fl_has_btag"] = ak.fill_none(ak.any(sel_jets.btagDeepFlavB > btag_cut, axis=-1), False)

        cat = ak.where(cat_fl, 2, cat)

        # =================================================================
        # === Final preselection ===
        # =================================================================
        presel = pho_id & Z_veto & (cat >= 0)

        # drop events without a preselected diphoton candidate (same as HHbbgg L1550)
        sel_none = ~ak.is_none(ak.firsts(diphotons))
        mask = presel & sel_none
        diphotons = diphotons[mask]

        # =================================================================
        # === Dump output ===
        # =================================================================
        first_diphoton = ak.firsts(diphotons)
        cat_mask = cat[mask]

        if self.output_location is not None:
            akarr = diphoton_ak_array(self, first_diphoton)

            akarr["n_ele"] = n_ele[mask]
            akarr["n_mu"] = n_mu[mask]
            akarr["n_lep"] = n_lep[mask]
            akarr["n_jets"] = n_jets[mask]
            akarr["category"] = cat_mask

            # === Event weights (same architecture as HHbbgg) ===
            if self.data_kind == "mc":
                event_weights = Weights(size=len(events[mask]), storeIndividual=True)
                event_weights._weight = ak.to_numpy(events.genWeight[mask])

                # Weight corrections (SFs) applied after selection
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        varying_function = available_weight_corrections[correction_name]
                        event_weights = varying_function(
                            events=events[mask],
                            photons=first_diphoton,
                            muons=sel_mu[mask],
                            electrons=sel_ele[mask],
                            jets=sel_jets[mask],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=year,
                            bTagEffFileName="WWgg",
                        )

                weight_arr = ak.Array(event_weights.weight())
                akarr["weight"] = weight_arr
                akarr["weight_central"] = weight_arr / events.genWeight[mask]
            else:
                akarr["weight"] = ak.ones_like(first_diphoton.pt)
                akarr["weight_central"] = ak.ones_like(first_diphoton.pt)

            # FL output variables (for post-hoc cut optimization)
            for k, v in fl_out.items():
                akarr[k] = ak.fill_none(v[mask], -999)

            # ---- raw leading lepton (pre-cut, for debugging) ----
            first_raw_ele_m = ak.firsts(raw_ele[mask])
            first_raw_mu_m = ak.firsts(raw_mu[mask])
            for var in ["pt", "eta", "phi", "dxy", "dz", "mvaIso_WP80", "mvaIso_WP90", "cutBased"]:
                if hasattr(first_raw_ele_m, var):
                    akarr[f"raw_ele_{var}"] = ak.fill_none(getattr(first_raw_ele_m, var), -999)
            for var in ["pt", "eta", "phi", "dxy", "dz", "tightId", "pfIsoId", "isGlobal"]:
                if hasattr(first_raw_mu_m, var):
                    akarr[f"raw_mu_{var}"] = ak.fill_none(getattr(first_raw_mu_m, var), -999)

            # ---- leading electron (for cut validation) ----
            ele_masked = sel_ele[mask]
            first_ele = ak.firsts(ele_masked)
            for var in ["pt", "eta", "phi", "mass", "dxy", "dz",
                        "mvaIso_WP80", "mvaIso_WP90", "cutBased",
                        "pfRelIso03_all", "mvaIso"]:
                if hasattr(first_ele, var):
                    akarr[f"lead_ele_{var}"] = ak.fill_none(getattr(first_ele, var), -999)

            # ---- leading muon (for cut validation) ----
            mu_masked = sel_mu[mask]
            first_mu = ak.firsts(mu_masked)
            for var in ["pt", "eta", "phi", "mass", "dxy", "dz",
                        "tightId", "mediumId", "looseId",
                        "pfIsoId", "isGlobal", "pfRelIso04_all",
                        "tunepRelPt"]:
                if hasattr(first_mu, var):
                    akarr[f"lead_mu_{var}"] = ak.fill_none(getattr(first_mu, var), -999)

            # ---- Z-veto variable ----
            Z_megamma = ak.ones_like(first_diphoton.pt) * -999
            has_ele_mask = ak.fill_none(n_ele[mask] > 0, False)
            if ak.any(has_ele_mask):
                ele_4v = ak.zip({"pt": first_ele.pt, "eta": first_ele.eta,
                                 "phi": first_ele.phi, "mass": first_ele.mass},
                                with_name="Momentum4D")
                pho_4v = ak.zip({"pt": first_diphoton.pho_lead.pt,
                                 "eta": first_diphoton.pho_lead.eta,
                                 "phi": first_diphoton.pho_lead.phi,
                                 "mass": first_diphoton.pho_lead.mass},
                                with_name="Momentum4D")
                Z_megamma = ak.fill_none(ak.where(has_ele_mask, (ele_4v + pho_4v).mass, -999), -999)
            akarr["Z_megamma"] = Z_megamma

            akarr = akarr[
                [field for field in akarr.fields if "lead_fixedGridRhoAll" not in field]
            ]

            fname = apply_naming_convention(self, events)
            metadata = {}
            if self.data_kind == "mc":
                metadata["sum_genw_presel"] = str(sum_genw_presel)
                metadata["sum_weight_central"] = str(
                    ak.sum(event_weights.weight())
                )
            else:
                metadata["sum_genw_presel"] = "Data"
            subdirs = []
            if "dataset" in events.metadata:
                subdirs.append(events.metadata["dataset"])
            subdirs.append("nominal")

            dump_ak_array(self, akarr, fname, self.output_location, metadata, subdirs)

        return {
            "WWgg": {
                "n_input": int(n_total_input),
                "cf_2photon": int(ak.sum(ak.num(photons) >= 2)),
                "cf_diphoton": int(ak.sum(has_dipho)),
                **ele_cf,
                **mu_cf,
                **fl_cf,
                "cf_jet_raw": int(ak.sum((ak.num(sel_jets) > 0) & has_dipho)),
                "cf_haslep": int(ak.sum((n_lep > 0) & has_dipho)),
                "cf_zveto": int(ak.sum(Z_veto & has_dipho)),
                "cf_phoid": int(ak.sum(pho_id & has_dipho)),
                "cf_fl_base": int(ak.sum(cat_fl_base)),
                "n_events": int(ak.sum(mask)),
                "cf_fh": int(ak.sum(cat_mask == 0)),
                "cf_sl": int(ak.sum(cat_mask == 1)),
                "cf_fl": int(ak.sum(cat_mask == 2)),
            }
        }
