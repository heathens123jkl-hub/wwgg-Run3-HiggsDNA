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

import awkward as ak
import numpy
import vector
import logging

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

        # === Filters & triggers ===
        events = self.apply_filters_and_triggers(events)

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

        # =================================================================
        # === Electron selection (loose cutBased, pT > 10, |η| < 2.5) ===
        # =================================================================
        electrons = events.Electron
        if not hasattr(electrons, 'ScEta'):
            electrons['ScEta'] = electrons.eta

        # raw leading electron (before cuts, for debugging)
        electrons["dR_pho"] = delta_r_mask(electrons, diphotons.pho_lead, 0.4)
        e_conept_raw = electrons.coneept if hasattr(electrons, "coneept") else electrons.pt
        raw_ele = electrons[(e_conept_raw >= 10.0) & electrons.dR_pho]
        raw_ele = raw_ele[ak.argsort(raw_ele.pt, ascending=False)]
        first_raw_ele = ak.firsts(raw_ele)

        sel_ele = electrons[select_electrons(self, electrons, diphotons)]

        # Tight electron cuts (AN2025_108 Table 13)
        # ID (cutBased>=2) already applied by select_electrons
        # Apply additional kinematics, isolation, quality cuts
        e_conept = sel_ele.coneept if hasattr(sel_ele, "coneept") else sel_ele.pt
        e_deltaEtaSC = sel_ele.deltaEtaSC if hasattr(sel_ele, "deltaEtaSC") else ak.zeros_like(sel_ele.eta)
        scEta = abs(sel_ele.eta + e_deltaEtaSC)
        sel_ele = sel_ele[
            (e_conept >= 10.0) &
            (sel_ele.miniPFRelIso_all <= 0.4) &
            (sel_ele.sip3d < 8) &
            (sel_ele.lostHits == 0) &
            (sel_ele.convVeto) &
            (sel_ele.hoe <= 0.10) &
            (sel_ele.eInvMinusPInv >= -0.04) &
            (sel_ele.promptMVA >= 0.30) &
            (((scEta <= 1.479) & (sel_ele.sieie <= 0.011)) |
             ((scEta > 1.479) & (sel_ele.sieie <= 0.030)))
        ]
        n_ele = ak.num(sel_ele)
        n_ele_post_pho = n_ele

        # =================================================================
        # === Muon selection (tight ID, loose iso, global, pT > 10) ===
        # =================================================================
        muons = events.Muon

        # raw leading muon (before cuts, for debugging)
        muons["dR_pho"] = delta_r_mask(muons, diphotons.pho_lead, 0.4)
        mu_conept_raw = muons.coneept if hasattr(muons, "coneept") else muons.pt
        raw_mu = muons[(mu_conept_raw >= 10.0) & muons.dR_pho]
        raw_mu = raw_mu[ak.argsort(raw_mu.pt, ascending=False)]
        first_raw_mu = ak.firsts(raw_mu)

        sel_mu = muons[select_muons(self, muons, diphotons)]

        # Tight muon cuts (AN2025_108 Table 14)
        # ID (mediumId) already applied by select_muons
        mu_conept = sel_mu.coneept if hasattr(sel_mu, "coneept") else sel_mu.pt
        sel_mu = sel_mu[
            (mu_conept >= 10.0) &
            (sel_mu.miniPFRelIso_all <= 0.4) &
            (sel_mu.sip3d < 8) &
            (sel_mu.promptMVA >= 0.5)
        ]
        n_mu = ak.num(sel_mu)
        n_mu_post_pho = n_mu

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
        #   Cat 2: FL — ≥2 leptons + MET>20 + pT(γγ)>91
        #               + Z→ll veto + b-veto  (WW→ℓνℓν)
        # =================================================================
        cat = -1 * ak.ones_like(n_jets, dtype=numpy.int32)  # -1 = unassigned

        # Cat 0: FH = 0 lepton + >=4 jets
        cat_fh = (n_lep == 0) & (n_jets >= 4)
        cat = ak.where(cat_fh, 0, cat)

        # Cat 1: SL = 1 lepton
        cat_sl = (n_lep == 1)
        cat = ak.where(cat_sl, 1, cat)

        # Cat 2: FL = >=2 leptons + MET>20 + pT(γγ)>91 + Z→ll veto + b-veto
        cat_fl_base = (n_lep >= 2)
        cat_fl = cat_fl_base
        if ak.any(cat_fl_base):
            # MET > 20 GeV
            met_pt = events.MET.pt if hasattr(events, 'MET') else ak.zeros_like(n_jets)
            cat_fl = cat_fl & (met_pt > 20)

            # pT(γγ) > 91 GeV (Table 34)
            dipho_pt = ak.fill_none(ak.firsts(diphotons.pt), 0)
            cat_fl = cat_fl & (dipho_pt > 91)

            # Z→ll veto: m(ll) < 80 or > 100 (Table 34)
            all_lep = ak.concatenate([sel_ele, sel_mu], axis=1)
            all_lep = all_lep[ak.argsort(all_lep.pt, ascending=False)]
            lead_lep = ak.firsts(all_lep)
            sublead_lep = ak.firsts(all_lep[..., 1:])
            mll = (lead_lep + sublead_lep).mass
            cat_fl = cat_fl & ak.fill_none((mll < 80) | (mll > 100), True)

            # FL lepton pT + dR requirements (Table 34)
            cat_fl = cat_fl & ak.fill_none(lead_lep.pt > 20, False)
            cat_fl = cat_fl & ak.fill_none(sublead_lep.pt > 10, False)
            lead_4v = ak.zip({"pt": lead_lep.pt, "eta": lead_lep.eta,
                              "phi": lead_lep.phi, "mass": lead_lep.mass},
                             with_name="Momentum4D")
            sublead_4v = ak.zip({"pt": sublead_lep.pt, "eta": sublead_lep.eta,
                                 "phi": sublead_lep.phi, "mass": sublead_lep.mass},
                                with_name="Momentum4D")
            cat_fl = cat_fl & ak.fill_none(lead_4v.deltaR(sublead_4v) > 0.4, False)

            # b-veto: no jet with DeepFlavour b-score > medium WP
            # medium WP per year (from AN Note §4.4)
            btag_medium_wp = {"2022preEE": 0.3040, "2022postEE": 0.3040,
                              "2023preBPix": 0.3040, "2023postBPix": 0.3040}
            btag_cut = btag_medium_wp.get(year, 0.3040)
            has_btag = ak.any(sel_jets.btagDeepFlavB > btag_cut, axis=-1)
            cat_fl = cat_fl & ~ak.fill_none(has_btag, False)

        cat = ak.where(cat_fl, 2, cat)

        # Save FL diagnostic flags (for events with n_lep>=2, which cut failed?)
        fl_diag = {}
        if ak.any(cat_fl_base):
            fl_diag["fl_has2lep"] = cat_fl_base
            fl_diag["fl_pass_met"] = (met_pt > 20)
            fl_diag["fl_pass_dipt"] = (dipho_pt > 91)
            fl_diag["fl_pass_mll"] = ak.fill_none((mll < 80) | (mll > 100), False)
            fl_diag["fl_pass_leadpt"] = ak.fill_none(lead_lep.pt > 20, False)
            fl_diag["fl_pass_subpt"] = ak.fill_none(sublead_lep.pt > 10, False)
            fl_diag["fl_pass_drll"] = ak.fill_none(lead_4v.deltaR(sublead_4v) > 0.4, False)
            fl_diag["fl_pass_btag"] = ~ak.fill_none(has_btag, True)
            fl_diag["fl_pass_all"] = cat_fl

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
            # cut-flow counters
            akarr["n_ele_post_pho"] = n_ele_post_pho[mask]
            akarr["n_mu_post_pho"] = n_mu_post_pho[mask]
            akarr["n_lep"] = n_lep[mask]
            akarr["n_jets"] = n_jets[mask]
            akarr["category"] = cat_mask

            # FL diagnostic flags
            for k, v in fl_diag.items():
                akarr[k] = ak.fill_none(v[mask], -1)

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
            subdirs = []
            if "dataset" in events.metadata:
                subdirs.append(events.metadata["dataset"])
            subdirs.append("nominal")

            dump_ak_array(self, akarr, fname, self.output_location, metadata, subdirs)

        return {
            "WWgg": {
                "n_input": int(n_total_input),
                "n_events": int(ak.sum(mask)),
                "n_fh": int(ak.sum(cat_mask == 0)),
                "n_sl": int(ak.sum(cat_mask == 1)),
                "n_fl": int(ak.sum(cat_mask == 2)),
            }
        }
