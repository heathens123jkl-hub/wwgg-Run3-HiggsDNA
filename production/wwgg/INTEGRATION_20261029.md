# WWgg 2024 corrections and veto integration

This release is integrated code with small-cohort runtime validation completed
on lxplus as of 2026-10-30 (scope and remaining omissions below).
Old configs retain their original opt-in state.
Data luminosity labels remain 109.95 fb^-1; no normalization or luminosity value
is changed by this patch.

## Configuration and ordering

`corrections_2024.json` is passed as `wwgg_corrections` to `WWggProcessor`.
The normal analysis JSON also accepts this dictionary through `**analysis`.
Enable `--doFlow-corrections` separately in the standard runner for MC.
Use `analysis_mc_2024.json` as the explicit MC validation configuration.

1. Data certification, shared event-level jet veto map and existing filters.
   Generator sum is computed before the jet veto map. A failed lumi mask now
   raises an error instead of silently admitting uncertified Data.
2. Existing registered photon energy corrections, then V5 JEC / JRV2 JER and
   Type-1 PUPPI MET when `jme` is enabled.
3. MC photon Flow in float64, recompute MVA, then existing photon selection.
4. Existing electron/muon/jet selection; tau object/pair tests and b veto are
   evaluated on the saved diphoton candidate and selected leptons/jets.
5. Existing photon/trigger/pileup weights and the explicit lepton components.

## Implemented changes

| Item | Implementation |
|---|---|
| Photon Flow | Shared HHbbgg helper, optional precision argument; WWgg selects float64. Original MVA and input variables preserved. No change to other workflows' default precision. |
| Photon SF | Existing LoosePhoIDSF, ElectronVetoSF and PreselSF use corrected inputs; TriggerSF nominal uses raw R9, as in the existing workflow. 2024 CSEV is preliminary, accepted provisionally by the user. |
| JEC | Summer24Prompt24_V5; rawFactor applied once; corrected mass and pt, jets reordered by corrected pt. |
| JER | Summer24Prompt24_JRV2; separate relative SFUncertainty, hybrid gen matching/stochastic smearing; same central inputs and random draws for variations. Default nonpositive policy: CMSSW 0.01 GeV energy floor. Alternative zero/unity policies remain configurable. |
| Forward JER | No exemption by default. If enabling the legacy unmatched 2.5<abs(eta)<3 exemption, also request `wwgg_selection.jet_pt_scheme=hhbbgg_2024`. Baseline jet selection otherwise retained. |
| JES | Eleven regrouped sources, correct RelativeBal mapping, signed shifts retained. Each configured variation reruns selection and features in a separate job/output directory. |
| Type-1 MET | Rebuilt from RawPuppiMET and muon-subtracted Jet/CorrT1METJet with matching V5 full and L1 calibrations. JES propagated. JER propagation deliberately disabled (optional prescription). |
| Jet ID | Matching bundled jetid.json.gz; raw fractions preserved. Existing shared jet veto map implementation retained. |
| Electrons | Bundled central 2024Prompt reco SF (including Below20) and cut-based Loose ID SF. |
| Muons | Bundled central 2024 muon_Z MediumID and LoosePFIso-over-MediumID SFs. |
| Outputs | Actual SF components, lepton up/down factors, veto flags/counts, corrected MET, configuration and payload hashes. No division by zero generator weight when forming weight_central. |

## b veto

2024 Tight UParT: btagUParTAK4B >= 0.4648 on existing selected jets with
pt > 30 GeV and abs(eta) < 2.5; reject if at least one is tagged.
It is not the partner analysis's PNet definition. No new jet cleaning is added.
Invalid tagger values in the accepted kinematic region raise an error.

## tau veto

Configuration: `tau_veto_2024.json`, identical to the tested initial DeepTau
definition, not an automatic choice of the S/B-optimal scan point.

- Stored Tau pt > 20 GeV, abs(eta) < 2.5, abs(dz) < 0.2, unit charge.
- Exclude decayMode 5 and 6; require deltaR(tau, each selected photon) > 0.2.
- Pair separation > 0.5; no opposite-sign requirement by default.
- NanoAODv15 ordinal WP thresholds (not bit masks):

| Pair | VSjet | VSe | VSmu |
|---|---:|---:|---:|
| e tau_h | 4 | 6 | 1 |
| mu tau_h | 4 | 2 | 4 |
| tau_h tau_h | 4 | 2 | 1 |

Use existing WWgg-selected electrons and muons. Reject the union of these
three pair definitions in every reconstructed category. A single tau alone
does not trigger the tau-pair veto. This is not a proof of full orthogonality
to the partner analysis.

Both vetoes support off/store_only/apply. Defaults in the new integration
configuration are apply. The paired validation first stores flags without
rejecting, then checks that apply selects exactly the expected subset.

## Explicit omissions and interpretation

- No matched WWgg b-veto or tau-veto efficiency SF yet. Do not treat these as
  already corrected; the untagged probability also needs an efficiency map.
- Lepton reco/ID/PF-isolation components do not correct the full extra
  prompt-MVA, mini-isolation, impact parameter and global-muon selection.
  Muon low/high-pt source prescriptions still need confirmation.
- No unsupported 2024 XY/recoil correction, guessed photon-MVA nominal POG SF,
  or unverified electron/muon momentum correction is added.
- The new nominal Flow uses the existing meta feature ordering. Its external
  training provenance remains as documented in the previous pilot.
- No generic object/weight systematics loop is silently claimed. Nonempty
  generic `systematics` raises an error. Supported explicit variation jobs are
  the eleven JES pairs, JER_up/down and four lepton component up/down pairs.
  Photon/trigger/pileup uncertainties and a complete statistical nuisance model
  still need implementation/validation.
- Type-1 has numerical checks against the supplied equations, not full
  production CMSSW closure. Small-cohort integrated Data and MC checks passed;
  this does not establish full physics closure or a complete uncertainty model.

## Runtime validation reviewed on 2026-10-30

- Three signal MC samples, 200 input events each: all 75 modes (nominal,
  22 JES variations and two JER variations per sample) passed. Downloaded
  artifacts were independently checked for hashes, weight products and duplicate
  event keys. Nominal selected counts summed over reconstructed categories:
  FH sample 51, SL sample 37, FL sample 58; variation ranges 49-53, 37-37,
  and 56-58. Generator denominators were unchanged. JES shifts propagated to
  MET; JER-only modes left MET unchanged as configured.
- FH Data: selected sidebands in 2024C gave baseline/store/apply counts 2/3/3;
  2024I gave 1/1/1. All four integrated events are FH. The extra 2024C event
  is 379416:178:192387902, mass 147.9404 GeV, four selected jets and fourth-jet
  pt 21.087 GeV. Its baseline rejected-object state was not exported, so its
  precise migration cause is not established by these output tables.
- SL Data: 2024I event 386554:304:664372173, mass 111.8671 GeV, one electron,
  no selected muon or analysis jet. Baseline/store/apply all retain this event;
  weight is 1 and both veto flags are false. MET changes from 17.36237 to
  16.79462 GeV with the integrated correction, identically in store/apply.
  No selected analysis jets does not mean no Type-1 eligible jets.
- Data tables contain only sidebands; weights are unity, no MC SF component
  columns are present, and applied-veto output matches stored-flag filtering.
  The runtime checker also tested disabled Data photon Flow and unity JER.
- The Data cohorts were selected to exercise the software path, not measure
  unbiased selection/veto efficiencies. They do not exercise every veto
  rejection branch or every lepton flavor. FL Data was not separately pursued.
- Per user instruction, do not expand statistics or rerun already verified
  samples merely to broaden these tests. Remaining correction prescriptions
  and systematic uncertainties are listed above, not implied to be complete.

Local independent review records (relative to workspace root):
`outputs/wwgg_data_jme_review_20261029/independent_review.json`,
`outputs/wwgg_data_review_20261030/independent_review.json`,
`outputs/wwgg_sl_data_review_20261030/independent_review.json`.

## Source files

- `higgs_dna/workflows/WWgg.py`: workflow calls, selection, flags, weights, metadata.
- `higgs_dna/tools/wwgg_corrections.py`: options, JME adapter, lepton SF, veto routing.
- `higgs_dna/tools/wwgg_jec_type1.py`, `wwgg_jer_math.py`: tested numerical cores.
- `higgs_dna/tools/wwgg_tau_veto.py`, `wwgg_b_veto.py`: tested veto definitions.
- `higgs_dna/tools/flow_corrections.py`: optional float64 evaluation.
- `higgs_dna/systematics/JSONs/WWgg/2024/`: pinned JME and lepton payloads.
- `production/wwgg/validate_integration.py`: same-cohort combined runtime checks.

Run the four focused test files before the small-cohort runtime test. The
Windows checks exercise wiring/numerical cores, not the unavailable full
coffea/torch/correctionlib production environment.
