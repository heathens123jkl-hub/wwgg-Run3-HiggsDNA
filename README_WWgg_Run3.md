# Run3 HH→WWγ Preselection — Setup & Run Guide

## Collaboration synchronization (2026-09-24)

See the [short update and usage guide](production/crosscheck_20260924/README.md)
for the new output variables, JSON selection switches and 2024C submission command.

## Output feature extension (2026-09-23; small-sample lxplus checks completed)

`WWgg.py` now calls the output-only `workflows/wwgg_features.py` module on the
selected candidate and selected objects. It exports fixed-schema photon,
lepton, jet, MET, FL cut-study and separate truth diagnostics. Missing values
are NaN with explicit validity flags. Existing selection and weights are not
changed by this module. See the canonical optimization tool's
`HIGGSDNA_FEATURES.md` for the variable definitions, installer and tests.
Do not use all output columns as ML inputs: mass, truth and provenance need
to remain outside the classifier input whitelist.

## Selection-study extension (2026-09-22; full-production validation pending)

The original selection remains the default. A copy of the analysis JSON can
contain a top-level `wwgg_selection` object with `z_veto` and
`diphoton_pt_over_mass`, each set to `apply` or `store_only`.
Setting both to `store_only` retains events rejected by these cuts while
preserving absolute photon pT, ID, isolation, eta and mass cuts.
Output includes baseline cut flags, event IDs, selection modes and candidate
list diagnostics. Relaxing candidate cuts can change object cleaning and
categories; reapplying old cuts to the stored candidate is not an exact replay.

The existing `wwgg_tools/selection_optimization/` tool contains the reviewed
installer and `HIGGSDNA_STUDY.md`. Test the four modes on identical small inputs
first. Full production continues to use the native command
`run_analysis.py --executor vanilla_lxplus --queue workday`; no separate
handwritten Condor submission is needed. Use new output directories and
preserve the actual production corrections and trigger settings.

Follow these steps. You should have a working test job in ~10 minutes.

---

## 1. Clone & Environment

```bash
cd your_own_EOS_directory
git clone https://github.com/heathens123jkl-hub/wwgg-Run3-HiggsDNA.git wwgg_Run3
cd wwgg_Run3

# Create conda environment
conda create -n wwgg-run3 python=3.12 -y
conda activate wwgg-run3

# Install all required packages
pip install coffea awkward uproot vector correctionlib scipy==1.14.0 pyarrow matplotlib rich dask distributed bokeh pyyaml jinja2

# Install HiggsDNA itself as a package
pip install -e . --no-deps
```

## 2. Download Correction / SF JSONs

The workflow now uses the full correction & weight architecture (aligned with the HHbbgg workflow). The following JSONs are **not** in the repo and must be downloaded with `pull_files.py`:

```bash
python3 higgs_dna/scripts/pull_files.py --target GoldenJSON   # lumi masks (needed for data)
python3 higgs_dna/scripts/pull_files.py --target SS-IJazZ     # photon scale/smearing (Scale2G_IJazZ, Smearing2G_IJazZ)
python3 higgs_dna/scripts/pull_files.py --target PU           # pileup weights
python3 higgs_dna/scripts/pull_files.py --target TriggerSF    # trigger scale factors
python3 higgs_dna/scripts/pull_files.py --target LooseMva     # loose photon ID SF  → JSONs/LoosePhoIDSF/
python3 higgs_dna/scripts/pull_files.py --target eVetoSF      # electron veto SF     → JSONs/ElectronVetoSF/
python3 higgs_dna/scripts/pull_files.py --target PreselSF     # preselection SF      → JSONs/Preselection/
```

The POG JSONs (jet ID, b-tagging, EGM) are bundled in `higgs_dna/systematics/JSONs/POG/`. No separate download needed.

## 3. Grid Proxy (for XRootD)

```bash
voms-proxy-init --voms cms -valid 192:00
```

## 4. Run Signal MC (with full corrections)

```json
{
  "samplejson": "cf_SL_samples.json",
  "workflow": "WWgg",
  "metaconditions": "Era2023_postBPix_v1",
  "taggers": [],
  "split_mc": false,
  "year": {"WWgg_SL_2024": ["2024"]},
  "corrections": {"WWgg_SL_2024": ["Smearing2G_IJazZ", "TriggerSF", "Pileup", "LoosePhoIDSF", "ElectronVetoSF", "PreselSF"]},
  "systematics": {}
}
```

```bash
python3 higgs_dna/scripts/run_analysis.py --json-analysis cf_SL.json --nano-version 15 --executor iterative --timeout 300 --skipbadfiles --dump ./output_SL --save hists_SL.coffea
```

MC weight output:
- `weight` = genWeight × TriggerSF × Pileup × LoosePhoIDSF × ElectronVetoSF × PreselSF
- `weight_central` = the SF product (weight / genWeight)
- metadata `sum_genw_presel` = sum of genWeights before selection (for lumi normalization: `weight_norm = xsec × lumi / sum_genw_presel`)

**Note:** MC must NOT contain `Scale2G_IJazZ` (that correction is data-only — the code raises an error if applied to MC). Data must NOT contain smearing or any weight SFs.

## 5. Run Data (2024)

```json
{
  "samplejson": "samples_data_2024_Run2024G.json",
  "workflow": "WWgg",
  "metaconditions": "Era2023_postBPix_v1",
  "taggers": [],
  "split_mc": false,
  "year": {"Run2024G_EG0": ["2024"], "Run2024G_EG1": ["2024"]},
  "corrections": {"Run2024G_EG0": ["Scale2G_IJazZ"], "Run2024G_EG1": ["Scale2G_IJazZ"]},
  "systematics": {}
}
```

```bash
python3 higgs_dna/scripts/run_analysis.py --json-analysis data_2024G.json --nano-version 15 --executor iterative --timeout 300 --skipbadfiles --dump ./output_data --save hists_data.coffea
```

Data-specific handling (automatic):
- Golden JSON lumi mask (requires `--target GoldenJSON` above)
- EcalBadCalibCrystal event removal
- Trigger bits applied (data is recorded by trigger; MC uses TriggerSF weight instead)
- `Scale2G_IJazZ` photon energy scale correction
- Output `weight = 1`

## 6. Check Output

```bash
python3 -c "
import awkward as ak, glob
fs = sorted(glob.glob('output_SL/*/nominal/*.parquet'))
e = ak.from_parquet(fs[0])
e = e[~ak.is_none(e.mass)]
t = len(e)
print(f'Events: {t}')
print(f'  FH(0): {int(ak.sum(e.category==0))}  SL(1): {int(ak.sum(e.category==1))}  FL(2): {int(ak.sum(e.category==2))}')
print(f'  weight_central mean: {ak.mean(e.weight_central):.3f}')
"
```

## 7. Cutflow with Gen-Truth Matching

```bash
python3 read_cutflow.py hists_SL.coffea
```

Every electron/muon cut shows three columns: total events, events with a real lepton (ΔR<0.2 matched to a gen prompt lepton), and events with only fake leptons. Real/fake columns are zero for data (no GenPart).

## 8. Use Other Samples

Pre-fetched 2024 samples are in the repo:

| File | Content |
|------|---------|
| `samples_2024_signal.json` | HH→WWγγ SM signal (FH/SL/FL), 276 files |
| `samples_2024_background.json` | Background (GGJets, TTGG, single-H), 1,561 files |

Keys: `WWgg_FH_2024`, `WWgg_SL_2024`, `WWgg_FL_2024` (signal); `GGJets_MGG-40to80_2024`, etc. (background).

---

## Code Architecture

```
higgs_dna/
├── workflows/WWgg.py          ← Main processor (all selection logic lives here)
├── selections/
│   ├── photon_selections.py   ← photon_preselection()
│   ├── diphoton_selections.py ← build_diphoton_candidates()
│   ├── lepton_selections.py   ← select_electrons(), select_muons()
│   └── jet_selections.py      ← select_jets(), jetvetomap
├── tools/jetID.py             ← add_jetId() (required for v15 NanoAOD)
└── scripts/run_analysis.py    ← Entry point: reads JSON config, runs the workflow
```

**Process flow (aligned with HHbbgg workflow):**

1. `run_analysis.py` reads JSON config → creates a `WWggProcessor`
2. `process(events)`:
   - lumi mask (data) → sum_genw_presel (MC) → read corrections
   - MET filters & triggers (data: HLT bits; MC: TriggerSF weight instead)
   - EcalBadCalibCrystal removal (data) → photon zero mass + SC eta → save pt_raw
   - object corrections (Smearing2G_IJazZ for MC, Scale2G_IJazZ for data)
   - photon preselection → diphoton + fiducial + mass window
   - gen-truth matching setup (MC only)
   - electron/muon baseline (manual, per-cut cutflow) + tight cuts
   - jet selection → Z-veto → photon ID → event categories → preselection mask
   - Weights container: genWeight × weight corrections (SFs) → `weight`, `weight_central`
3. Parquet dump + cutflow counters

**Framework-level code (corrections, weights, dumping) is copied from the HHbbgg workflow** — only the physics selections (cut values, categories, lepton cuts, gen-truth matching) are WWgg-specific.

---

## Selection Conditions

All cuts follow CMS HH→WWγγ analysis standards (AN-2020-165) and the HH→bbγγ Run 3 object definitions, with tight lepton ID taken from AN2025_108.

### Photons

| Cut | Value | Why |
|-----|-------|-----|
| pT | > 25 GeV | Suppress soft QCD photons |
| Supercluster η | Barrel (EB) or Endcap (EE) only | Exclude forward region with poor resolution |
| mvaID | > -0.9 | BDT photon ID: separates real photons from jet fakes |
| H/E | < 0.08 | Real photons deposit almost all energy in ECAL; jets leak into HCAL |
| R9 / isolation | R9 > 0.8 OR charged iso×pT < 20 OR relIso < 0.3 | Converted/fake photons with low R9 must be isolated |
| Shower shape (4 branches) | EB/EE × hiR9/loR9 | CMS cut-based photon ID categories |
| electronVeto | == 1 | Rejects superclusters matched to a pixel seed (likely electrons) |

### Diphoton Pair

| Cut | Value | Why |
|-----|-------|-----|
| Lead pT | > 35 GeV | H→γγ trigger threshold |
| pT/mγγ | lead > 1/3, sublead > 1/4 | Suppress mismeasured mγγ |
| \|η\| | < 2.5 (both) | Barrel/endcap acceptance |
| pfRelIso03 × pT | < 10 (both) | HLT-mimicking isolation |
| m(γγ) | ∈ [100, 180] GeV | Higgs mass window |

### Electrons — Tight (AN2025_108 Table 13)

Baseline (`select_electrons`-equivalent, manual per-cut implementation with cutflow):
pT > 10, |η| < 2.5 + transition veto, cutBased ≥ 2, dR(γ) > 0.4 ×2, |dxy| < 0.05, |dz| < 0.1

Tight (sequential):
conept ≥ 10, miniPFRelIso ≤ 0.4, sip3d < 8, lostHits == 0, convVeto, H/E ≤ 0.10, 1/E−1/p ≥ −0.04, promptMVA ≥ 0.30, σiηiη ≤ 0.011 (EB) / 0.030 (EE)

### Muons — Tight (AN2025_108 Table 14)

Baseline: pT > 10, |η| < 2.4, mediumId, pfIsoId ≥ 2, isGlobal, dR(γ) > 0.4 ×2, |dxy| < 0.05, |dz| < 0.1

Tight (sequential): conept ≥ 10, miniPFRelIso ≤ 0.4, sip3d < 8, promptMVA ≥ 0.5

### AK4 Jets (aligned with HH→bbγγ Run 3)

| Cut | Value |
|-----|-------|
| pT | > 20 GeV |
| \|η\| | < 4.7 |
| jetId | tightLepVeto (== 6) |
| dR(jet, γ) | > 0.4 |

No dR cleaning of jets against leptons (HH→bbγγ convention).

### Event-level

| Cut | Value |
|-----|-------|
| Z-veto | \|m(e₁+γ_lead) − 91.2\| > 5 GeV |
| Photon ID | lead/sub mvaID > -0.9 |
| MET filters | flag_goodVertices, HBHENoise, etc. |

### Event Categories (AN-2020-165 §5.2-5.4)

| Category | Requirement | W→WW decay |
|----------|-------------|------------|
| **FH** (Fully Hadronic) | n_lep == 0 **and** n_jets ≥ 4 | WW→qqqq |
| **SL** (Semi-Leptonic) | n_lep == 1 | WW→qqℓν |
| **FL** (Fully Leptonic) | n_lep ≥ 2 | WW→ℓνℓν |

**FL-specific cuts are NOT applied in the workflow.** They are saved as output variables for post-hoc optimization:

| Output variable | Meaning |
|-----------------|---------|
| `fl_met_pt` | MET (PuppiMET) |
| `fl_dipho_pt` | pT(γγ) |
| `fl_mll` | m(ℓℓ) |
| `fl_lead_lep_pt` | leading lepton pT |
| `fl_sublead_lep_pt` | subleading lepton pT |
| `fl_drll` | ΔR(ℓ,ℓ) |
| `fl_has_btag` | event has b-tagged jet (DeepFlavour medium) |

## Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'higgs_dna'` | Run `pip install -e . --no-deps` |
| `ModuleNotFoundError: No module named 'coffea'` | `pip install coffea` |
| `FileNotFoundError: LoosePhoIDSF_2024.json` | `pull_files.py --target LooseMva` |
| `FileNotFoundError: pileup_2024.json.gz` | `pull_files.py --target PU` |
| `FileNotFoundError: Cert_Collisions2024...` | `pull_files.py --target GoldenJSON` |
| `ValueError: Scale corrections should only be applied to data!` | Remove `Scale2G_IJazZ` from MC corrections (it's data-only) |
| `OSError: [3010] permission denied` | No grid proxy. Run `voms-proxy-init --voms cms` |
| `Operation expired` | Run `voms-proxy-init --voms cms -valid 192:00` |
| `Skipping bad file after 4 attempts` | Proxy expired or file server issue. Refresh proxy and re-run |
