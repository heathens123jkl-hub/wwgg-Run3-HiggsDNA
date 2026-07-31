# Run3 HH→WWγ Preselection — Setup & Run Guide

Follow these steps. You should have a working test job in ~10 minutes.

---

## 1. Clone & Environment

```bash
cd /eos/home-x/$USER
git clone https://github.com/heathens123jkl-hub/wwgg-Run3-HiggsDNA.git wwgg_Run3
cd wwgg_Run3

# Create conda environment
conda create -n wwgg-run3 python=3.12 -y
conda activate wwgg-run3

# Install all required packages
pip install coffea awkward uproot vector correctionlib scipy==1.14.0 pyarrow matplotlib rich dask distributed bokeh pyyaml jinja2

# Install HiggsDNA itself as a package
pip install -e .
```

## 2. Verify POG JSONs

The POG JSONs (jet ID, b-tagging, EGM, pileup, muon) required for scale factors and corrections are bundled in `higgs_dna/systematics/JSONs/POG/`. No separate download is needed.

If the bundled files are outdated or missing, clone the official source:
```bash
cd higgs_dna/systematics/JSONs/
git clone https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration.git POG_tmp
# The official repo nests files under POG_tmp/POG/. Move them up one level:
cp -r POG_tmp/POG/* . && rm -rf POG_tmp
```
(Source: [cms-nanoAOD/jsonpog-integration](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration) — official CMS JSON POG integration for NanoAOD.)

## 3. Grid Proxy (for XRootD)

```bash
voms-proxy-init --voms cms -valid 192:00
```


## 4. Run the Test

A pre-configured single-file FL signal test is included.

### Local (iterative)

```bash
python3 higgs_dna/scripts/run_analysis.py \
    --json-analysis wwgg_2024_fl_test.json \
    --nano-version 15 \
    --executor iterative \
    --timeout 300 \
    --dump ./test_output
```

### On HTCondor

```bash
python3 higgs_dna/scripts/run_analysis.py \
    --json-analysis wwgg_2024_fl_test.json \
    --nano-version 15 \
    --executor vanilla_lxplus \
    --queue workday \
    --timeout 300 \
    --dump ./test_condor_output

# Monitor jobs
condor_q $USER
```

## 5. Check Output

```bash
python3 -c "
import awkward as ak, glob
fs = sorted(glob.glob('test_output/*/nominal/*.parquet'))
e = ak.from_parquet(fs[0])
e = e[~ak.is_none(e.mass)]
t = len(e)
print(f'Events: {t}')
print(f'  FH(0): {int(ak.sum(e.category==0))}  SL(1): {int(ak.sum(e.category==1))}  FL(2): {int(ak.sum(e.category==2))}')
"
```

## 6. Use Other Samples

Pre-fetched 2024 samples are in the repo:

| File | Content |
|------|---------|
| `samples_2024_signal.json` | HH→WWγγ SM signal (FH/SL/FL), 276 files |
| `samples_2024_background.json` | Background (GGJets, TTGG, single-H), 1,561 files |

Keys: `WWgg_FH_2024`, `WWgg_SL_2024`, `WWgg_FL_2024` (signal); `GGJets_MGG-40to80_2024`, etc. (background).

To run a different sample, edit `wwgg_2024_fl_test.json` — change `samplejson` and `year`:

```json
{
  "samplejson": "samples_2024_signal.json",
  "workflow": "WWgg",
  "metaconditions": "Era2023_postBPix_v1",
  "taggers": [],
  "split_mc": false,
  "year": {"WWgg_SL_2024": ["2023postBPix"]},
  "corrections": {},
  "systematics": {}
}
```

Note: always use `Era2023_postBPix_v1` for `metaconditions` and `"2023postBPix"` for `year` when running 2024 data (the 2024 metaconditions are not yet available).

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

**How it works:**
1. `run_analysis.py` reads the JSON config → creates a `WWggProcessor` instance
2. For each file, `WWggProcessor.process(events)` is called
3. All cuts are applied as vectorized operations on awkward arrays (no event loops)
4. Events passing all cuts are written to parquet files

**Where cuts are defined:**
- Thresholds stored as `self.XXX` attributes in `WWggProcessor.__init__()` (lines 80-117)
- `select_*()` functions read these attributes to build boolean masks
- Additional tight lepton cuts from AN2025_108 are inlined in `process()` (lines ~170-220)

---

## Selection Conditions

All cuts follow CMS HH→WWγγ analysis standards (AN-2020-165) and the HH→bbγγ Run 3 object definitions, with tight lepton ID taken from AN2025_108.

### Photons

Applied by `photon_preselection()` (source: `photon_selections.py`, called at WWgg.py line 146):

| Cut | Value | Why |
|-----|-------|-----|
| pT | > 25 GeV | Suppress soft QCD photons |
| Supercluster η | Barrel (EB) or Endcap (EE) only | Exclude forward region with poor resolution |
| mvaID | > -0.9 | BDT photon ID: separates real photons from jet fakes. -0.9 keeps >99% of real photons |
| H/E | < 0.08 | Real photons deposit almost all energy in ECAL; jets leak into HCAL |
| R9 / isolation | R9 > 0.8 OR charged iso×pT < 20 OR relIso < 0.3 | Unconverted photons have R9≈1. Converted/fake photons with low R9 must be isolated |
| Shower shape (4 branches) | EB/EE × hiR9/loR9, each with own sieie + track iso thresholds | CMS cut-based photon ID categories |
| electronVeto | == 1 | Rejects superclusters matched to a pixel seed (likely electrons) |

### Diphoton Pair

All photon pairs are built, sorted by pT. The best pair is selected by `build_diphoton_candidates()` + store-flag fiducial (WWgg.py lines 153-155):

| Cut | Value | Why |
|-----|-------|-----|
| Lead pT | > 35 GeV | H→γγ trigger threshold |
| pT/mγγ | lead > 1/3, sublead > 1/4 | Suppress mismeasured mγγ when one photon is very soft |
| \|η\| | < 2.5 (both) | Barrel/endcap acceptance |
| pfRelIso03 × pT | < 10 (both) | HLT-mimicking isolation requirement |

The `store_flag` mode (same as HHbbgg) adds per-candidate fiducial flags without dropping events, avoiding event-length mismatch bugs downstream.

### Electrons — Tight (AN2025_108 Table 13)

Baseline selection by `select_electrons()` (lepton_selections.py lines 5-40). Extra tight cuts applied at WWgg.py lines 172-190:

| Cut | Value | Where applied | Why |
|-----|-------|--------------|-----|
| Cone-corrected pT | ≥ 10 GeV | Manual post-select | More accurate than raw pT (subtracts nearby energy) |
| \|η\| | < 2.5, exclude 1.444–1.566 | select_electrons | Exclude barrel-endcap transition region |
| \|dxy\| | < 0.05 cm | select_electrons | Prompt electrons from W/Z point to primary vertex |
| \|dz\| | < 0.1 cm | select_electrons | Same along beam direction |
| ID | cutBased ≥ 2 (EGamma POG WP-loose) | select_electrons | Minimal ID to keep high signal efficiency |
| dR(e, γ) | > 0.4 from both photons | select_electrons | Avoid double-counting with photons |
| miniPFRelIso | ≤ 0.4 | Manual post-select | Relative isolation in dynamic cone |
| sip3d | < 8 | Manual post-select | 3D impact parameter significance |
| lostHits | == 0 | Manual post-select | No missing tracker layers (good track quality) |
| convVeto | True | Manual post-select | Rejects photon conversion electrons |
| H/E | ≤ 0.10 | Manual post-select | Hadronic leakage (looser than photon H/E) |
| 1/E−1/p | ≥ −0.04 | Manual post-select | ECAL energy vs track momentum consistency |
| promptMVA | ≥ 0.30 | Manual post-select | BDT: prompt electrons (from W/Z) vs non-prompt (from b/c) |
| σiηiη | ≤ 0.011 (barrel) / 0.030 (endcap) | Manual post-select | Shower width in η; barrel/endcap uses supercluster η |

### Muons — Tight (AN2025_108 Table 14)

Baseline by `select_muons()` (lepton_selections.py lines 43-83). Extra tight cuts at WWgg.py lines 205-213:

| Cut | Value | Where applied | Why |
|-----|-------|--------------|-----|
| Cone-corrected pT | ≥ 10 GeV | Manual post-select | Same as electrons |
| \|η\| | < 2.4 | select_muons | Muon system acceptance |
| \|dxy\| | < 0.05 cm | select_muons | Prompt muons point to primary vertex |
| \|dz\| | < 0.1 cm | select_muons | Same along beam direction |
| ID | mediumId | select_muons | PF muon with quality requirements |
| Iso | pfIsoId ≥ 2 | select_muons | PF isolation flag (loose WP) |
| Global muon | True | select_muons | Matched in tracker + muon chambers → best resolution |
| dR(μ, γ) | > 0.4 from both photons | select_muons | Avoid double-counting |
| miniPFRelIso | ≤ 0.4 | Manual post-select | Relative isolation |
| sip3d | < 8 | Manual post-select | 3D impact parameter significance |
| promptMVA | ≥ 0.5 | Manual post-select | BDT: prompt muons vs non-prompt |

### AK4 Jets (aligned with HH→bbγγ Run 3)

| Cut | Value | Why |
|-----|-------|-----|
| pT | > 20 GeV | Suppress pileup jets |
| \|η\| | < 4.7 | Full tracker acceptance |
| jetId | tightLepVeto (== 6) | Highest quality; rejects leptons misidentified as jets |
| dR(jet, γ) | > 0.4 | Avoid jet-photon overlap |

No dR cleaning of jets against leptons (following HH→bbγγ convention).

### Event-level

| Cut | Value | Why |
|-----|-------|-----|
| Z-veto | \|m(e₁+γ_lead) − 91.2\| > 5 GeV | Suppress Z→ee where one electron fakes a photon |
| Photon ID | lead/sub mvaID > -0.9 | Ensures the two selected photons pass ID |
| MET filters | flag_goodVertices, HBHENoise, etc. | Remove detector noise events |

### Event Categories (AN-2020-165 §5.2-5.4)

Each event is assigned to exactly one category based on lepton multiplicity. The three categories are **orthogonal** (no event can belong to two at once).

| Category | Requirement | W→WW decay | Subsequent analysis |
|----------|-------------|------------|-------------------|
| **FH** (Fully Hadronic) | n_lep == 0 **and** n_jets ≥ 4 | WW→qqqq | Binary DNN |
| **SL** (Semi-Leptonic) | n_lep == 1 | WW→qqℓν | Multiclass DNN |
| **FL** (Fully Leptonic) | n_lep ≥ 2 + FL-specific cuts | WW→ℓνℓν | Cut-based |

**FL-specific cuts** (AN-2020-165 Table 34, applied only to events with ≥2 leptons):

| Cut | Value | Why |
|-----|-------|-----|
| MET | > 20 GeV | Signature of neutrinos from W→ℓν |
| pT(γγ) | > 91 GeV | Optimized S/√B for FL channel |
| Z→ll veto | m(ll) ∉ [80, 100] GeV | Remove Z→ll background |
| Lead lepton pT | > 20 GeV | Tighter pT for FL channel purity |
| Sublead lepton pT | > 10 GeV | Standard subleading threshold |
| ΔR(l, l) | > 0.4 | Avoid overlapping leptons |
| b-veto | DeepFlavour medium WP | Suppress ttH background (~85% rejection, ~5% signal loss) |

Events not matching any category (n_lep == 0 but n_jets < 4, or failing all FL cuts) are discarded.

## Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'higgs_dna'` | Run `pip install -e .` |
| `ModuleNotFoundError: No module named 'coffea'` | `pip install coffea` |
| `No module named 'yaml'` / `'dask'` / `'bokeh'` | `pip install pyyaml dask distributed bokeh jinja2` |
| `FileNotFoundError: jetid.json.gz` | Run `git pull` (POG JSONs are now in the repo) |
| `OSError: [3010] permission denied` | No grid proxy. Run `voms-proxy-init --voms cms` |
| `Operation expired` | Run `voms-proxy-init --voms cms -valid 192:00` |
