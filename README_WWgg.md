# HH→WWγγ Non-Resonant Preselection for Run3

Run3 HiggsDNA workflow implementing the HH→WWγγ non-resonant preselection
based on CMS AN-2020-165.

## Overview

```
NanoAOD → photon preselection → diphoton candidates → fiducial cuts
→ electron selection → muon selection → AK4 jet selection
→ dR cleaning → Z-veto → category → parquet output
```

### Event categories (AN Note §5.2-5.4)

| Cat | Name | Requirement |
|-----|------|-------------|
| 0 | FH (Fully Hadronic) | 0 lepton + >= 4 jets (WW → qqqq) |
| 1 | SL (Semi-Leptonic) | 1 lepton (WW → qqℓν) |
| 2 | FL (Fully Leptonic) | >= 2 leptons + MET>20 + pT(γγ)>91 + Z→ll veto + b-veto (WW → ℓνℓν) |

## Environment

```bash
# lxplus 7 with conda
conda activate higgs-dna-run3

# VOMS proxy (for XRootD access to remote files)
voms-proxy-init --voms cms -valid 192:00
```

Requires:
- Python 3.12+
- coffea, awkward, uproot, vector, correctionlib
- HiggsDNA framework (clone from upstream repo)

## Configuration

Two JSON files are needed: samples JSON and analysis JSON.

### samples JSON (`wwgg_samples.json`)

```json
{
    "SampleName": [
        "root://cms-xrd-global.cern.ch//store/mc/.../file1.root",
        "root://cms-xrd-global.cern.ch//store/mc/.../file2.root"
    ]
}
```

### analysis JSON (`wwgg_analysis.json`)

```json
{
    "samplejson": "wwgg_samples.json",
    "workflow": "WWgg",
    "metaconditions": "Era2023_postBPix_v1",
    "taggers": [],
    "split_mc": false,
    "year": {"SampleName": ["2023postBPix"]},
    "corrections": {},
    "systematics": {}
}
```

Available metaconditions:
- `Era2022_preEE_v1`, `Era2022_postEE_v1`
- `Era2023_preBPix_v1`, `Era2023_postBPix_v1`

## Running

```bash
cd HiggsDNA_Run3

# Basic run (single file)
python higgs_dna/scripts/run_analysis.py \
    --json-analysis wwgg_analysis.json \
    --nano-version 13 \
    --executor iterative \
    --timeout 300 \
    --dump /path/to/output/

# With efficiency counter
python higgs_dna/scripts/run_analysis.py \
    --json-analysis wwgg_analysis.json \
    --nano-version 13 \
    --executor iterative \
    --timeout 300 \
    --dump /path/to/output/ \
    --save hists_WWGG_SL.coffea
```

Key arguments:
- `--nano-version`: 12 or 13 (depends on NanoAOD version)
- `--executor`: `iterative` (single machine), `dask/local` (parallel)
- `--dump`: parquet output directory
- `--save`: coffea efficiency output file
- `--timeout`: XRootD timeout in seconds (300 recommended)

## Output

Parquet files contain per-event variables:

```
Diphoton: mass, pt, eta, phi, rapidity
Lead photon: lead_pt, lead_eta, lead_mvaID, lead_hoe, lead_r9, ...
Sublead photon: sublead_pt, sublead_eta, sublead_mvaID, ...
Electron: lead_ele_pt, lead_ele_eta, lead_ele_cutBased, ...
Muon: lead_mu_pt, lead_mu_eta, lead_mu_tightId, lead_mu_pfIsoId, ...
Event: n_ele, n_mu, n_lep, n_jets, category
FL diagnostics: fl_pass_met, fl_pass_dipt, fl_pass_mll, fl_pass_btag, ...
```

## Diagnostic Scripts

```bash
# Lepton efficiency & cut-flow analysis
python debug_lepton_eff.py <parquet_dir>

# FL-specific misclassification diagnosis
python debug_fl_classification.py <parquet_dir>

# Selection cut validation plots
python plot_cut_validation.py <parquet_dir> <output_dir> <label>

# Full results & confusion matrix
python show_full_results.py <FH_dir> <SL_dir> <FL_dir>

# Efficiency from coffea output
python calc_efficiency.py
```

## File Structure

```
higgs_dna/workflows/WWgg.py          # Main workflow processor
higgs_dna/workflows/__init__.py      # Workflow registration (+2 lines)
wwgg_*.json                          # Sample & analysis configs
debug_lepton_eff.py                  # Lepton efficiency diagnostic
debug_fl_classification.py           # FL misclassification diagnostic
plot_cut_validation.py               # Selection cut validation plots
show_full_results.py                 # Full results summary
calc_efficiency.py                   # Efficiency from coffea files
README_WWgg.md                       # This file
```

## Selection Conditions

Selection follows AN-2020-165 §4-5. See the full table in the code comments or ask Claude.

## Authors

Weitao Xiong (IHEP, CAS) — Run3 WWgg workflow adaptation
