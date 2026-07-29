# Run3 HH→WWγγ Preselection — Complete Setup & Run Guide

**Author**: Weitao Xiong  
**Date**: July 2026  
**Repo**: https://github.com/heathens123jkl-hub/wwgg-Run3-HiggsDNA

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Conda Environment Setup](#3-conda-environment-setup)
4. [POG JSON Files](#4-pog-json-files)
5. [Verify Installation](#5-verify-installation)
6. [Prepare Your Analysis JSON](#6-prepare-your-analysis-json)
7. [Prepare Your Sample JSON](#7-prepare-your-sample-json)
8. [Run Locally (Iterative)](#8-run-locally-iterative)
9. [Submit to HTCondor](#9-submit-to-htcondor)
10. [Check Output](#10-check-output)
11. [Diagnostic Scripts](#11-diagnostic-scripts)
12. [Common Issues](#12-common-issues)

---

## 1. Prerequisites

You need a **lxplus** account (or any CERN machine with EOS access and conda).

You also need a **grid certificate** loaded into your browser to access XRootD files. If you don't have one, ask your supervisor.

Verify:
```bash
voms-proxy-init --voms cms -valid 192:00
```

If this fails with "AUP not signed", go to https://cernaccount.web.cern.ch/ and sign the CMS AUP.

---

## 2. Clone the Repository

```bash
cd /eos/home-x/$USER/WWyy   # or wherever you want to put it
git clone https://github.com/heathens123jkl-hub/wwgg-Run3-HiggsDNA.git HiggsDNA_WWgg
cd HiggsDNA_WWgg
```

---

## 3. Conda Environment Setup

Weitao's existing conda environment is at `/eos/home-x/xiongw/WWyy/conda_envs/higgs-dna-run3`.  
You can either use his environment or create your own.

### Option A: Use Weitao's environment (fastest)

```bash
conda activate /eos/home-x/xiongw/WWyy/conda_envs/higgs-dna-run3
```

### Option B: Build your own

```bash
conda create -n higgs-dna-run3 python=3.12 -y
conda activate higgs-dna-run3
pip install coffea[dask,spark]==2025.4.0 awkward uproot vector correctionlib
pip install pyarrow matplotlib rich
pip install scipy==1.14.0   # 1.18 has API break
```

---

## 4. POG JSON Files

HiggsDNA needs POG JSONs for jet veto maps and b-tag scale factors. Clone them:

```bash
cd higgs_dna/systematics/JSONs
git clone https://github.com/cms-jet/POGJSONHIN.git POG 2>/dev/null || echo "already exists"
cd POG
# Flatten nested POG/POG/ directories if needed
find . -name "*.json" -exec cp {} ../ \; 2>/dev/null
cd ../../../..
```

If this fails, the workflow will still run — the jet veto map step will be skipped (try/except).

---

## 5. Verify Installation

```bash
conda activate /eos/home-x/xiongw/WWyy/conda_envs/higgs-dna-run3
python -c "import awkward, coffea, uproot, higgs_dna; print('OK')"
```

---

## 6. Prepare Your Analysis JSON

The analysis JSON tells HiggsDNA which workflow, which samples, and which era to use.

Create a file named `my_analysis.json`:

```json
{
  "samplejson": "my_samples.json",
  "workflow": "WWgg",
  "metaconditions": "Era2023_postBPix_v1",
  "taggers": [],
  "split_mc": false,
  "year": {
    "MY_SAMPLE_NAME": ["2024"]
  },
  "corrections": {},
  "systematics": {}
}
```

**Field explanations:**

| Field | Value | Meaning |
|-------|-------|---------|
| `samplejson` | path to samples JSON | Points to the file listing ROOT input files |
| `workflow` | `"WWgg"` | Must match the registered workflow name. Available: `WWgg`, `HHbbgg`, `base`, `zmmy`, etc. |
| `metaconditions` | `"Era2023_postBPix_v1"` | Era configuration. Options: `Era2022_preEE_v1`, `Era2022_postEE_v1`, `Era2023_preBPix_v1`, `Era2023_postBPix_v1`. For 2024 data, use `Era2023_postBPix_v1` as a placeholder. |
| `taggers` | `[]` | No taggers needed for WWgg preselection |
| `split_mc` | `false` | Set to `true` to split MC by event number (not needed for preselection) |
| `year` | `{"MY_SAMPLE_NAME": ["2024"]}` | Map from sample name → era tag. Use `"2024"` for RunIII2024, `"2022preEE"` for 2022 preEE, etc. |
| `corrections` | `{}` | Corrections (efficiency SF, smearing, etc.). Leave empty for first test. |
| `systematics` | `{}` | Systematics variations. Leave empty for first test. |

**Available era tags for `year` field:**
- `"2022preEE"`, `"2022postEE"`
- `"2023preBPix"`, `"2023postBPix"`
- `"2024"` (placeholder)

---

## 7. Prepare Your Sample JSON

The sample JSON lists the ROOT files for each dataset name.

### Option A: Fetch from DAS (recommended)

If you know the DAS dataset name:

```bash
echo "MY_SAMPLE_NAME /DAS/DATASET/NAME" > /tmp/das_input.txt
python higgs_dna/scripts/samples/fetch_datasets.py -i /tmp/das_input.txt -w Eurasia --dbs-instance prod/global
```

Then replace the Infn redirector with the CERN one:

```bash
sed -i 's|root://xrootd-cms.infn.it/|root://cms-xrd-global.cern.ch/|g' /tmp/das_input.json
```

The output JSON (e.g. `das_input.json`) is your sample JSON. Point `samplejson` in your analysis JSON to it.

### Option B: Manual (if you have ROOT files on EOS)

```json
{
  "MY_SAMPLE_NAME": [
    "root://cms-xrd-global.cern.ch//store/mc/.../file1.root",
    "root://cms-xrd-global.cern.ch//store/mc/.../file2.root"
  ]
}
```

**Weitao's pre-fetched 2024 samples** are available in:
- `samples_2024_signal.json` (FH/SL/FL, SM kl=1, 276 files)
- `samples_2024_background.json` (GGJets, TTGG, single-H, 1,561 files)

For a quick test with 1 file:
```bash
python -c "import json; d=json.load(open('samples_2024_signal.json')); sl=d['WWgg_SL_2024']; json.dump({'WWgg_SL_test':[sl[0]]}, open('my_test_samples.json','w'))"
```

---

## 8. Run Locally (Iterative)

For small tests (a few files):

```bash
conda activate /eos/home-x/xiongw/WWyy/conda_envs/higgs-dna-run3

python higgs_dna/scripts/run_analysis.py \
    --json-analysis my_analysis.json \
    --nano-version 15 \
    --executor iterative \
    --timeout 300 \
    --dump ./my_output
```

**Arguments:**

| Argument | Value | When to use |
|----------|-------|------------|
| `--json-analysis` | path | Points to your analysis JSON |
| `--nano-version` | `12`, `13`, or `15` | 12=2022 v12, 13=2023 v13, **15**=2024 v15 |
| `--executor` | `iterative` or `vanilla_lxplus` | `iterative` for small tests, `vanilla_lxplus` for full production |
| `--timeout` | 300 | XRootD timeout in seconds. Increase if files are slow. |
| `--dump` | directory | Where parquet output goes |
| `--max` | N | Process only N chunks (for testing) |
| `--save` | file.coffea | Save efficiency metadata |

**Output**: Parquet files in `--dump / SAMPLE_NAME / nominal / *.parquet`

Each parquet file contains per-event variables including:
- Photon: `lead_pt`, `lead_eta`, `lead_mvaID`, `lead_hoe`, `lead_r9`, `lead_sieie`, sublead equivalents, ...
- Diphoton: `mass`, `pt`, `eta`, `phi`
- Event-level: `n_ele`, `n_mu`, `n_lep`, `n_jets`, `category`
- Lepton details: `lead_ele_pt`, `lead_ele_cutBased`, `lead_mu_pt`, `lead_mu_tightId`, ...
- Raw lepton (pre-cut): `raw_ele_pt`, `raw_mu_pt`, ...

---

## 9. Submit to HTCondor

For full production (hundreds of files):

```bash
conda activate /eos/home-x/xiongw/WWyy/conda_envs/higgs-dna-run3
voms-proxy-init --voms cms -valid 192:00

python higgs_dna/scripts/run_analysis.py \
    --json-analysis my_analysis.json \
    --nano-version 15 \
    --executor vanilla_lxplus \
    --queue workday \
    --timeout 300 \
    --dump ./my_output
```

**Condor-specific arguments:**

| Argument | Value | Meaning |
|----------|-------|---------|
| `--executor` | `vanilla_lxplus` | Auto-generates `.sub` file and runs `condor_submit` |
| `--queue` | `workday` | Job queue. Options: `workday`, `longlunch`, `tomorrow` |
| `--voms` | path | Alternative path to grid proxy file (default: auto-detect) |

**Monitor jobs:**
```bash
condor_q $USER              # list your jobs
condor_q $USER -nobatch     # more detail
```

**Job output** is in the `--dump` directory, same structure as iterative mode.

**Note**: Do NOT use `cmsenv` (CMSSW environment). HiggsDNA uses conda. Mixing them causes import errors.

---

## 10. Check Output

After the job completes:

```bash
# List output files
ls my_output/MY_SAMPLE_NAME/nominal/

# Quick category check
python -c "
import awkward as ak, glob
fs = sorted(glob.glob('my_output/*/nominal/*.parquet'))
e = ak.concatenate([ak.from_parquet(f) for f in fs])
e = e[~ak.is_none(e.mass)]
print(f'Events: {len(e)}')
print(f'  FH(0): {int(ak.sum(e.category==0))}')
print(f'  SL(1): {int(ak.sum(e.category==1))}')
print(f'  FL(2): {int(ak.sum(e.category==2))}')
"
```

---

## 11. Diagnostic Scripts

Several helper scripts are included in the repo:

| Script | Usage | What it does |
|--------|-------|-------------|
| `debug_lepton_eff.py` | `python debug_lepton_eff.py <parquet_dir>` | Cut-flow + lepton efficiency breakdown |
| `debug_fl_classification.py` | `python debug_fl_classification.py <parquet_dir>` | FL misclassification root cause analysis |
| `plot_cut_validation.py` | `python plot_cut_validation.py <parquet_dir> <out_dir> <label>` | Selection cut validation plots (7 figures per sample) |
| `show_full_results.py` | `python show_full_results.py <FH_dir> <SL_dir> <FL_dir>` | Full confusion matrix across 3 samples |
| `calc_efficiency.py` | `python calc_efficiency.py` | Selection efficiency from coffea output |

---

## 12. Common Issues

### `AttributeError: no field named 'jetId'`
Solution: Use `--nano-version 15` (v15 requires dynamic jetId calculation, which WWgg.py handles). If running v13, use `--nano-version 13`.

### `OSError: File did not open properly: [ERROR] Operation expired`
Your VOMS proxy expired. Run:
```bash
voms-proxy-init --voms cms -valid 192:00
```

### `OSError: [3010] ... permission denied`
You are using the wrong XRootD redirector. Replace `root://xrootd-cms.infn.it/` with `root://cms-xrd-global.cern.ch/` in your sample JSON.

### `no field named 'ScEta'`
WWgg.py adds `ScEta` from `eta` if missing. This warning is harmless.

### `You are running over MC and not applying trigger SF`
Harmless warning. Trigger SF is only needed for final measurements.

### `FutureWarning: torch.distributed.reduce_op is deprecated`
XRootD/PyTorch version mismatch. Harmless, can be ignored.

### Condor job stuck in Idle
Means no slots available. Wait a few minutes. If stuck >30 min, try `--queue tomorrow`.

---

## Quick Start (Copy-Paste)

```bash
# One-time setup
cd /eos/home-x/$USER
git clone https://github.com/heathens123jkl-hub/wwgg-Run3-HiggsDNA.git WWgg
cd WWgg
conda activate /eos/home-x/xiongw/WWyy/conda_envs/higgs-dna-run3
voms-proxy-init --voms cms -valid 192:00

# Create test sample (1 file from 2024 SL)
python -c "import json; d=json.load(open('samples_2024_signal.json')); sl=d['WWgg_SL_2024']; json.dump({'WWgg_SL_test':[sl[0]]}, open('my_test_samples.json','w'))"

# Create test analysis JSON
cat > my_test_analysis.json << 'EOF'
{
  "samplejson": "my_test_samples.json",
  "workflow": "WWgg",
  "metaconditions": "Era2023_postBPix_v1",
  "taggers": [],
  "split_mc": false,
  "year": {"WWgg_SL_test": ["2024"]},
  "corrections": {},
  "systematics": {}
}
EOF

# Run (local, 1 chunk)
python higgs_dna/scripts/run_analysis.py \
    --json-analysis my_test_analysis.json \
    --nano-version 15 \
    --executor iterative \
    --timeout 300 \
    --dump ./test_output \
    --max 1

# Check result
ls test_output/WWgg_SL_test/nominal/
```
