# Run3 HH→WWγγ Preselection — Setup & Run Guide

This guide gets you from zero to a running test job in ~10 minutes.

---

## 1. Clone the Repository

```bash
cd /eos/home-x/$USER/WWyy
git clone https://github.com/heathens123jkl-hub/wwgg-Run3-HiggsDNA.git HiggsDNA_WWgg
cd HiggsDNA_WWgg
```

## 2. Environment

```bash
conda activate /eos/home-x/xiongw/WWyy/conda_envs/higgs-dna-run3
```

If this fails (e.g. `Could not find conda environment`), you need the conda environment. Ask Weitao or build one:

```bash
conda create -n higgs-dna-run3 python=3.12 -y
conda activate higgs-dna-run3
pip install coffea==2025.4.0 awkward uproot vector correctionlib scipy==1.14.0 pyarrow matplotlib rich
```

## 3. Grid Proxy (XRootD access)

```bash
voms-proxy-init --voms cms -valid 192:00
```

## 4. Sample JSON

A single-file test JSON is already prepared — `wwgg_2024_fl_test_samples.json` (one file from the 2024 FL signal, verified working).

The analysis JSON below already points to it. No further setup needed.

If you want to use other samples later, pre-fetched sample JSONs are in the repo:

| File | Content | Keys |
|------|---------|------|
| `samples_2024_signal.json` | HH→WWγγ SM signal | `WWgg_FH_2024`, `WWgg_SL_2024`, `WWgg_FL_2024` |
| `samples_2024_background.json` | GGJets, TTGG, single-H→γγ | `GGJets_MGG-40to80_2024`, `GGJets_MGG-80_2024`, `TTGG_2024`, `GluGluHtoGG_2024`, `VBFHtoGG_2024`, `ttHtoGG_2024` |

## 5. Analysis JSON

Already in the repo — `wwgg_2024_fl_test.json`:

```json
{
  "samplejson": "wwgg_2024_fl_test_samples.json",
  "workflow": "WWgg",
  "metaconditions": "Era2023_postBPix_v1",
  "taggers": [],
  "split_mc": false,
  "year": {"WWgg_FL_test": ["2024"]},
  "corrections": {},
  "systematics": {}
}
```

## 6. Run (Local)

```bash
python higgs_dna/scripts/run_analysis.py \
    --json-analysis wwgg_2024_fl_test.json \
    --nano-version 15 \
    --executor iterative \
    --timeout 300 \
    --dump ./test_output
```

Replace `iterative` with `vanilla_lxplus --queue workday` to submit to HTCondor instead.

## 7. Check Output

Check status: `condor_q $USER`

## 8. Check Output

```bash
python -c "
import awkward as ak, glob
fs = sorted(glob.glob('test_output/*/nominal/*.parquet'))
e = ak.concatenate([ak.from_parquet(f) for f in fs])
e = e[~ak.is_none(e.mass)]
t = len(e)
print(f'Events: {t}')
print(f'  FH(0): {int(ak.sum(e.category==0)):6d} ({ak.sum(e.category==0)/t*100:.1f}%)')
print(f'  SL(1): {int(ak.sum(e.category==1)):6d} ({ak.sum(e.category==1)/t*100:.1f}%)')
print(f'  FL(2): {int(ak.sum(e.category==2)):6d} ({ak.sum(e.category==2)/t*100:.1f}%)')
"
```

## Common Issues

| Error | Fix |
|-------|-----|
| `AttributeError: no field named 'jetId'` | Use correct `--nano-version` (13 for 2023, 15 for 2024) |
| `File did not open properly: [ERROR] Operation expired` | Run `voms-proxy-init --voms cms -valid 192:00` |
| `File did not open properly: [3010] permission denied` | Wrong XRootD redirector. Use `root://cms-xrd-global.cern.ch/` |
| `You are running over MC and not applying trigger SF` | Harmless warning, ignore |
| Condor job stuck in Idle | Wait a few minutes. If >30 min, try `--queue tomorrow` |

> Note: The POG JSON files (jet veto maps) may be missing — the workflow skips this step if files are unavailable. This is fine for a test run. If you need them, clone `https://github.com/cms-jet/POGJSONHIN.git` into `higgs_dna/systematics/JSONs/`.
