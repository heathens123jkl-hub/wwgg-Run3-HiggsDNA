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

Pre-fetched samples are already in the repository. For a quick test, create a single-file sample:

```bash
python -c "
import json
d = json.load(open('samples_2024_signal.json'))
sl = d['WWgg_SL_2024']
json.dump({'WWgg_SL_test': [sl[0]]}, open('my_test_samples.json', 'w'))
"
```

Available pre-fetched samples:

| File | Content | Files |
|------|---------|-------|
| `samples_2024_signal.json` | HH→WWγγ SM signal: FH (87), SL (132), FL (57) | 276 |
| `samples_2024_background.json` | GGJets (40-80, 80-inf), TTGG, GluGluH→γγ, VBF H→γγ, ttH→γγ | 1,561 |

To use these pre-fetched samples, point `"samplejson"` in your analysis JSON to the file name listed above. The sample JSON key names are `WWgg_FH_2024`, `WWgg_SL_2024`, `WWgg_FL_2024` (signal), and `GGJets_MGG-40to80_2024`, `GGJets_MGG-80_2024`, `TTGG_2024`, `GluGluHtoGG_2024`, `VBFHtoGG_2024`, `ttHtoGG_2024` (background).

> The sample JSONs use `root://cms-xrd-global.cern.ch/` as the XRootD redirector. If you fetch your own samples with `fetch_datasets.py`, it uses `root://xrootd-cms.infn.it/` by default — run `sed -i 's|xrootd-cms.infn.it|cms-xrd-global.cern.ch|g' your_samples.json` to fix it.

## 5. Analysis JSON

Create `my_test_analysis.json`:

```json
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
```

| Field | Value | Note |
|-------|-------|------|
| `samplejson` | your samples JSON | Pre-fetched or self-made |
| `workflow` | `"WWgg"` | Must match exactly |
| `metaconditions` | `"Era2023_postBPix_v1"` | 2024 reuses this |
| `year` | `{"KEY": ["2024"]}` | Key must match the sample JSON key |
| `corrections`/`systematics` | `{}` | Empty for first test |

## 6. Run (Local)

```bash
python higgs_dna/scripts/run_analysis.py \
    --json-analysis my_test_analysis.json \
    --nano-version 15 \
    --executor iterative \
    --timeout 300 \
    --dump ./test_output \
    --max 1
```

## 7. Run (HTCondor)

```bash
python higgs_dna/scripts/run_analysis.py \
    --json-analysis my_test_analysis.json \
    --nano-version 15 \
    --executor vanilla_lxplus \
    --queue workday \
    --timeout 300 \
    --dump ./test_output
```

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
