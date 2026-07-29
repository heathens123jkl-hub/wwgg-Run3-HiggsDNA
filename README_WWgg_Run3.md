# Run3 HH→WWγ Preselection — Setup & Run Guide

Follow these steps. You should have a working test job in ~10 minutes.

---

## 1. Clone & Environment

```bash
mkdir -p /eos/home-x/$USER/WWyy && cd /eos/home-x/$USER/WWyy
git clone https://github.com/heathens123jkl-hub/wwgg-Run3-HiggsDNA.git test_WWgg
cd test_WWgg

# Create conda environment
conda create -n wwgg-run3 python=3.12 -y
conda activate wwgg-run3

# Install all required packages
pip install coffea awkward uproot vector correctionlib scipy==1.14.0 pyarrow matplotlib rich dask distributed bokeh pyyaml jinja2

# Install HiggsDNA itself as a package
pip install -e .
```

## 2. Grid Proxy (for XRootD)

```bash
voms-proxy-init --voms cms -valid 192:00
```

If you don't have a grid certificate, use EOS local paths instead of `root://` URLs (lxplus has `/eos/cms/` mounted directly).

## 3. Run the Test

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

## 4. Check Output

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

## 5. Use Other Samples

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

## Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'higgs_dna'` | Run `pip install -e .` |
| `ModuleNotFoundError: No module named 'coffea'` | `pip install coffea` |
| `No module named 'yaml'` / `'dask'` / `'bokeh'` | `pip install pyyaml dask distributed bokeh jinja2` |
| `FileNotFoundError: jetid.json.gz` | Run `git pull` (POG JSONs are now in the repo) |
| `OSError: [3010] permission denied` | No grid proxy. Run `voms-proxy-init --voms cms` or use EOS local paths |
| `Operation expired` | Run `voms-proxy-init --voms cms -valid 192:00` |
