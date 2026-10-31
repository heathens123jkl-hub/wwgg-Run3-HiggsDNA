# Full available 2024 sample re-production

Code and job descriptions live on AFS. Event outputs, per-file coffea summaries,
input provenance and success records live in a new EOS directory. Small Condor
administrative logs remain under the AFS checkout. The existing conda environment
is reused, not installed onto AFS. This submits nominal production only.

## Included samples

| Group | Sample | NanoAOD files |
|---|---|---:|
| Signal | WWgg FH / SL / FL | 87 / 132 / 57 |
| Single-H | ggH / VBF / ttH | 118 / 76 / 40 |
| Single-H | ZH / W+H / W-H | 50 / 102 / 67 |
| Continuum | GGBox MGG40to80 / MGG80 | 583 / 682 |
| Continuum | TTGG | 62 |
| Continuum | WGG-4FS | Resolved through DAS during preparation |
| Data | C, D, E, F, G, H, I; EGamma0 and EGamma1 | See generated plan |

The script uses only `samples_data_2024_full.json`, not overlapping partial lists.
It removes exact repeated LFNs within a sample and rejects file overlap between
samples. File disjointness does **not** prove event disjointness: merge Data using
unique `(run, luminosityBlock, event)` keys across EGamma streams and I processing
versions. Establish certified run/lumi coverage before normalizing MC; no new
luminosity value is assumed here. Existing lists have not been refreshed via DAS
except WGG. Additional TTJets/TTGJets/W+jets/DY are not supplied by these lists.

DD is derived from new Data control regions, not submitted as independent MC.
The earlier bbgg/ggtautau veto-study samples are not WWgg signal or continuum
production inputs. Save all generated-decay samples in every reconstructed
category to retain migration contributions.

## Prepare on lxplus

Use a **new** AFS checkout, preserving the patched EOS installation and old output.
Check `fs lq` for AFS quota first. The script copies missing non-code calibration
assets from the validated installation (up to 3 GiB), keeps new tracked payloads,
and checks required Flow files. No conda environment is copied.

```bash
conda activate "/eos/home-x/$USER/WWyy/conda_envs/wwgg-run3-rebuild-20260927"
mkdir -p "/afs/cern.ch/user/x/$USER/WWyy"
cd "/afs/cern.ch/user/x/$USER/WWyy"
git clone --depth 1 https://github.com/heathens123jkl-hub/wwgg-Run3-HiggsDNA.git HiggsDNA_20261031
cd HiggsDNA_20261031
fs lq
voms-proxy-init --voms cms --valid 192:00
python production/wwgg/prepare_all_20261031.py prepare \
  --assets-from "/eos/home-x/$USER/WWyy/HiggsDNA_Run3" \
  --output "/eos/home-x/$USER/WWyy/production_20261031"
```

WGG discovery failure stops preparation. It does not silently omit the sample.
The generated plan prints counts and records a file/config/asset hash manifest.
Do not edit the checkout or assets after preparation/submission. This check does
not prove that every remote input is currently accessible or that all new
background processes have passed integrated runtime validation.

## Submit the prepared plan

```bash
python production/wwgg/prepare_all_20261031.py submit \
  --output "/eos/home-x/$USER/WWyy/production_20261031"
```

One input file per job, 1 CPU, 10 GiB memory, `tomorrow` queue, materialization
limit 100 by default. The same submitted cluster covers all groups. MC enables
Flow and b-efficiency accumulation, skips HLT selection and applies trigger SF;
Data uses HLT selection and no Flow/MC SF. Both vetoes are applied. No b/tau-veto
SF is invented. Current correction limitations remain in the integration guide.

Jobs with nonzero exit status are held rather than automatically retried. Inspect
`production_jobs_20261031/submit.log`, `condor_q`, and the per-job logs. A second
submission attempt is blocked to avoid duplicate jobs; resolve an ambiguous
submission manually. Before releasing a failed job, inspect and archive/remove
only that job's partial outputs. Do not rerun the entire production.

## Before merging or training

Require a SUCCESS record and readable summary/output for every input partition;
inspect held/missing jobs and aggregate generator denominators exactly once.
Retain the original Parquet partitions: b-efficiency metadata must be summed
with matching configuration and duplicate-partition protection. These maps and
the checked UParT SF will supply the later b-veto weight. Rebuild DD, check
background reweighting and Data deduplication before new BDT/fit inputs.

No Condor jobs are submitted by merely cloning the repository or running prepare.
