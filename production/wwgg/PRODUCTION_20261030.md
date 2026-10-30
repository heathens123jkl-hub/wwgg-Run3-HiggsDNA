# WWgg nominal re-production, 2026-10-30

This release enables the integrated corrections and both vetoes in explicit new
configs. It is a nominal re-production release, not a complete final systematic
uncertainty model. See [integration and validation](INTEGRATION_20261029.md).

## Configurations

| Config | Samples | b-efficiency accumulation |
|---|---|---|
| `production_signal_20261030.json` | WWgg FH, SL, FL; all files in the existing signal list | On |
| `production_singleH_20261030.json` | ggH, VBF, ttH, ZH, W+H, W-H | On |
| `production_data_CDEI_20261030.json` | Existing C/D/E/I data list, including its I_v2 entries | Off |

The Data config is **not all 2024 data**: F/G/H are not in that list. Do not assign
full-year luminosity to these outputs without matching the actual certified
coverage. This release does not change luminosity normalization.

Use all three generated WWgg decay samples when constructing reconstructed FH
and SL signal templates; do not equate the generated decay mode with the output
category. GGJets, WGG and TTGG production configs must receive the same explicit
correction/veto settings and validated sample/year mappings before submission.
Do not reuse older background configs with empty corrections. DD must be rebuilt
from the new data/control selection, and background reweighting checked again.

## Environment and submission

Keep code on AFS, outputs on EOS, and retain the previously validated conda
environment. A clean checkout needs external correction/model assets, which are
not all in Git. Reuse the validated assets from the existing EOS installation;
do not replace new tracked WWgg payloads with older files. In particular, check
the seven Flow/MVA files under `higgs_dna/tools/flows/2024_model` and
`higgs_dna/tools/flows/run3_mvaID_models`, and the existing photon/PU/POG assets.
Keep the checkout and assets fixed while jobs are running. Ensure `PYTHONPATH`
resolves this checkout rather than an older editable installation.

From the prepared checkout, with a valid CMS proxy, submit each group into a
fresh directory. For example:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python higgs_dna/scripts/run_analysis.py \
  --json-analysis production/wwgg/production_signal_20261030.json \
  --nano-version 15 --executor vanilla_lxplus --queue workday --timeout 300 \
  --no-trigger --doFlow-corrections \
  --dump "/eos/home-x/$USER/WWyy/production_20261030/signal"
```

For single-H replace the config with `production_singleH_20261030.json` and
the output suffix with `singleH`. MC keeps the validated no-HLT-selection
configuration and applies the registered trigger SF.

For Data use `production_data_CDEI_20261030.json`, output suffix `data_CDEI`,
and **omit both `--no-trigger` and `--doFlow-corrections`**. Data trigger
selection remains enabled; photon Flow and MC SFs are not applied.

## b-veto efficiency bookkeeping

MC configs set `wwgg_corrections.b_efficiency=true`. The collector runs after
the tau veto and before the b veto on baseline candidates. For each reconstructed
channel, jet flavor and pt/eta bin it stores tagged/all counts and generator-weight
moments, including signed sums, squared sums and absolute sums. It does not change
event weights or choose a negative-weight efficiency prescription.

The per-partition record is the `wwgg_b_efficiency` Parquet metadata. **Retain
original output partitions**: ordinary Parquet concatenation does not add these
records. A later merge must deduplicate successful partitions and check matching
selection/binning/configuration before summing moments. Do not average individual
partition efficiencies. The six focused collector tests passed; accumulation-on
full production has not yet been validated end to end.

## Remaining work after nominal production

- Validate output completeness, denominators, categories and metadata using the
  new production itself; no additional enlarged pilot is requested.
- Build and inspect b-tag efficiency maps, then apply the matching fixed-WP
  untagged-event SF. The 2024 UParT SF payload has been checked separately, but
  **no b-veto SF is currently applied**.
- No matching 2024 tau-veto SF prescription has been confirmed; it remains
  explicitly uncorrected. Tested veto rejection is not proof of complete
  orthogonality to another analysis.
- Finish photon/trigger/PU and remaining selection uncertainty prescriptions;
  the supported explicit JES/JER/lepton jobs are not a complete nuisance model.
- Rebuild DD/reweighting, BDT inputs, category optimization and fit templates
  consistently with the new selection. Preserve the previous production.

## Main code locations

- `higgs_dna/workflows/WWgg.py`: workflow integration and bookkeeping.
- `higgs_dna/tools/wwgg_corrections.py`: configuration, JME and lepton SF routing.
- `higgs_dna/tools/wwgg_jec_type1.py`, `wwgg_jer_math.py`: JEC/JER/MET numerics.
- `higgs_dna/tools/flow_corrections.py`: optional float64 Flow evaluation.
- `higgs_dna/tools/wwgg_tau_veto.py`, `wwgg_b_veto.py`: veto definitions.
- `higgs_dna/tools/wwgg_b_efficiency.py`: pre-b-veto efficiency moments.
- `higgs_dna/systematics/JSONs/WWgg/2024/`: pinned JME/lepton payloads.
