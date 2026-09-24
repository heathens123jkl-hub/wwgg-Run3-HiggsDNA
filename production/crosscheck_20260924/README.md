# WWgg HiggsDNA update and 2024C production

## What changed

- Added parquet variables for cut optimization and ML: photon, lepton, jet and MET observables, FL study variables, and MC truth diagnostics.
- Added configurable electron-photon Z veto and diphoton pT/mass cuts, with the original cut decisions saved in the output.
- Fixed empty/null rows in the selected-event parquet output.

Existing corrections and the original selection remain unchanged when both selection switches are set to `apply`.

## Selection switches in the analysis JSON

Add this top-level block to the analysis JSON. **For our 2024C comparison, both cuts are enabled:**

```json
"wwgg_selection": {
  "z_veto": "apply",
  "diphoton_pt_over_mass": "apply"
}
```

- `apply`: apply the cut and save its decision.
- `store_only`: save the decision without applying that cut, for optimization studies.

Omitting this block defaults to `apply` for both cuts. Other selections remain active; `store_only` does not relax photon ID. Relaxing the diphoton cuts can change the selected candidate and object cleaning.

## Run the full 2024C sample

The supplied `data_2024C_baseline.json` already enables both cuts. Its sample list, `samples_2024C.json`, contains all 2024C files from our production list: **167 EGamma0 + 247 EGamma1 files**.

Use the updated checkout in your working HiggsDNA environment, with the correction files installed and a valid CMS proxy. Run from the repository root:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 higgs_dna/scripts/run_analysis.py --json-analysis production/crosscheck_20260924/data_2024C_baseline.json --nano-version 15 --executor vanilla_lxplus --queue workday --timeout 300 --dump ./output_crosscheck_2024C_baseline_20260924 --save hists_crosscheck_2024C_baseline_20260924.coffea
```

HiggsDNA submits the jobs directly to Condor. The command processes the full input list with no event/chunk limit. The parquet output goes to `output_crosscheck_2024C_baseline_20260924/`.

For other samples, use the corresponding analysis JSON with the same `wwgg_selection` block. Keep each sample's existing `year`, `corrections` and `samplejson` settings. General setup is documented in [README_WWgg_Run3.md](../../README_WWgg_Run3.md).
