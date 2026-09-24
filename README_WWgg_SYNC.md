# WWgg production synchronization with Tahir

Prepared on 2026-09-24. Repository: https://github.com/heathens123jkl-hub/wwgg-Run3-HiggsDNA

## Version and status

The analysis implementation was introduced in commit `4a13f74`. Record the
full checked-out commit with `git rev-parse HEAD` when exchanging results.
Use the [full 2024C data cross-check](production/crosscheck_20260924/README.md)
to compare identical input and selection settings before full production.
The shared 2024C test uses the original selection: both flags are
`apply`. The loose production configuration described below is for the
separate optimization study, not this comparison with Tahir.

The inspected `higgs_dna/workflows/WWgg.py` SHA256 is
`98ca982bdc2ba7e2f9b6a02dd53c3587845bd58e9ffad8a94daf0da1679e45e8`.
The output feature schema is version `2`.

The reported lxplus small-sample runs checked the SL signal sample in four
selection modes and the FL signal sample in baseline/loose modes. The empty
output-mask slots were removed without changing the SL yields. These checks
are not a complete validation of data or continuum-background production.
Full production and file-coverage checks are still separate tasks.

## Agree on three separate selection definitions

| Purpose | Configuration | Meaning |
| --- | --- | --- |
| Reproduce the original nominal selection | Both flags `apply`, or omit `wwgg_selection` | Apply the original Z veto and diphoton transverse-momentum/mass cuts. |
| Produce inputs for cut optimization | Both flags `store_only` | Save the cut decisions without rejecting events on these two conditions. |
| Produce a fake-photon DD control sample | Must be specified with Tahir | Photon-ID/control-region changes are additional to the two flags above. |

The loose production JSONs are in `production/configs_loose_20260923/`.
They refer to sample lists under `production/samples/`. They keep the
configured corrections and use:

```json
"wwgg_selection": {
  "z_veto": "store_only",
  "diphoton_pt_over_mass": "store_only"
}
```

Photon ID, isolation, eta, absolute photon transverse-momentum and other
existing selections are still applied. These loose outputs are therefore
not automatically a photon-ID sideband sample.

Changing diphoton cuts can change which photon pair is selected, and hence
object cleaning, categories and candidate-dependent weights. Applying the
old cut flags to the saved loose candidate does not exactly reproduce a
baseline production. Compare baseline with baseline when investigating the
old data-count discrepancy; do not compare old tight totals to new loose totals.

## Reproducible data cross-check

The agreed comparison now covers all 414 Run2024C EG0/EG1 files in the
existing production list, with no file/chunk limit. The linked instructions
include the native Condor command. The original single-file test remains
available only as a diagnostic. Compare streams separately and check overlap
removal and file coverage before quoting merged counts.

Before large production, agree on the same code commit, analysis JSON,
NanoAOD file, era, correction payload versions and processing range. Record
package versions from the working environment as well. Use the same random
configuration for MC smearing when comparing event kinematics.

From the chosen checkout, verify the runtime import (one-line command):

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python -c 'import importlib; m=importlib.import_module("higgs_dna.workflows.WWgg"); print(m.__file__)'
```

It must point into that checkout. Continue to use the native
`python3 higgs_dna/scripts/run_analysis.py` entry point. Use `iterative` for
the agreed small-file test and `vanilla_lxplus` for batch production; do not
introduce a separate submission implementation. Use distinct output paths.

Compare event identifiers, selected candidate kinematics, FH/SL/FL counts,
weights for MC, and the saved cut flags. For data, agree on EG0/EG1 overlap
handling and the luminosity mask. The file list and processed-event coverage
must also match, since a successful process exit does not prove that all
input files were processed when skipping bad files is enabled.

The parquet metadata records the processor/helper fingerprints, feature
schema, selection flags, dataset, era, source file and correction names.
Example for a chosen output file:

```bash
python -c 'import pyarrow.parquet as pq; print({k.decode():v.decode() for k,v in (pq.read_metadata("CHECK.parquet").metadata or {}).items() if k.startswith(b"wwgg_")})'
```

Replace `CHECK.parquet` with an actual file. Metadata is provenance evidence,
not a substitute for comparing the event content. Missing observables can be
NaN with validity flags; do not replace them with physical zeros. Truth,
event identity and diphoton mass are not automatic ML inputs.

## Continuum-background handoff

For the first ML study, request the existing GGJets + DDQCDGJets estimate,
with event-level variables, original event identity and component labels.
Separate files are also suitable. Histogram-only templates cannot supply
general event-level ML inputs. Keep original weights and post-fit factors
separately; do not add standalone GJet/QCD MC on top of a DD estimate of the
same contribution.

Before choosing training weights, obtain:

- Exact photon-ID and mass-sideband definitions, including the blinded region.
- DD construction code, prompt contamination treatment and any resampling.
- Independent event counts and weight sums/uncertainties per category/component.
- The distribution and region fitted by TFractionFitter, template normalization
  convention, fitted fractions, derived scale factors and uncertainties.
- Category-specific fit and variable-validation plots, especially for SL.

The reported overall GGJets factor `0.624` and SL factor about `6.5` should
not be compared as universal corrections without their definitions and
uncertainties. A normalization fit alone does not establish suitability of
all training-variable distributions. Final continuum modeling remains a
separate data-sideband analytic-fit task.
