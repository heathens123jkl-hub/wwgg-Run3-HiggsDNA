# Same-input data cross-check (Weitao / Tahir)

Use the same Git commit of this repository. This test fixes one Run2024C
EGamma0 NanoAOD v15 input, taken from the existing production sample list.
It tests event selection and parquet output before merging or normalization.
It is not a validation of DD sideband construction or TFractionFitter.

## Before running

Use the working HiggsDNA environment and a valid CMS proxy. Run from the
repository root. Follow `README_WWgg_Run3.md` for correction payload setup;
both users must use identical Golden JSON and photon-scale payload contents.
The Git checkout alone does not fix externally downloaded corrections.
Record the commit and environment (each command is one line):

```bash
git rev-parse HEAD
python -m pip freeze > crosscheck_environment.txt
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python -c 'import importlib; print(importlib.import_module("higgs_dna.workflows.WWgg").__file__)'
```

The imported workflow must be inside this checkout.

## Baseline first

Both users run exactly this command in their own checkout:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 higgs_dna/scripts/run_analysis.py --json-analysis production/crosscheck_20260924/baseline.json --nano-version 15 --executor iterative --chunk 100000 --max 1 --timeout 300 --dump ./output_crosscheck_baseline_20260924 --save hists_crosscheck_baseline_20260924.coffea
```

This processes one chunk from one explicitly named file. Coffea may adjust
chunk boundaries; compare the actual processed entry range in the logs/output
filename, not just the requested chunk size. Bad-file skipping is deliberately
not enabled. Use a fresh output directory for each new attempt.

Compare the input range, selected FH/SL/FL counts and event identities before
interpreting any difference. The following command prints category counts,
duplicate-identity counts and a hash of sorted event identities per category.
It does not print masses or event identifiers. Run it on the baseline output:

```bash
python -c 'from pathlib import Path; import pyarrow.parquet as pq; import hashlib; fs=sorted(Path("output_crosscheck_baseline_20260924").rglob("*.parquet")); assert fs,"No parquet outputs"; rows=[r for f in fs for r in pq.ParquetFile(f).read(columns=["run","luminosityBlock","event","category"]).to_pylist()]; assert all(all(r[k] is not None for k in r) for r in rows),"Null identity/category"; assert all(r["category"] in (0,1,2) for r in rows); groups={c:sorted((r["run"],r["luminosityBlock"],r["event"]) for r in rows if r["category"]==i) for i,c in enumerate(("FH","SL","FL"))}; print({c:{"rows":len(v),"duplicates":len(v)-len(set(v)),"identity_hash":hashlib.sha256(repr(v).encode()).hexdigest()} for c,v in groups.items()})'
```

Matching hashes establish matching identity lists, not matching kinematics or
all output features. If identities differ, compare the two event lists and
cutflows privately to find the first difference. If they agree, then compare
candidate/object variables and parquet schema. A small sample may contain no
FL events and cannot establish FL agreement for the full dataset.

## Loose configuration second

After the baseline comparison, both users can run:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 higgs_dna/scripts/run_analysis.py --json-analysis production/crosscheck_20260924/loose.json --nano-version 15 --executor iterative --chunk 100000 --max 1 --timeout 300 --dump ./output_crosscheck_loose_20260924 --save hists_crosscheck_loose_20260924.coffea
```

Change the output path in the count command to compare loose with loose.
Baseline and loose use the same input, corrections and settings except the
two explicit selection flags. Loose does not relax photon ID, and does not
produce a fake-photon DD control sample automatically.

## Scope of the result

Neither input access nor the real NanoAOD test has been run from the local
Windows preparation environment. No expected counts are pre-filled. This
first check does not test EG0/EG1 overlap removal or era merging. If the
same-file results agree but full totals differ, next compare file coverage,
luminosity masks, stream deduplication and merge scripts. Keep parquet,
coffea and environment outputs outside Git.
