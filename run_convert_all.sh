#!/bin/bash
# flashgg B-track: convert parquet -> ROOT (mktree, writes TTree) -> rename trees.
# Run in conda env higgs-dna-run3 from HiggsDNA_Run3 directory.
# Assumes merged/*/merged.parquet already exist (merge step done).
# Usage: bash run_convert_all.sh

set -e
CATS=cat_dict_wwgg.json
PP=higgs_dna/scripts/postprocessing
OM=outfiles_wwgg.yaml

declare -A PROC_MAP=(
    [gghh_FH]=FH:output_gghh_FH
    [gghh_SL]=SL:output_gghh_SL
    [gghh_FL]=FL:output_gghh_FL
    [ggH]=ggH:output_GluGluHToGG
    [VBFH]=VBFH:output_VBFHToGG
    [ttH]=ttH:output_ttHToGG
    [ZH]=ZH:output_ZHToGG
    [WplusH]=WplusH:output_WplusHToGG
    [WminusH]=WminusH:output_WminusHToGG
)

for P in gghh_FH gghh_SL gghh_FL ggH VBFH ttH ZH WplusH WminusH; do
    echo "--- Process $P ---"
    MD=${PROC_MAP[$P]%%:*}
    ON=${PROC_MAP[$P]##*:}
    PYTHONPATH=$PWD python3 $PP/convert_parquet_to_root.py merged/$MD/merged.parquet root_out/merged.root mc --process $P --cats $CATS --abs --outfiles-map $OM
    python3 rename_trees.py root_out/$ON.root root_out/${ON}_v2.root
done

echo "--- Data ---"
PYTHONPATH=$PWD python3 $PP/convert_parquet_to_root.py merged/data/merged.parquet root_out/merged.root data --cats $CATS --abs --outfiles-map $OM
python3 rename_trees.py root_out/allData.root root_out/allData_v2.root

echo "===== Verify entry counts ====="
python3 -c 'import uproot; [print(fn, {k.split("/")[-1]: uproot.open("root_out/"+fn)[k].num_entries for k in uproot.open("root_out/"+fn).keys() if "/" in k}) for fn in ["output_gghh_FH_v2.root","output_gghh_SL_v2.root","output_gghh_FL_v2.root","output_GluGluHToGG_v2.root","output_VBFHToGG_v2.root","output_ttHToGG_v2.root","output_ZHToGG_v2.root","output_WplusHToGG_v2.root","output_WminusHToGG_v2.root","allData_v2.root"]]'

echo "===== Done ====="
