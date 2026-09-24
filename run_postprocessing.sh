#!/bin/bash
# flashgg B-track: merge parquet -> convert to ROOT -> rename trees.
# Run in conda env from HiggsDNA_Run3 directory.
# Usage: bash run_postprocessing.sh

set -e
CATS=cat_dict_wwgg.json
PP=higgs_dna/scripts/postprocessing
OM=outfiles_wwgg.yaml

mkdir -p merged root_out

echo "===== Step 1: Merge signal MC ====="
for S in FH SL FL; do
    echo "--- Signal $S ---"
    python3 $PP/merge_parquet.py --source 2024_${S}_v2/WWgg_${S}_2024/nominal/ --target merged/${S}/ --cats $CATS --abs
done

echo "===== Step 2: Merge single Higgs MC ====="
declare -A SH=(
    [ggH]="2024_singleH/GluGluHtoGG_2024"
    [VBFH]="2024_singleH/VBFHtoGG_2024"
    [ttH]="2024_singleH/ttHtoGG_2024"
    [ZH]="2024_singleH/VH_ZH_2024"
    [WplusH]="2024_singleH/VH_WplusH_2024"
    [WminusH]="2024_singleH/VH_WminusH_2024"
)
for K in ggH VBFH ttH ZH WplusH WminusH; do
    echo "--- SingleH $K ---"
    python3 $PP/merge_parquet.py --source ${SH[$K]}/nominal/ --target merged/${K}/ --cats $CATS --abs
done

echo "===== Step 3: Merge all data ====="
python3 $PP/merge_parquet.py --source data_all_links/ --target merged/data/ --cats $CATS --is-data --merge-all-data --abs

echo "===== Step 4: Convert to ROOT + rename ====="
declare -A OUTNAMES=(
    [gghh_FH]="output_gghh_FH"
    [gghh_SL]="output_gghh_SL"
    [gghh_FL]="output_gghh_FL"
    [ggH]="output_GluGluHToGG"
    [VBFH]="output_VBFHToGG"
    [ttH]="output_ttHToGG"
    [ZH]="output_ZHToGG"
    [WplusH]="output_WplusHToGG"
    [WminusH]="output_WminusHToGG"
)
for P in gghh_FH gghh_SL gghh_FL ggH VBFH ttH ZH WplusH WminusH; do
    echo "--- Process $P ---"
    MERGEDIR=${P%_*}
    [ "$P" = "gghh_FH" ] && MERGEDIR=FH
    [ "$P" = "gghh_SL" ] && MERGEDIR=SL
    [ "$P" = "gghh_FL" ] && MERGEDIR=FL
    python3 $PP/convert_parquet_to_root.py merged/${MERGEDIR}/merged.parquet root_out/merged.root mc --process ${P} --cats $CATS --abs --outfiles-map $OM
    python3 rename_trees.py root_out/${OUTNAMES[$P]}.root root_out/${OUTNAMES[$P]}_v2.root
done

echo "--- Data ---"
python3 $PP/convert_parquet_to_root.py merged/data/merged.parquet root_out/merged.root data --cats $CATS --abs --outfiles-map $OM
python3 rename_trees.py root_out/allData.root root_out/allData_v2.root

echo "===== Done ====="
ls root_out/
