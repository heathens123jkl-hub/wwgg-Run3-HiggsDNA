"""
Rename trees in converted ROOT files to flashgg convention:
  gghh_FH_13TeV_FH  ->  gghh_FH_125_13TeV_FH     (insert missing mass)
  ggH_125_13TeV_FH  ->  unchanged
  Data_13TeV_FH     ->  unchanged

Usage: python3 rename_trees.py <input.root> <output.root>
"""
import uproot
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]

# sigma x BR in fb: single-H from LHC Higgs XSWG table (13.6 TeV, mH=125).
# Signal: sigma(ggF->HH) = 34.13 fb (XSWG NNLO FTapprox) x BR(HH->WWgg) = 9.77e-4
# (YR4: 2 x BR(H->WW)=0.2152 x BR(H->gg)=2.270e-3) x W-decay fractions
# qqqq 0.4544 / qqln 0.4393 / lnln 0.1061.
XSBR_FB = {
    "gghh_FH": 0.01515,
    "gghh_SL": 0.01465,
    "gghh_FL": 0.00354,
    "ggH": 117.4,
    "VBFH": 9.34,
    "ttH": 1.34,
    "ZH": 2.15,
    "WplusH": 2.04,
    "WminusH": 1.29,
}

with uproot.recreate(dst) as fout:
    fin = uproot.open(src)
    for key in fin.keys():
        path = key.split(";")[0]
        if "/" not in path:  # skip the DiphotonTree directory entry itself
            continue
        tree_name = path.split("/")[-1]
        if "_125_" in tree_name:
            new_name = tree_name
        elif tree_name.startswith("Data"):
            new_name = tree_name
        else:
            # gghh_FH_13TeV_FH -> gghh_FH_125_13TeV_FH
            new_name = tree_name.replace("_13TeV_", "_125_13TeV_")
        print(f"{tree_name} -> {new_name}")
        data = fin[key].arrays(library="ak")
        # merge_parquet normalizes weights to efficiency (weight / sum_genw_presel);
        # flashgg expects sigma x BR x efficiency, so multiply by sigma x BR
        proc = new_name.split("_125_")[0]
        if proc in XSBR_FB:
            data["weight"] = data["weight"] * XSBR_FB[proc]
        # flashgg signal fit requires dZ (diphoton vertex z); HiggsDNA output
        # has no vertex z, so fill with 0 (all events treated as right-vertex)
        data["dZ"] = np.zeros(fin[key].num_entries, dtype=np.float64)
        fout.mktree(f"DiphotonTree/{new_name}", data)
print(f"Done: {dst}")
