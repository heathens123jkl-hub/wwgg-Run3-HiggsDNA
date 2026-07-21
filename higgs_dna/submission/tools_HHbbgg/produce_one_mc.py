import os
import argparse
import json


def write_sample_txt(keyword, year, cmsdas):
    filename = f"samples_mc_{year}_{keyword}.txt"
    with open(filename, "w") as f:
        f.write(keyword + " " + cmsdas)
    return filename


def fetch_datasets(sample_file, dbs_instance='prod/global', region='Yolo'):
    command = f"python scripts/samples/fetch_datasets.py -i {sample_file} -w {region} --dbs-instance {dbs_instance}"
    os.system(command)


def run_analysis(nano_version, parent_dir, keyword, year, memory):
    memoryLine = f"--memory {memory} " if memory is not None else ""
    smear = ""
    deco = ""
    doflow = "--doFlow-corrections "
    triggerGroup = ""
    if not any(y in year for y in ["2016", "2017", "2018"]):
        smear = "--Smear-sigma-m "
        deco = "--doDeco "
    if "2025" in year:
        smear = ""
        deco = ""
    if year == "2018":
        triggerGroup = '--triggerGroup ".*EGamma.*2018.*" '
    command = (
        f"python scripts/run_analysis.py "
        f"--json-analysis runner_mc_{year}_{keyword}.json "
        f"--dump {parent_dir} "
        f"{doflow}"
        f"--fiducialCuts store_flag "
        f"{smear}"
        f"{deco}"
        f"{triggerGroup}"
        f"--executor vanilla_lxplus "
        # f"--queue longlunch "
        f"--queue workday "
        f"{memoryLine}"
        f"--debug "
        f"--nano-version {nano_version} "
        f"--timeout 200"
    )
    print(command)
    os.system(command)


def update_json_config(keyword, year, split_mc=False):
    # Using the preliminary JSON except for 2024 - to be updated for final results
    if "2024" in year:
        json_path = "submission/tools_HHbbgg/runner_mc_template.json"
    else:
        json_path = "submission/tools_HHbbgg/prelim_runner_mc_template.json"
    with open(json_path, "r") as f:
        config = json.load(f)
    config["samplejson"] = f"samples_mc_{year}_{keyword}.json"
    if "split_mc" in config:
        if ("2024" in year) or ("2025" in year):
            config["split_mc"] = True  # Enable MC splitting for 2024 and 2025
        else:
            config["split_mc"] = False
    if "year" in config:
        config["year"].pop("GluGluToHH", None)
        config["year"][keyword] = [f"{year}"]
    if "metaconditions" in config:
        if "2016postVFP" in year:
            config["metaconditions"] = "Era2016_legacyPostVFP_v1"
        elif "2016preVFP" in year:
            config["metaconditions"] = "Era2016_legacyPreVFP_v1"
        elif "2017" in year:
            config["metaconditions"] = "Era2017_legacy_v1"
        elif "2018" in year:
            config["metaconditions"] = "Era2018_legacy_v1"
        else:
            config["metaconditions"] = "Era2022_v1"
    if "corrections" in config:
        config["corrections"][keyword] = config["corrections"].pop("GluGluToHH", [])
        if "GluGluHtoGG" in keyword:
            config["corrections"][keyword].append("NNLOPS")
        if "2025" in year:
            config["corrections"][keyword].remove("ElectronVetoSF")
        if any(x in year for x in ("2024", "2025", "2018", "2017", "2016")):
            if any("jet" in corr for corr in config["corrections"][keyword]):
                jerc_idxs = [
                    idx for idx, corr in enumerate(config["corrections"][keyword])
                    if "jet" in corr
                ]
                for jerc_idx in jerc_idxs:
                    # Remove '_syst' from jerc as we don't have for 2025/Run2 yet
                    run2_years = ["2016", "2017", "2018"]
                    if any(x in year for x in run2_years):
                        config["corrections"][keyword][jerc_idx] = config["corrections"][keyword][jerc_idx].replace("_pnetNu_syst", "_pnetNu_Run2_v15")
                        config["corrections"][keyword][jerc_idx] = config["corrections"][keyword][jerc_idx].replace("fatjet_syst","fatjet_Run2_v15")
                    elif "2024" not in year:
                        config["corrections"][keyword][jerc_idx] = config["corrections"][keyword][jerc_idx].replace("_syst", "")
            # Remove bTag SF correction for now as we don't have SFs
            if any("PNet_bTagShapeSF" == corr for corr in config["corrections"][keyword]):
                config["corrections"][keyword].remove("PNet_bTagShapeSF")
                # Add multi fixed WP bTag SFs for 2024 and Run2
                if any(x in year for x in ("2024")):
                    config["corrections"][keyword].append("bTagMultiFixedWP_UParTAK4LMTXTXXT")
                if any(x in year for x in ("2018", "2017", "2016")):
                    config["corrections"][keyword].append("bTagMultiFixedWP_UParTAK4LMTXTXXT_Run2_v15")
            if any(x in year for x in ("2018", "2017", "2016")):
                if any("Smearing2G_IJazZ" == corr for corr in config["corrections"][keyword]):
                    config["corrections"][keyword].remove("Smearing2G_IJazZ")
                    config["corrections"][keyword].append("Smearing_Trad")
                config["corrections"][keyword].append("L1PreFiring")
    if "systematics" in config:
        config["systematics"][keyword] = config["systematics"].pop("GluGluToHH", [])
        if any(x in year for x in ("2024", "2025", "2018", "2017", "2016")):
            if any("PNet_bTagShapeSF" == syst for syst in config["systematics"][keyword]):
                config["systematics"][keyword].remove("PNet_bTagShapeSF")
                if any(x in year for x in ("2024")):
                    config["systematics"][keyword].append("bTagMultiFixedWP_UParTAK4LMTXTXXT")
                if any(x in year for x in ("2018", "2017", "2016")):
                    config["systematics"][keyword].append("bTagMultiFixedWP_UParTAK4LMTXTXXT_Run2_v15")
    new_filename = f"runner_mc_{year}_{keyword}.json"
    with open(new_filename, "w") as f:
        json.dump(config, f, indent=4)

    return new_filename


def main():
    parser = argparse.ArgumentParser(description="Run MC production example pipeline.")
    parser.add_argument("-k", "--keyword", required=True, help="Keyword for dataset filtering")
    parser.add_argument("-c", "--cmsdas", required=True, help="Keyword for cmsdas filtering")
    parser.add_argument("-p", "--parent-dir", required=True, help="Directory to store output parquets")
    parser.add_argument("-y", "--year", required=True, choices=["2022postEE","2022preEE","2023postBPix","2023preBPix", "2024", "2025", "2018", "2017","2016preVFP","2016postVFP"], help="year")
    parser.add_argument("-n", "--nano", required=True, help="nano-version")
    parser.add_argument("-m", "--memory", help="condor job memory")
    parser.add_argument("-s", "--split-mc", action="store_true", help="Enable MC splitting by event ID (even for 2024, odd for 2025)")
    parser.add_argument(
        "-w",
        "--where",
        help="Specify the region for xrootd prefix (only for grid mode).",
        default="Eurasia",
        choices=["Americas", "Eurasia", "Yolo"],
    )
    parser.add_argument(
        "--dbs-instance",
        dest="instance",
        help="The DBS instance to use for querying datasets (only for grid mode).",
        type=str,
        default="prod/global",
        choices=["prod/global", "prod/phys01", "prod/phys02", "prod/phys03"],
    )

    args = parser.parse_args()

    # Write dataset sample file
    sample_file = write_sample_txt(args.keyword, args.year, args.cmsdas)

    # Fetch datasets
    fetch_datasets(sample_file, dbs_instance=args.instance, region=args.where)

    # Update and save JSON configuration
    update_json_config(args.keyword, args.year, split_mc=args.split_mc)

    # Launch jobs
    run_analysis(args.nano, args.parent_dir, args.keyword, args.year, args.memory)


if __name__ == "__main__":
    main()
