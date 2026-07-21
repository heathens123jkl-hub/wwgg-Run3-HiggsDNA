import os
import argparse
import json


def write_sample_txt(keyword, year, cmsdas):
    filename = f"samples_data_{year}_{keyword}.txt"
    with open(filename, "w") as f:
        f.write(keyword + " " + cmsdas)
    return filename


def fetch_datasets(sample_file):
    command = f"python scripts/samples/fetch_datasets.py -i {sample_file} -w Yolo"
    os.system(command)


def validate_run_analysis(nano_version, parent_dir, keyword, year, memory):
    memoryLine = f"--memory {memory} " if memory is not None else ""
    smear = ""
    deco = ""
    triggerGroup = ""
    if not any(y in year for y in ["2016", "2017", "2018", "2025"]):
        smear = "--Smear-sigma-m "
        deco = "--doDeco "
    if year == "2018":
        triggerGroup = '--triggerGroup ".*EGamma.*2018.*" '
    command = (
        f"python scripts/run_analysis.py "
        f"--json-analysis runner_data_{year}_{keyword}.json "
        f"--dump {parent_dir} "
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
        f"--validate"
    )
    print(command)
    os.system(command)


def run_analysis(nano_version, parent_dir, keyword, year, memory):
    memoryLine = f"--memory {memory} " if memory is not None else ""
    smear = ""
    deco = ""
    triggerGroup = ""
    if not any(y in year for y in ["2016", "2017", "2018", "2025"]):
        smear = "--Smear-sigma-m "
        deco = "--doDeco "
    if year == "2018":
        triggerGroup = '--triggerGroup ".*EGamma.*2018.*" '
    command = (
        f"python scripts/run_analysis.py "
        f"--json-analysis runner_data_{year}_{keyword}.json "
        f"--dump {parent_dir} "
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


def update_json_config(keyword, year):
    with open("submission/tools_HHbbgg/runner_data_template.json", "r") as f:
        config = json.load(f)
    config["samplejson"] = f"samples_data_{year}_{keyword}.json"
    if "year" in config:
        config["year"].pop("Run2023Cv1", None)
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
        config["corrections"][keyword] = config["corrections"].pop("Run2023Cv1", [])
        config["corrections"][keyword] = ["Scale2G_IJazZ"]
        if "2022" in keyword:
            config["corrections"][keyword].append("jec_pnetNu_Run" + keyword[7:])
            config["corrections"][keyword].append("jec_AK8_Run" + keyword[7:])
        elif "2023" in keyword:
            if "Cv4" in keyword:
                config["corrections"][keyword].append("jec_pnetNu_RunCv4")
                config["corrections"][keyword].append("jec_AK8_RunCv4")
            elif "Dv" in keyword:
                config["corrections"][keyword].append("jec_pnetNu_RunD")
                config["corrections"][keyword].append("jec_AK8_RunD")
            else:
                config["corrections"][keyword].append("jec_pnetNu_RunCv123")
                config["corrections"][keyword].append("jec_AK8_RunCv123")
        elif "2024" in keyword:
            config["corrections"][keyword].append("jec_pnetNu_Data2024")
            config["corrections"][keyword].append("jec_AK8_Data2024")
        elif "2025" in keyword:
            config["corrections"][keyword].append("jec_pnetNu_Data2025")
            config["corrections"][keyword].append("jec_AK8_Data2025")
        elif "2018" or "2016" or "2017" in keyword:
            config["corrections"][keyword].append("jec_pnetNu_Run2_v15_Run" + keyword[7])
            config["corrections"][keyword].append("jec_AK8_Run2_v15_Run" + keyword[7])
            config["corrections"][keyword].append("Scale_Trad")
            if any("Scale2G_IJazZ" == corr for corr in config["corrections"][keyword]):
                config["corrections"][keyword].remove("Scale2G_IJazZ")
    if "systematics" in config:
        config["systematics"][keyword] = config["systematics"].pop("Run2023Cv1", [])
    new_filename = f"runner_data_{year}_{keyword}.json"
    with open(new_filename, "w") as f:
        json.dump(config, f, indent=4)

    return new_filename


def main():
    parser = argparse.ArgumentParser(description="Run Data production example pipeline.")
    parser.add_argument("-k", "--keyword", required=True, help="Keyword for dataset filtering")
    parser.add_argument("-c", "--cmsdas", required=True, help="Keyword for cmsdas filtering")
    parser.add_argument("-p", "--parent-dir", required=True, help="Directory to store output parquets")
    parser.add_argument("-y", "--year", required=True, choices=["2022postEE","2022preEE","2023postBPix","2023preBPix", "2024", "2025", "2018","2017","2016preVFP","2016postVFP"], help="year")
    parser.add_argument("-n", "--nano", required=True, help="nano-version")
    parser.add_argument("-m", "--memory", help="condor job memory")

    args = parser.parse_args()

    # Write dataset sample file
    sample_file = write_sample_txt(args.keyword, args.year, args.cmsdas)

    # Fetch datasets
    fetch_datasets(sample_file)

    # Update and save JSON configuration
    update_json_config(args.keyword, args.year)

    # Validate sample
    # validate_run_analysis(args.nano, args.parent_dir, args.keyword, args.year, args.memory)

    # Launch jobs
    run_analysis(args.nano, args.parent_dir, args.keyword, args.year, args.memory)


if __name__ == "__main__":
    main()
