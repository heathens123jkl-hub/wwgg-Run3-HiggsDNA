#!/usr/bin/env python
import argparse
from higgs_dna.utils.logger_utils import setup_logger
import uproot
import awkward as ak
import os
import json
import yaml
import numpy as np
from importlib import resources
from higgs_dna.scripts.postprocessing.tools.postprocessing_tools import split_awkward_arrays_by_length, ensure_nweight_LHEScale, make_tree, prepare_branch_array

def main():
    parser = argparse.ArgumentParser(
        description="Simple utility script to convert one parquet file into one ROOT file."
    )
    parser.add_argument("source", type=str, help="Path to input file.")
    parser.add_argument("target", type=str, help="Path to desired output file.")
    parser.add_argument("type", type=str, choices=["mc", "MC", "mC", "Mc", "data", "Data", "DATA"], help="Type of dataset (data or mc).")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Debugging verbosity for logger.")
    parser.add_argument("--process", type=str, default="", help="Production mode.")
    parser.add_argument(
        "--notag",
        dest="notag",
        action="store_true",
        default=False,
        help="create NOTAG dataset as well.",
    )
    parser.add_argument(
        "--do-syst",
        dest="do_syst",
        action="store_true",
        default=False,
        help="create branches for systematic variations",
    )
    parser.add_argument(
        "--cats",
        type=str,
        dest="cats_dict",
        default="",
        help="Dictionary containing category selections.",
    )
    parser.add_argument(
        "--vars",
        type=str,
        dest="vars_dict",
        default="",
        help="Dictionary containing variations.",
    )
    parser.add_argument(
        "--abs",
        dest="abs",
        action="store_true",
        default=False,
        help="Uses absolute path for the dictionary files.",
    )
    parser.add_argument(
        "--genBinning",
        type=str,
        dest="genBinning",
        default="",
        help="Optional: Path to the JSON containing the binning at gen-level.",
    )
    parser.add_argument(
        "--outfiles-map",
        dest="outfiles_map",
        type=str,
        default="",
        help="Path to YAML defining ROOT-output filename scheme."
    )
    parser.add_argument(
        "--tbasket-length",
        type=int,
        dest="tbasket_length",
        default=5000,
        help="Length of the tbasket in the ROOT file.",
    )
    args = parser.parse_args()
    source_path = args.source
    target_path = args.target
    notag = True if (args.type == "mc" and args.notag == True) else False
    process = args.process if (args.process != "") else "data"

    logger_verbosity = "DEBUG" if args.verbose else "INFO"

    logger = setup_logger(level=logger_verbosity)

    BASEDIR = resources.files("higgs_dna").joinpath("")

    if args.genBinning != "":
        if args.abs:
            genBinning_path = os.path.realpath(args.genBinning)
        else:
            genBinning_path = os.path.join(BASEDIR, "scripts/postprocessing/sample_gen_binning.json")
        with open(genBinning_path, 'r') as json_file:
            gen_binning = json.load(json_file)
    else:
        gen_binning = None

# Dictionary for renaming variables in ROOT tree output for final fits
    rename_dict = {
        "mass": "CMS_hgg_mass"
    }

    # Ensure that the target directory exists
    os.makedirs('/'.join(target_path.split("/")[:-1]), exist_ok=True)

    df_dict = {}
    # load outfile‐templates
    if args.outfiles_map:
        om_file = args.outfiles_map
    else:
        om_file = os.path.join(BASEDIR, "scripts/postprocessing/config_jsons/outfiles.yaml")

    with open(om_file, "r") as f:
        if om_file.endswith((".yml", ".yaml")):
            templates = yaml.safe_load(f)
        else:
            templates = json.load(f)

    # build actual filenames by substituting into target_path
    outfiles = {
        key: target_path.replace("merged.root", template)
        for key, template in templates.items()
    }

    # Loading category informations (used for naming of files to read/write)
    if args.cats_dict != "":
        if args.abs:
            cats_path = os.path.realpath(args.cats_dict)
        else:
            cats_path = os.path.join(BASEDIR, args.cats_dict)
        with open(cats_path) as pf:
            # with resources.open_text("higgs_dna", args.cats_dict) as pf:
            cat_dict = json.load(pf)
        for cat in cat_dict:
            logger.debug(f"Found category: {cat}")
    else:
        logger.info(
            "You provided an invalid dictionary containing categories information, have a look at your version of prepare_output_file.py"
        )
        logger.info(
            "An inclusive NOTAG category is used as default"
        )
        cat_dict = {"NOTAG": {"cat_filter": [("pt", ">", -1.0)]}}

# Loading variation informations (used for naming of files to read/write)
# Active object systematics, weight systematics are just different sets of weights contained in the nominal file
    if args.vars_dict != "":
        if args.abs:
            vars_path = os.path.realpath(args.vars_dict)
        else:
            vars_path = os.path.join(BASEDIR, args.vars_dict)
        # with resources.open_text("higgs_dna", args.vars_dict) as pf:
        with open(vars_path) as pf:
            variation_dict = json.load(pf)
        for var in variation_dict:
            logger.debug(f"Found variation: {var}")
    else:
        if args.do_syst:
            raise Exception(
                "You provided an invalid dictionary containing systematic variations information, have a look at your version of prepare_output_file.py"
            )

    if args.do_syst:
        # object systematics come from a different file (you are supposed to have merged .parquet with the merge_parquet.py script)
        for var in variation_dict:
            df_dict[var] = {}
            for cat in cat_dict:
                var_path = source_path.replace(
                    "merged.parquet", f"{variation_dict[var]}/{cat}_merged.parquet"
                )
                logger.info(
                    f"Starting conversion of one parquet file to ROOT. Attempting to read file {var_path} for category: {cat}."
                )

                eve = ak.from_parquet(var_path)
                
                # ------------------------------------------------------------------
                # Guarantee the counter branch for the ragged array "weight_LHEScale".
                # Uproot expects the counter to be present **and** of type int32.
                if "nweight_LHEScale" not in eve.fields:
                    logger.debug("nweight_LHEScale missing - creating with constant 9 (int32).")
                    # keep as plain NumPy to preserve native little‑endian dtype
                    eve["nweight_LHEScale"] = np.full(len(eve), 9, dtype=np.int32)
                else:
                    # Cast to int32 to satisfy uproot's leaf‑list requirement; keep as NumPy array
                    logger.debug("nweight_LHEScale exists - casting to int32 for ROOT compatibility.")
                    # keep as plain NumPy to preserve native little‑endian dtype
                    eve["nweight_LHEScale"] = np.asarray(eve["nweight_LHEScale"], dtype=np.int32)
                # ------------------------------------------------------------------

                logger.info("Successfully read from parquet file with awkward.")

                dict = {}
                for i in eve.fields:
                    i_re = rename_dict[i] if i in rename_dict else i
                    dict[i_re] = eve[i]

                df_dict[var][cat] = dict

                logger.debug(
                    f"Successfully created dict from awkward arrays for {var} variation for category: {cat}."
                )

        logger.info(f"Attempting to write dict to ROOT file {target_path}.")
    else:
        for cat in cat_dict:
            var_path = source_path.replace("merged.parquet", f"{cat}_merged.parquet")
            logger.info(
                f"Starting conversion of one parquet file to ROOT. Attempting to read file {var_path}."
            )

            eve = ak.from_parquet(var_path)
            
            # ------------------------------------------------------------------
            # Guarantee the counter branch for the ragged array "weight_LHEScale".
            # Uproot expects the counter to be present **and** of type int32.
            if "nweight_LHEScale" not in eve.fields:
                logger.debug("nweight_LHEScale missing - creating with constant 9 (int32).")
                # keep as plain NumPy to preserve native little‑endian dtype
                eve["nweight_LHEScale"] = np.full(len(eve), 9, dtype=np.int32)
            else:
                # Cast to int32 to satisfy uproot's leaf‑list requirement; keep as NumPy array
                logger.debug("nweight_LHEScale exists - casting to int32 for ROOT compatibility.")
                # keep as plain NumPy to preserve native little‑endian dtype
                eve["nweight_LHEScale"] = np.asarray(eve["nweight_LHEScale"], dtype=np.int32)
            # ------------------------------------------------------------------

            logger.info("Successfully read from parquet file with awkward.")

            dict = {}
            for i in eve.fields:
                i_re = rename_dict[i] if i in rename_dict else i
                dict[i_re] = eve[i]

            df_dict[cat] = dict

            logger.debug(
                f"Successfully created dict from awkward arrays without variation for category: {cat}."
            )

    cat_postfix = {"ggh": "GG2H", "vbf": "VBF", "tth": "TTH", "vh": "VH", "dy": "DY"}

# For MC: {inputTreeDir}/{production-mode}_{mass}_{sqrts}_{category}
# For data: {inputTreeDir}/Data_{sqrts}_{category}
    labels = {}
    names = {}
    if args.type == "mc":
        for cat in cat_dict:
            if len(process.split("_"))>1:
                # If process of the form {process}_{mass}
                names[
                    cat
                ] = f"DiphotonTree/{process.split('_')[0]}_{process.split('_')[-1]}_13TeV_{cat}"  # _"+cat_postfix[process]
            else:
                names[
                cat
                ] = f"DiphotonTree/{process}_125_13TeV_{cat}"  # _"+cat_postfix[process]
            labels[cat] = []
        if len(process.split("_"))>1:
            name_notag = "DiphotonTree/" + process.split('_')[0] + f"_{process.split('_')[-1]}_13TeV_NOTAG"
        else:
            name_notag = "DiphotonTree/" + process + "_125_13TeV_NOTAG"
        # flashggFinalFit needs to have each systematic variation in a different branch
        if args.do_syst:
            for var in variation_dict:
                for cat in cat_dict:
                    # for object systematics we have different files storing the variated collections with the nominal weights
                    syst_ = var
                    logger.info("found syst: %s for category: %s" % (syst_, cat))
                    if len(process.split("_"))>1:
                        labels[cat].append(
                            [
                                "DiphotonTree/" + process.split('_')[0] + f"_{process.split('_')[-1]}_13TeV_{cat}_" + syst_,
                                "weight",
                                syst_,
                                cat,
                            ]
                        )
                    else:
                        labels[cat].append(
                        [
                            "DiphotonTree/" + process + f"_125_13TeV_{cat}_" + syst_,
                            "weight",
                            syst_,
                            cat,
                        ]
                )

    else:
        for cat in cat_dict:
            labels[cat] = []
            labels[cat].append([f"DiphotonTree/Data_13TeV_{cat}", cat])
            names[cat] = f"DiphotonTree/Data_13TeV_{cat}"

    # Now we want to write the dictionary to a root file, since object systematics don't come from
    # the nominal file we have to separate again the treatment of them from the object ones
    with uproot.recreate(outfiles[process]) as file:
        logger.debug(outfiles[process])
        # Final fit want a separate tree for each category and variation,
        # the naming of the branches are quite rigid:
        # For MC: {inputTreeDir}/{production-mode}_{mass}_{sqrts}_{category}_{syst}
        # For data: {inputTreeDir}/Data_{sqrts}_{category}
        for cat in cat_dict:
            logger.debug(f"writing category: {cat}")

            if args.do_syst:
                # check that the category actually contains something before attempting to split/write
                # to avoid confusing uproot with completely empty structures
                if len(df_dict["NOMINAL"][cat]["weight"]):
                    split_nominal_dict = split_awkward_arrays_by_length(df_dict["NOMINAL"][cat], logger, target_length=int(args.tbasket_length))

                    for i, current_dict in enumerate(split_nominal_dict):
                        current_dict = ensure_nweight_LHEScale(current_dict)
                        logger.debug(f"Adding {i + 1}th dict out of {len(split_nominal_dict)}")

                        array_sizes = {key: arr.nbytes for key, arr in current_dict.items()}
                        logger.debug(f"Size of current_dict: {sum(array_sizes.values())} bytes")

                        if i == 0:
                            make_tree(file, names[cat], current_dict)
                        else:
                            file[names[cat]].extend(current_dict)
                    
                    for syst_name, weight, syst_, c in labels[cat]:
                        # Skip "NOMINAL" as information included in nominal tree
                        if syst_ == "NOMINAL":
                            continue
                        logger.debug(f"{syst_name}, {weight}, {syst_}, {c}")
                        # If the name is not in the variation dictionary it is assumed to be a weight systematic
                        var_list = [
                                ["CMS_hgg_mass", "CMS_hgg_mass"],
                                [weight, "weight"],
                                ["HTXS_Higgs_pt", "HTXS_Higgs_pt"],
                                ["HTXS_Higgs_y", "HTXS_Higgs_y"],
                                ["PTH", "PTH"],
                                ["YH", "YH"],
                                ["fiducialGeometricFlag", "fiducialGeometricFlag"]
                            ]
                        if gen_binning != None:
                            for keys in gen_binning:
                                var_list.append(["diffVariable_" + keys, "diffVariable_" + keys])

                        if syst_ not in variation_dict:
                            logger.debug(f"found weight syst {syst_}")
                            red_dict = {}
                            for key, new_key in var_list:
                                if "NOMINAL" in df_dict and cat in df_dict["NOMINAL"] and key in df_dict["NOMINAL"][cat]:
                                    red_dict[new_key] = df_dict["NOMINAL"][cat][key]

                            logger.info(f"Adding {syst_name}01sigma to out tree...")

                            split_dict = split_awkward_arrays_by_length(red_dict, logger, target_length=int(args.tbasket_length))
                            for i, current_dict in enumerate(split_dict):
                                current_dict = ensure_nweight_LHEScale(current_dict)
                                logger.debug(f"Adding {i + 1}th dict out of {len(split_dict)}")

                                array_sizes = {key: arr.nbytes for key, arr in current_dict.items()}
                                logger.debug(f"Size of current_dict: {sum(array_sizes.values())} bytes")

                                if i == 0:
                                    make_tree(file, syst_name + "01sigma", current_dict)
                                else:
                                    file[syst_name + "01sigma"].extend(current_dict)

                        else:
                            red_dict = {}
                            for key, new_key in var_list:
                                if syst_ in df_dict and cat in df_dict[syst_] and key in df_dict[syst_][cat]:
                                    red_dict[new_key] = df_dict[syst_][cat][key]

                            split_dict = split_awkward_arrays_by_length(red_dict, logger, target_length=int(args.tbasket_length))

                            logger.info(f"Adding {syst_name}01sigma to out tree...")
                            for i, current_dict in enumerate(split_dict):
                                current_dict = ensure_nweight_LHEScale(current_dict)
                                logger.debug(f"Adding {i + 1}th dict out of {len(split_dict)}")

                                array_sizes = {key: arr.nbytes for key, arr in current_dict.items()}
                                logger.debug(f"Size of current_dict: {sum(array_sizes.values())} bytes")

                                if i == 0:
                                    make_tree(file, syst_name + "01sigma", current_dict)
                                else:
                                    file[syst_name + "01sigma"].extend(current_dict)

                else:
                    logger.info(f"no events survived category selection for cat: {cat}")

            else:
                # if there are no syst there is no df_dict["NOMINAL"] entry in the dict
                if len(df_dict[cat][[*df_dict[cat]][0]]):

                    logger.info(f"Adding cat {cat} to ROOT file...")

                    # single-shot write: TTree.extend yields wrong entry counts in this
                    # uproot version, so write the whole category in one assignment
                    full_dict = {key: prepare_branch_array(array) for key, array in df_dict[cat].items()}
                    full_dict = ensure_nweight_LHEScale(full_dict)
                    make_tree(file, names[cat], full_dict)

                    if notag:
                        # this is wrong, to be fixed
                        logger.info("Adding also NOTAG to ROOT file...")

                        split_dict = split_awkward_arrays_by_length(df_dict[cat], logger, target_length=int(args.tbasket_length))
                        for i, current_dict in enumerate(split_dict):
                            current_dict = ensure_nweight_LHEScale(current_dict)
                            logger.debug(f"Adding {i + 1}th dict out of {len(split_dict)}")

                            array_sizes = {key: arr.nbytes for key, arr in current_dict.items()}
                            logger.debug(f"Size of current_dict: {sum(array_sizes.values())}")

                            if i == 0:
                                make_tree(file, names[cat], current_dict)
                            else:
                                file[names[cat]].extend(current_dict)
                else:
                    logger.info(f"no events survived category selection for cat: {cat}")

        logger.info(
            f"Successfully converted parquet file to ROOT file for process {process}."
        )

if __name__ == "__main__":
    main()
