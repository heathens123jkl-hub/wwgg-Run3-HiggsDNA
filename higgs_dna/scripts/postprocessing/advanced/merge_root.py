#!/usr/bin/env python
import argparse
import json
import yaml
import ast
import os
import glob
import awkward as ak
from higgs_dna.utils.logger_utils import setup_logger
import pyarrow.parquet as pq
import numpy as np
import uproot
from importlib import resources
from higgs_dna.scripts.postprocessing.tools.Btag_WeightSum_Calculation import Get_WeightSum_Btag, Renormalize_BTag_Weights, Get_bin_edges_and_ration, apply_rescaling, Get_ratio_with_bWeight
from higgs_dna.scripts.postprocessing.tools.LHE_WeightSum_Calculation import Get_WeightSum_LHE, Renormalize_LHE_Weights
from higgs_dna.scripts.postprocessing.tools.postprocessing_tools import filter_and_set_diff_variable, split_awkward_arrays_by_length, ensure_nweight_LHEScale, make_tree

def get_active_gen_binning(gen_binning, diff_variable, logger):
    if gen_binning is None:
        return None

    if not diff_variable:
        return gen_binning

    if diff_variable in gen_binning:
        return {diff_variable: gen_binning[diff_variable]}

    logger.warning(
        f"Differential variable '{diff_variable}' was requested but not found in genBinning keys: {list(gen_binning.keys())}. "
        "Falling back to all genBinning entries."
    )
    return gen_binning

def get_dataset(_args, folder_path, cat, is_data, is_syst, source_path, target_path, cat_dict, gen_binning, logger, rename_dict):

    renamed_dict = {}

    if (not is_data) & (not _args.skip_normalisation):
        logger.info(
            "Extracting sum of gen weights (before selection) from metadata of files to be merged."
        )

        if _args.do_b_weight_normalisation:
            IsBtagNorm_sys_arr, WeightSum_preBTag_arr, WeightSum_postBTag_arr, WeightSum_postBTag_sys_arr = Get_WeightSum_Btag([folder_path], logger)
        if _args.do_theory_weight_normalisation:
            sum_LHEPdf_beforesel_arr, sum_LHEScale_beforesel_arr, do_lhe_norm = Get_WeightSum_LHE([folder_path], logger)
        source_files = glob.glob("%s/*.parquet" % folder_path)
        sum_genw_beforesel = 0
        for f in source_files:
            sum_genw_beforesel += float(pq.read_schema(f).metadata[b'sum_genw_presel'])
        logger.debug(f"sum_genw_beforesel {(sum_genw_beforesel)}")
        logger.info(
            "Successfully extracted sum of gen weights (before selection)"
        )

    logger.info("-" * 125)
    logger.info(
        f"Attempting to read ParquetDataset from {folder_path}, for category: {cat}"
    )
    if is_data and _args.merge_data:
        path = glob.glob(os.path.join(source_path, '**', '*.parquet'), recursive=True)
    else:
        path = folder_path
    dataset = pq.ParquetDataset(path, filters=cat_dict[cat]["cat_filter"])
    logger.info("ParquetDataset read successfully.")

    # If syst, read less branches to save memory
    if is_syst:
        columns_to_read = ["mass", "weight", "fiducialGeometricFlag"]
        table = dataset.read(columns=columns_to_read)

    # Load the piece into an Awkward Array
    table = dataset.read()
    eve = ak.from_arrow(table)
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

    # If MC then open the merged dataset and add normalised weight column (sumw = efficiency)
    # TODO: can we add column before writing table and prevent re-reading in as awkward array
    if (not is_data) & (not _args.skip_normalisation):
        # Add filtering for differentials here
        if gen_binning != None:
            for keys in gen_binning:
                var_dict = {ast.literal_eval(key): value for key, value in gen_binning[keys].items()}
                # If the length of the gen_binning tuple is 5 => Use first element in the tuple as primary selection variable
                # Example: ('GenPTH', 0, 15, 'in', '(GenDPhiJ0J1, >=, -3.1416);(GenDPhiJ0J1, <, -2.0944)')
                if len(list(var_dict.keys())[0]) == 5:
                    selectionVariableName = list(var_dict.keys())[0][0]
                    # Remove the first element from the Tuples, as it is defined in selectionVariableName
                    var_dict = {k[1:]: v for k, v in var_dict.items()}
                else:
                    selectionVariableName = keys
                if selectionVariableName not in eve.fields:
                    logger.warning(
                        f"Skipping gen-binning variable '{keys}' because selection field '{selectionVariableName}' is not present in input parquet fields."
                    )
                    continue
                eve = filter_and_set_diff_variable(eve, var_dict, selectionVariableName, "diffVariable_" + keys)
        syst_weight_fields = [field for field in eve.fields if (("weight_" in field) and ("Up" in field or "Down" in field))]
        logger.debug(f"Found systematic weight fields: {syst_weight_fields}")
        # Add column for unnormalised weight
        eve['weight_nominal'] = eve['weight']
        if len(eve) > 0:
            for weight_field in ["weight"] + syst_weight_fields:
                eve[weight_field] = eve[weight_field] / sum_genw_beforesel
            logger.info(
                "Successfully added normalised weight column and normalized weights of the systematics to dataset"
            )
            if _args.do_b_weight_normalisation:
                if (WeightSum_preBTag_arr[0] / WeightSum_postBTag_arr[0]) != 1:
                    eve = Renormalize_BTag_Weights(eve,syst_weight_fields, target_path, cat, WeightSum_preBTag_arr[0], WeightSum_postBTag_arr[0], WeightSum_postBTag_sys_arr[0], IsBtagNorm_sys_arr[0], logger)
                logger.info(
                    "Successfully added normalised b weight column."
                )
            if _args.do_theory_weight_normalisation:
                if do_lhe_norm[0]:
                    eve = Renormalize_LHE_Weights(eve, target_path, cat, sum_LHEPdf_beforesel_arr[0], sum_LHEScale_beforesel_arr[0], logger)
                    logger.info(
                        "Successfully added normalised LHE weight columns."
                    )
            if(_args.BTagRescaleVariableInfo):
                if not _args.do_b_weight_normalisation:
                    logger.warning("B-Tag weight rescaling requested but B-Tag weight normalisation not performed. Skipping B-Tag weight rescaling. Please enable --do-b-weight-normalisation to perform B-Tag weight reNormalization first.")
                    exit(0)
                if(cat == "NOTAG"):
                    BTagRescaleVariable_Info = _args.BTagRescaleVariableInfo.split(',')
                    BTagRescaleVariable_Info[1:] = [int(i) for i in BTagRescaleVariable_Info[1:]]
                    if len(BTagRescaleVariable_Info) !=4:
                        raise Exception("Wrong format for BTagRescaleVariableInfo, please provide info in the following format: 'VariableName,nbins,min,max'")
                    logger.info("Starting B-Tag weight rescaling process")
                    xaxis_edges,ratio_val = Get_bin_edges_and_ration(eve,target_path,logger,Variable_info=BTagRescaleVariable_Info,plot_name="bTagWeight")

                    rescaled_weights_dict = apply_rescaling(eve,['weight',"bTagWeight"]+syst_weight_fields,xaxis_edges,ratio_val,logger,Variable_info=BTagRescaleVariable_Info)
                    for weight_field in ["weight","bTagWeight"]+syst_weight_fields:
                            #With rescaled weight we have to make sure the total sum of the weight conserved.
                            weight_factor = ak.sum(eve[weight_field])/ak.sum(rescaled_weights_dict[weight_field])
                            eve[weight_field] = rescaled_weights_dict[weight_field]*weight_factor

                    Get_ratio_with_bWeight(eve,BTagRescaleVariable_Info,bweight_name="bTagWeight",plot_name=target_path+f"/{BTagRescaleVariable_Info[0]}_bTagWeight_rescaled",plot_ratio_min=0.9,plot_ratio_max=1.2)
                else:
                        logger.warning(f"skiping the B-Weight rescaling. The scale can be derived in bins of {_args.BTagRescaleVariableInfo} only for NOTAG category sinace we need to derive the scale before we apply any cut")
        else:
            logger.info(
                "No events survived category selection. Skipping normalisation step."
            )

    # Rename fields and store them in the dictionary
    for field_name in eve.fields:
        new_field_name = rename_dict.get(field_name, field_name)
        renamed_dict[new_field_name] = eve[field_name]
    logger.info("-" * 125)
    return renamed_dict

def create_empty_tree(keys=["CMS_hgg_mass", "nweight_LHEScale"]):
    """
    Create a dict with an empty awkward array for all expected keys.
    """
    empty_dict = {}
    for key in keys:
        if key == "nweight_LHEScale":
            empty_dict[key] = np.array([], dtype=np.int32)
        else:
            empty_dict[key] = np.array([], dtype=np.float64)
    return empty_dict

def main():
    parser = argparse.ArgumentParser(
        description="Simple utility script to merge all parquet files in one folder."
    )
    parser.add_argument(
    "--source",
    type=str,
    default="",
    help="Comma separated paths (with trailing slash) to folder where multiple parquet files are located. Careful: Folder should ONLY contain parquet files!",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Comma separated paths (with trailing slash) to desired folder. Resulting merged file is placed there.",
    )
    parser.add_argument(
        "--cats",
        type=str,
        dest="cats_dict",
        default="",
        help="Dictionary containing category selections.",
    )
    parser.add_argument(
        "--type",
        type=str,
        dest="type",
        default="",
        help="Type of dataset (data or mc).",
    )
    parser.add_argument(
        "--notag",
        dest="notag",
        action="store_true",
        default=False,
        help="create NOTAG dataset as well.",
    )
    parser.add_argument("--process", type=str, default="", help="Production mode.")
    parser.add_argument(
        "--vars",
        type=str,
        dest="vars_dict",
        default="",
        help="Dictionary containing variations.",
    )
    parser.add_argument(
        "--do-syst",
        dest="do_syst",
        action="store_true",
        default=False,
        help="create branches for systematic variations",
    )
    parser.add_argument(
        "--merge-all-data",
        dest="merge_data",
        action="store_true",
        default=False,
        help="Flag if all eras should be merged.",
    )
    parser.add_argument(
        "--skip-normalisation",
        default=False,
        action="store_true",
        help="Independent of file type, skip normalisation step",
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
        "--diff-variable",
        type=str,
        dest="diff_variable",
        default="",
        help="Optional: Differential variable key to select from genBinning (e.g. 'PTH').",
    )
    parser.add_argument(
        "--do-b-weight-normalisation",
        default=False,
        action="store_true",
        help="Perform the bweight normalization to make sure the number of event remain the same before and after applying the b tagging weights",
    )
    parser.add_argument(
        "--BTagRescaleVariableInfo",
        nargs="?",
        const="n_jets,10,0,10",
        default=None,
        type=str,
        help="Rescaling variable info. If passed with no value, defaults to 'n_jets,10,0,10', other variable and bin info can be provided 'JetHT,50,0,1000' ",
    )
    parser.add_argument(
        "--do-theory-weight-normalisation",
        dest="do_theory_weight_normalisation",
        default=False,
        action="store_true",
        help="Perform theory-weight normalization (LHEScale/LHEPdf/AlphaS/PS) to focus on acceptance effects.",
    )
    parser.add_argument(
        "--do-lhe-weight-normalisation",
        dest="do_theory_weight_normalisation",
        default=False,
        action="store_true",
        help="DEPRECATED: use --do-theory-weight-normalisation.",
    )
    parser.add_argument(
        "--outfiles-map",
        dest="outfiles_map",
        type=str,
        default=None,
        help="Path to YAML/JSON defining ROOT-output filename templates."
    )
    parser.add_argument(
        "--tbasket-length",
        type=int,
        dest="tbasket_length",
        default=5000,
        help="Length of the tbasket in the ROOT file.",
    )
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Debugging verbosity for logger.")
    args = parser.parse_args()
    source_path = args.source
    target_path = args.target

    BASEDIR = resources.files("higgs_dna").joinpath("")

    logger_verbosity = "DEBUG" if args.verbose else "INFO"

    logger = setup_logger(level=logger_verbosity)

    # load outfiles templates
    if args.outfiles_map:
        ofm_file = args.outfiles_map
    else:
        ofm_file = os.path.join(BASEDIR, "scripts/postprocessing/config_jsons/outfiles.yaml")

    with open(ofm_file, "r") as f:
        if ofm_file.endswith((".yml", ".yaml")):
            templates = yaml.safe_load(f)
        else:
            templates = json.load(f)

    # now build the actual outfiles dict by replacing "merged.root"
    outfiles = {
        key: target_path.replace("merged.root", tpl)
        for key, tpl in templates.items()
    }
    # ────────────────────────────────────────────────────────────────────────
 

    # Create target directory if it does not exist
    os.makedirs("/".join(target_path.split('/')[:-1]), exist_ok=True)

    notag = True if (args.type == "mc" and args.notag == True) else False
    process = args.process if (args.process != "") else "data" 
    is_data = (args.type == "data") or (args.type == "Data")

    if args.genBinning != "":
        if args.abs:
            genBinning_path = args.genBinning
        else:
            genBinning_path = os.path.join(BASEDIR, "scripts/postprocessing/sample_gen_binning.json")
        with open(genBinning_path, 'r') as json_file:
            gen_binning = json.load(json_file)
        gen_binning = get_active_gen_binning(gen_binning, args.diff_variable, logger)
    else:
        gen_binning = None

    rename_dict = {
        "mass": "CMS_hgg_mass"
    }

    if args.cats_dict != "":
        if args.abs:
            cats_path = args.cats_dict
        else:
            cats_path = os.path.join(BASEDIR, "category.json")
        with open(cats_path) as pf:
            cat_dict = json.load(pf)
        for cat in cat_dict:
            logger.info(f"Found category: {cat}")
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
            vars_path = args.vars_dict
        else: 
            vars_path = os.join(BASEDIR, args.vars_dict)
        with open(vars_path) as pf:
            variation_dict = json.load(pf)
        for var in variation_dict:
            logger.debug(f"Found variation: {var}")
    else:
        if args.do_syst:
            raise Exception(
                "You provided an invalid dictionary containing systematic variations information, have a look at your version of merge_root.py"
            )

    df_dict = {}
    if not args.merge_data:
        if args.do_syst or is_data:
            for var, var_value in variation_dict.items():
                df_dict[var] = {}
                logger.info(
                f"Variation: {var}"
                )
                for subfolder in os.listdir(source_path):
                    subfolder_path = os.path.join(source_path, subfolder)
                    if var_value in subfolder_path:
                        if os.path.isdir(subfolder_path):
                            for cat, _ in cat_dict.items():
                                if args.do_syst and not is_data:
                                    is_syst = True
                                else:
                                    is_syst = False
                                dict = get_dataset(args, subfolder_path, cat, is_data, is_syst, source_path, target_path, cat_dict, gen_binning, logger, rename_dict)
                                df_dict[var][cat] = dict

                    else:
                        continue
        else:
            df_dict["NOMINAL"] = {}
            for cat, _ in cat_dict.items():
                subfolder_path = os.path.join(source_path, "nominal")
                dict = get_dataset(args, subfolder_path, cat, is_data, False, source_path, target_path, cat_dict, gen_binning, logger, rename_dict)
                df_dict["NOMINAL"][cat] = dict

    else:
        for var, var_value in variation_dict.items():
            df_dict[var] = {}
            for cat, _ in cat_dict.items():
                if var == "NOMINAL":
                    is_syst = False
                else:
                    is_syst = True
                dict = get_dataset(args, source_path, cat, is_data, is_syst, source_path, target_path, cat_dict, gen_binning, logger, rename_dict)
                df_dict[var][cat] = dict


    labels = {}
    names = {}
    if args.type == "mc":
        for cat in cat_dict:
            if len(process.split("_"))>1:
                # If process of the form {process}_{mass}
                names[
                    cat
                ] = f"DiphotonTree/{process.split('_')[0]}_{process.split('_')[-1]}_13TeV_{cat}"
            else:
                names[
                cat
                ] = f"DiphotonTree/{process}_125_13TeV_{cat}"
            labels[cat] = []
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
            logger.debug(f"Writing category: {cat}")
            # in case the current category is empty, we use this to get the field names later
            fallback = next((c for c in cat_dict if len(df_dict["NOMINAL"][c]["weight"]) > 0), None)
            if fallback is not None:
                fallback_field_names = list(df_dict["NOMINAL"][fallback].keys())
            else:
                logger.info("No non-empty category found!")
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
                                # ["PTH", "PTH"],
                                # ["YH", "YH"],
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
                    logger.info(f"No events survived category selection for cat: {cat}. Empty tree will be written.")
                    make_tree(file, names[cat], create_empty_tree(fallback_field_names))

                    # now for each syst variation, do the same empty tree
                    for syst_name, weight, syst_, c in labels[cat]:
                        if syst_ == "NOMINAL":
                            continue
                        make_tree(file, syst_name + "01sigma", create_empty_tree(fallback_field_names))

            else:
                # if there are no syst there is no df_dict["NOMINAL"] entry in the dict
                if len(df_dict["NOMINAL"][cat][[*df_dict["NOMINAL"][cat]][0]]):
                    split_nominal_dict = split_awkward_arrays_by_length(df_dict["NOMINAL"][cat], logger, target_length=int(args.tbasket_length))

                    for i, current_dict in enumerate(split_nominal_dict):
                        current_dict = ensure_nweight_LHEScale(current_dict)
                        logger.debug(f"Adding {i + 1}th dict out of {len(split_nominal_dict)}")

                        array_sizes = {key: arr.nbytes for key, arr in current_dict.items()}
                        logger.debug(f"Size of current_dict: {sum(array_sizes.values())} bytes")

                        if i == 0:
                            current_dict = ensure_nweight_LHEScale(current_dict)
                            make_tree(file, names[cat], current_dict)
                        else:
                            current_dict = ensure_nweight_LHEScale(current_dict)
                            file[names[cat]].extend(current_dict)

                    if notag: # this is wrong, to be fixed
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
                else:
                    logger.info(f"No events survived category selection for cat: {cat}. Empty tree will be written.")
                    make_tree(file, names[cat], create_empty_tree(fallback_field_names))

        logger.info(
            f"Successfully wrote ROOT file for process {process}."
        )

if __name__ == "__main__":
    main()
