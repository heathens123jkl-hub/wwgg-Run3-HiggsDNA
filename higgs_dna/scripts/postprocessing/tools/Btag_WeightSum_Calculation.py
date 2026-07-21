import glob
import awkward as ak
import pyarrow.parquet as pq

import hist
from hist import Hist
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
hep.style.use(hep.style.CMS)
from pathlib import Path

def Get_WeightSum_Btag(source_paths,logger):
    # All systematic variation list
    bTag_sys_variation = ['lfstats1', 'hfstats2' , 'jes', 'cferr2', 'lf', 'hf', 'lfstats2', 'hfstats1', 'cferr1']
    sum_weight_central_arr, sum_weight_central_wo_bTagSF_arr = [], []
    sum_weight_bTagSF_sys_arr = []
    flag_bWeight_sys_array = []
    


    for i, source_path in enumerate(source_paths):
        # create metadata object to access the column names of the parquet files
        dataset_columns = ak.metadata_from_parquet(glob.glob("%s/*.parquet" % source_path)[0])["columns"]
        # check if the b-tag systematic weight column is present in the dataset
        flag_bWeight_sys = "weight_bTagSF_sys_jesDown" in dataset_columns
        del dataset_columns
        if (flag_bWeight_sys):
            logger.info(
                f"Attampeting Extracting sum of central weights and bweight systematics from metadata of files to be merged from {source_path}"
            )
        else:
            logger.info(
                "Skiping the renormalization of b-tagging systematic weights. Please check if you have stored the weights for bTag systematic variation. Dont worry if you are not evaluating btaging systematic for now"
            )
        source_files = glob.glob("%s/*.parquet" % source_path)
        sum_weight_central,sum_weight_central_wo_bTagSF = 0,0
        sum_weight_bTagSF_sys_dct = {}
        if (flag_bWeight_sys):
            # dictionory to store up and down variation together
            for numSys in range(0, len(bTag_sys_variation)):
                sum_weight_bTagSF_sys_dct["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Up"] = 0
                sum_weight_bTagSF_sys_dct["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Down"] = 0

        for f in source_files:
            try:
                # read the sum of the weights from metadata without any systematic variation
                sum_weight_central += float(pq.read_schema(f).metadata[b'sum_weight_central'])
                sum_weight_central_wo_bTagSF += float(pq.read_schema(f).metadata[b'sum_weight_central_wo_bTagSF'])
            except:
                logger.info(
                    "Skiping the renormalization of weights from b-tagging systematics. Please check if you have stored sum of the weights after applying the b-weight systematics in the metadata with proper naming. Example: sum_weight_bTagSF_jesUp, sum_weight_bTagSF_jesDown."
                )
                # return sum of the weights before and after b-weight to 1 so that the ration will be one  and merge_parquet.py will not process renormalization
                sum_weight_central,sum_weight_central_wo_bTagSF = 1.0,1.0 
            if (flag_bWeight_sys):
                for numSys in range(0, len(bTag_sys_variation)):
                    try:
                        # read the sum of the weights from metadata for all systematic variation
                        sum_weight_bTagSF_sys_dct["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Up"] += float(pq.read_schema(f).metadata[bytes('sum_weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Up',encoding='utf8')])
                        sum_weight_bTagSF_sys_dct["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Down"] += float(pq.read_schema(f).metadata[bytes('sum_weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Down',encoding='utf8')])
                    except:
                        logger.info(
                            "Skiping the renormalization of weights from btagging systematics. Please check if you have stored sum of the weights after appling the bweight systematics in the metadata with proper nameing : example: sum_weight_bTagSF_jesUp, sum_weight_bTagSF_jesDown"
                        )
                        flag_bWeight_sys = False
                        break

        sum_weight_central_arr.append(sum_weight_central)
        sum_weight_central_wo_bTagSF_arr.append(sum_weight_central_wo_bTagSF)

        flag_bWeight_sys_array.append(flag_bWeight_sys)
        sum_weight_bTagSF_sys_arr.append(sum_weight_bTagSF_sys_dct)
        
    logger.info(
        "Successfully extracted sum of weights with and without b-tag weights."
    )
    if (flag_bWeight_sys):
        logger.info(
            "Successfully extracted sum of systematic weights with and without b-tag SF"
        )

    IsBtagNorm_sys_arr,WeightSum_preBTag_arr,WeightSum_postBTag_arr,dir_WeightSum_postBTag_sys_arr = flag_bWeight_sys_array, sum_weight_central_wo_bTagSF_arr, sum_weight_central_arr,sum_weight_bTagSF_sys_arr
    return IsBtagNorm_sys_arr,WeightSum_preBTag_arr,WeightSum_postBTag_arr,dir_WeightSum_postBTag_sys_arr

def Renormalize_BTag_Weights(dataset,Allweight_fields,target_path,cat,WeightSum_preBTag,WeightSum_postBTag,WeightSum_postBTag_sys,IsBtagNorm_sys,logger):
    bTag_sys_variation = ['lfstats1', 'hfstats2' , 'jes', 'cferr2', 'lf', 'hf', 'lfstats2', 'hfstats1', 'cferr1']
    other_sys_weight_fields = [field for field in Allweight_fields if "bTagSF" not in field]
    logger.info(
        f"Attempting to renormalize the weights wrt no b-tag SF from {target_path}{cat}_merged.parquet"
    )
    # Modify existing column

    if (WeightSum_preBTag != 0 and WeightSum_postBTag != 0):
        Norm_nominal_factor = WeightSum_preBTag / WeightSum_postBTag
        for weight_field in ['weight','bTagWeight']+other_sys_weight_fields:
            dataset[weight_field] = dataset[weight_field] * Norm_nominal_factor
        logger.info(
            f"Successfully renormalised weights wrt no b-tag SF from {target_path}{cat}_merged.parquet"
        )
    else:
        logger.info(
            f"Skipping weights renormalisation wrt No bTagSF from {target_path}{cat}_merged.parquet"
        )
    if (IsBtagNorm_sys):
        for numSys in range(0, len(bTag_sys_variation)):
            dataset['weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Up'] = dataset['weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Up'] * (WeightSum_preBTag / WeightSum_postBTag_sys["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Up"])
            dataset['weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Down'] = dataset['weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Down'] * (WeightSum_preBTag / WeightSum_postBTag_sys["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Down"])
        logger.info(
            f"Successfully renormalised bTagSF systematic weights wrt no b-tag SF from {target_path}{cat}_merged.parquet"
                )
    else:
        logger.info(
            f"Skipping systematic weights renormalisation wrt no b-tag SF from {target_path}{cat}_merged.parquet"
        )
    return dataset

def get_ratio(plt_numerator, plt_denominator):
    val_num = plt_numerator.values()
    var_num = plt_numerator.variances()
    val_denom = plt_denominator.values()
    val_denom = np.where(val_denom == 0, np.nan, val_denom)
    var_denom = plt_denominator.variances()
    var_denom = np.where(var_denom == 0, np.nan, var_denom)
    ratio_val_num = val_num / val_denom
    ratio_val_num = np.clip(ak.nan_to_num(ratio_val_num, nan=1), -9999.0, 9999.0)
    # error calculation
    ratio_err_num = np.sqrt(var_num) / val_denom
    ratio_err_num = np.clip(ak.nan_to_num(ratio_err_num, nan=1), -9999.0, 9999.0)

    return ratio_val_num, ratio_err_num


def Get_ratio_with_bWeight(dataset,Variable_info,bweight_name,plot_name="Unnamed",plot_ratio_min=0.8,plot_ratio_max=1.8):
    var_parquet = Variable_info[0]
    bweight_parquet = bweight_name

    # define the histogram
    n_bins = Variable_info[1]
    x_low =  Variable_info[2]
    x_high = Variable_info[3]
    x_axies_name = Variable_info[0]

    Hist_with_bweight = (
        Hist.new.Reg(n_bins, x_low, x_high, overflow=False, underflow=False, name=x_axies_name)
        .Weight()
    )
    Hist_without_bweight = (
        Hist.new.Reg(n_bins, x_low, x_high, overflow=False, underflow=False, name=x_axies_name)
        .Weight()
    )
    # Filling histogram
    Hist_with_bweight.fill(dataset[var_parquet],weight=dataset[bweight_parquet])
    Hist_without_bweight.fill(dataset[var_parquet])

    # Plot
    f, ax = plt.subplots(
        2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08), sharex=True
    )

    Hist_with_bweight.plot(label="w b-weight",ax=ax[0])
    Hist_without_bweight.plot(label="w/o b-weight",ax=ax[0])

    # store edges from the exiting hist
    xaxis_edges = Hist_without_bweight.axes.edges[0]

    # Get the ratio plot
    ratio_val, ratio_err = get_ratio(plt_numerator=Hist_without_bweight,plt_denominator=Hist_with_bweight)
    # Plot ratios
    hep.histplot(
        ratio_val,
        bins=xaxis_edges,
        # label=f"{dataset}/{denom_name}",
        yerr=np.abs(ratio_err),
        histtype="errorbar",
        stack=False,
        color="black",
        ax=ax[1],
        marker="o",
        markersize=4,
        elinewidth=2,
    )

    ax[1].axhline(y=1, linestyle="--", color="k", linewidth=1, alpha=0.8)
    ax[1].set_xlabel(ax[0].get_xlabel())
    ax[1].set_ylim(plot_ratio_min,plot_ratio_max)
    ax[1].set_ylabel("ratio",loc="center")

    ax[0].set_ylabel('Events/bin')
    ax[0].set_xlabel("")
    ax[0].legend()
    ax[0].set_xlim(x_low, x_high)
    ax[0].set_ylim(bottom=0)

    # directory for output plots
    plots_dir = './'
    full_path = Path(plots_dir) / f"{plot_name}.png"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path, bbox_inches="tight")
    return ratio_val.to_list()

def Get_bin_edges_and_ration(dataset,target_paths,logger,Variable_info,plot_name="bTagWeight"):
    Is_Variable_exit = Variable_info[0] in dataset.fields and  "bTagWeight" in dataset.fields
    if (Is_Variable_exit):
        logger.info(
            f"Attampeting to rederive the scale in bins of {Variable_info[0]} in range [{Variable_info[2]},{Variable_info[3]}] with  {Variable_info[1]} bins "
        )
    else:
        logger.info(
            f"Skiping the renormalization because {Variable_info[0]} or bTagWeight does not exist in fields "
        )
        exit(0)
    ratio_val = Get_ratio_with_bWeight(dataset,Variable_info,bweight_name="bTagWeight",plot_name=str(Path(target_paths))+f"/{Variable_info[0]}_{plot_name}",plot_ratio_min=0.9,plot_ratio_max=1.2)
    xaxis_edges = np.arange(Variable_info[2], Variable_info[3]+(Variable_info[3]-Variable_info[2])/Variable_info[1],(Variable_info[3]-Variable_info[2])/Variable_info[1])
    return xaxis_edges,ratio_val

def apply_rescaling(batch_arr,Allweight_fields,xaxis_edges,ratio_val,logger,Variable_info):
    logger.info(
            f"Attampeting to apply the scale in bins of {Variable_info[0]} in range [{Variable_info[2]},{Variable_info[3]}] with  {Variable_info[1]} bins "
        )
    OnesLike = ak.ones_like(batch_arr["bTagWeight"])
    rescaled_weights_dict = {}
    n_bins = len(ratio_val)
    for i,ratio in enumerate(ratio_val):
        if i == 0:
            Mask = (batch_arr[Variable_info[0]] < xaxis_edges[i+1])
        elif i == n_bins - 1:
            Mask = (batch_arr[Variable_info[0]] >= xaxis_edges[i])
        else:
            Mask = (batch_arr[Variable_info[0]] >= xaxis_edges[i]) & (batch_arr[Variable_info[0]] < xaxis_edges[i+1])

        for weight_field in Allweight_fields:
            if weight_field not in rescaled_weights_dict:
                rescaled_weights_dict[weight_field] = []
            rescaled_weights_dict[weight_field].append(ak.where(Mask, batch_arr[weight_field]*ratio, OnesLike))

    for weight_field in Allweight_fields:
        rescaled_weights_dict[weight_field] = ak.prod(rescaled_weights_dict[weight_field],axis=0)

    logger.info(
            f"Successfully applied the scale "
        )
    return rescaled_weights_dict