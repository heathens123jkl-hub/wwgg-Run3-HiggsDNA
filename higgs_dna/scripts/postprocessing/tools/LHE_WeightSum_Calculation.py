import glob
import awkward as ak
import pyarrow.parquet as pq


def Get_WeightSum_LHE(source_paths, logger):
    sum_LHEScale_fields, index_sum_LHEScale, sum_LHEPdf_fields, index_sum_LHEPdf = None, None, None, None
    sum_LHEPdf_beforesel_arr, sum_LHEScale_beforesel_arr, do_lhe_norm = [], [], []
    flag_has_lhe_weights = False

    for i, source_path in enumerate(source_paths):
        source_files = glob.glob("%s/*.parquet" % source_path)

        # Check if this is a nominal path (should be normalized)
        if 'nominal' not in source_path:
            logger.info(
                f"Skipping LHE weight normalization for non-nominal dataset: {source_path}"
            )
            do_lhe_norm.append(False)
            sum_LHEPdf_beforesel_arr.append([])
            sum_LHEScale_beforesel_arr.append([])
            continue

        # Check if LHE weights are stored by accessing one field of the first parquet file
        dataset_check_fields = ak.metadata_from_parquet(source_files[0])["columns"]
        flag_has_lhe_weights = "weight_LHEScale" in dataset_check_fields.fields and "weight_LHEPdf" in dataset_check_fields.fields
        del dataset_check_fields

        if flag_has_lhe_weights:
            logger.info(
                f"Attempting to extract sum of LHE weights from metadata of files to be merged from {source_path}"
            )
        else:
            logger.info(
                "Skipping the renormalization of LHE weights. Please check if you have stored the weights for LHE scale and LHE PDF variations. "
                "Don't worry if you are not evaluating LHE scale or PDF systematics for now"
            )

        sum_LHEPdf_beforesel, sum_LHEScale_beforesel = [], []

        for f in source_files:
            try:
                # Extract LHE weight field names from the first file
                if sum_LHEScale_fields is None:
                    sum_LHEScale_fields = [key for key in pq.read_schema(f).metadata.keys() if b'sum_weight_LHEScale' in key]
                    sum_LHEPdf_fields = [key for key in pq.read_schema(f).metadata.keys() if b'sum_weight_LHEPdf' in key]
                    # Extract indices from field names (e.g., b'sum_weight_LHEScale_0' -> 0)
                    index_sum_LHEScale = [int(key.decode().split('_')[-1]) for key in sum_LHEScale_fields]
                    index_sum_LHEPdf = [int(key.decode().split('_')[-1]) for key in sum_LHEPdf_fields]

                # Accumulate LHEScale weights
                for LHEScale_idx, LHEScale_field in zip(index_sum_LHEScale, sum_LHEScale_fields):
                    if LHEScale_idx >= len(sum_LHEScale_beforesel):
                        sum_LHEScale_beforesel.append(float(pq.read_schema(f).metadata[LHEScale_field]))
                    else:
                        sum_LHEScale_beforesel[LHEScale_idx] += float(pq.read_schema(f).metadata[LHEScale_field])

                # Accumulate LHEPdf weights
                for LHEPdf_idx, LHEPdf_field in zip(index_sum_LHEPdf, sum_LHEPdf_fields):
                    if LHEPdf_idx >= len(sum_LHEPdf_beforesel):
                        sum_LHEPdf_beforesel.append(float(pq.read_schema(f).metadata[LHEPdf_field]))
                    else:
                        sum_LHEPdf_beforesel[LHEPdf_idx] += float(pq.read_schema(f).metadata[LHEPdf_field])
            except:
                logger.info(
                    "Skipping the renormalization of LHE weights. Please check if you have stored sum of the LHE weights in the metadata with proper naming. "
                    "Example: sum_weight_LHEScale_0, sum_weight_LHEPdf_0."
                )
                flag_has_lhe_weights = False
                break

        do_lhe_norm.append(flag_has_lhe_weights)
        sum_LHEPdf_beforesel_arr.append(sum_LHEPdf_beforesel)
        sum_LHEScale_beforesel_arr.append(sum_LHEScale_beforesel)

    logger.info(
        "Successfully extracted sum of LHE weights."
    )
    if flag_has_lhe_weights:
        logger.info(
            "Successfully extracted sum of LHE scale and PDF weights"
        )

    return sum_LHEPdf_beforesel_arr, sum_LHEScale_beforesel_arr, do_lhe_norm


def Renormalize_LHE_Weights(dataset, target_path, cat, sum_LHEPdf_beforesel, sum_LHEScale_beforesel, logger):
    logger.info(
        f"Attempting to renormalize the LHE weights from {target_path}{cat}_merged.parquet"
    )
    logger.info(
        f"Successfully renormalized LHEScale weights from {target_path}{cat}_merged.parquet"
    )
    for col in range(len(sum_LHEScale_beforesel)):
        dataset['weight_LHEScale'][:, col] = dataset['weight_LHEScale'][:, col] / sum_LHEScale_beforesel[col]
    for col in range(len(sum_LHEPdf_beforesel)):
        dataset['weight_LHEPdf'][:, col] = dataset['weight_LHEPdf'][:, col] / sum_LHEPdf_beforesel[col]
    logger.info(
        f"Successfully renormalized LHEPdf weights from {target_path}{cat}_merged.parquet"
    )

    return dataset
