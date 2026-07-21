import numpy as np
import awkward as ak
import uproot
from packaging.version import parse as parse_version
_use_mktree = parse_version(uproot.__version__) >= parse_version("5.7.0")


def extract_tuples(input_string):
    tuples = []
    # Remove leading and trailing parentheses and split by comma
    tuple_strings = input_string.strip("()").split(";")
    for tuple_str in tuple_strings:
        # Remove leading and trailing whitespace and parentheses
        tuple_elements = tuple_str.strip("()").split(",")
        # Strip each element and append to the list of tuples
        tuples.append(tuple(map(str.strip, tuple_elements)))
    return tuples

def extract_filter(dataset, additionalConditionTuple):
    variable, operator, value = additionalConditionTuple

    if operator == ">":
        return dataset[variable] > float(value)
    elif operator == ">=":
        return dataset[variable] >= float(value)
    elif operator == "<":
        return dataset[variable] < float(value)
    elif operator == "<=":
        return dataset[variable] <= float(value)
    elif operator == "==":
        if ("True" in value) or ("False" in value):
            value = bool(value)
            return dataset[variable] == value
        else:
            return dataset[variable] == float(value)

def filter_and_set_diff_variable(dataset, ranges_dict, selectionVariableName="GenPTH", diffVariableName="diffVariable_GenPTH"):
    # Initialize diff variable in the awkward array
    dataset[diffVariableName] = 0

    # Specify variables which need the absolute value for the selection (eg. rapidity)
    absolute_value_vars = ["GenYH"]

    for range_min, range_max, fiducialTag, additionalConditions in ranges_dict.keys():
        diffId = ranges_dict[(range_min, range_max, fiducialTag, additionalConditions)]

        if fiducialTag == "in":
            condition = (dataset["fiducialGeometricFlag"] == True)
        else:
            condition = (dataset["fiducialGeometricFlag"] == False)

        if selectionVariableName in absolute_value_vars:
            condition = condition & (np.abs(dataset[selectionVariableName]) >= range_min) & (np.abs(dataset[selectionVariableName]) < range_max)
        else:
            condition = condition & (dataset[selectionVariableName] >= range_min) & (dataset[selectionVariableName] < range_max)

        if additionalConditions != "":
            tuple_list = extract_tuples(additionalConditions)
            for additionalCondition in tuple_list:
                condition = condition & extract_filter(dataset, additionalCondition)
        dataset[diffVariableName] = ak.where(condition, diffId, dataset[diffVariableName])

    return dataset

def _resolve_numpy_dtype(awkward_type):
    """
    Traverse an Awkward Type and return the underlying NumPy dtype, if any.
    """
    if isinstance(awkward_type, ak.types.NumpyType):
        return np.dtype(awkward_type.primitive)
    if isinstance(awkward_type, (ak.types.OptionType, ak.types.ListType, ak.types.RegularType)):
        return _resolve_numpy_dtype(awkward_type.content)
    if isinstance(awkward_type, ak.types.UnknownType):
        return None
    return None

def _default_fill_value(dtype):
    """
    Provide a sensible fill value to replace missing data according to dtype.
    Floats -> NaN, ints -> 0, bools -> False, fallback -> 0.
    """
    if dtype is None:
        return np.nan
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.bool_):
        return False
    if np.issubdtype(dtype, np.integer):
        return 0
    return 0

def prepare_branch_array(array):
    """
    Sanitize branch arrays before writing to ROOT:
      * fill missing values so we don't propagate Awkward option types
      * convert 1D arrays to NumPy for better uproot compatibility
      * keep jagged arrays as Awkward
    """
    if isinstance(array, ak.Array):
        dtype = _resolve_numpy_dtype(array.type)
        if dtype is None and len(array) == 0:
            # empty branch with unknown dtype; default to float64
            return np.empty(0, dtype=np.float64)

        fill_value = _default_fill_value(dtype)

        # Replace None entries to avoid optional types in uproot
        try:
            array = ak.fill_none(array, fill_value)
        except TypeError:
            # if fill_value is incompatible (e.g. ints with NaN), cast to float and retry
            array = ak.values_astype(array, np.float64)
            array = ak.fill_none(array, np.nan)

        # Convert simple 1D arrays to NumPy, keep jagged ones as Awkward
        try:
            return ak.to_numpy(array, allow_missing=False)
        except (TypeError, ValueError):
            return array

    return np.asarray(array)

def split_awkward_arrays_by_length(d, logger, target_length=5000):
    """
    Split the branch dictionary into chunks that all share the same event range.
    This assumes every branch has the same length (number of events). If a
    mismatch is detected we raise, since writing misaligned data would corrupt
    the ROOT output.
    """
    if not d:
        return [d]

    prepared = {key: prepare_branch_array(array) for key, array in d.items()}

    ref_key, ref_array = next(iter(prepared.items()))
    total_events = len(ref_array)

    if total_events == 0:
        logger.debug("All branches are empty; skipping split and returning one chunk.")
        return [prepared]

    for key, array in prepared.items():
        current_len = len(array)
        if current_len != total_events:
            msg = (
                f"Branch length mismatch detected while chunking ROOT output: "
                f"'{key}' has {current_len} entries, expected {total_events} (matching '{ref_key}')."
            )
            logger.error(msg)
            raise ValueError(msg)

    num_chunks = (total_events + target_length - 1) // target_length  # ceiling division

    split_dicts = []
    for i in range(num_chunks):
        start = i * target_length
        end = min((i + 1) * target_length, total_events)
        current_split = {}
        for key, array in prepared.items():
            slice_obj = slice(start, end)
            current_split[key] = array[slice_obj]
        split_dicts.append(current_split)

    return split_dicts

def ensure_nweight_LHEScale(d):
    """
    Ensure the branch `nweight_LHEScale`, when present, is stored with dtype int32.
    """
    if "nweight_LHEScale" not in d:
        return d

    branch = d["nweight_LHEScale"]
    if isinstance(branch, ak.Array):
        d["nweight_LHEScale"] = ak.values_astype(branch, np.int32)
    else:
        d["nweight_LHEScale"] = np.asarray(branch, dtype=np.int32)
    return d

def make_tree(file, treename, branch_dict):
    """
    Create a ROOT TTree with the given name and branches from the provided dictionary.
    The branch_dict should have the branch names as keys
    and the corresponding arrays (as NumPy or Awkward) as values.
    """
    if _use_mktree:
        file.mkdir(treename.rsplit("/", 1)[0])
        file.mktree(treename, branch_dict)
    else:
        file[treename] = branch_dict
