import awkward as ak
import numpy as np
import pandas as pd
import os
import torch
import onnxruntime

# ref: https://github.com/microsoft/onnxruntime/issues/8313
_default_session_options = onnxruntime.capi._pybind_state.get_default_session_options()


def get_default_session_options_new():
    _default_session_options.inter_op_num_threads = 1
    _default_session_options.intra_op_num_threads = 1
    return _default_session_options


onnxruntime.capi._pybind_state.get_default_session_options = (
    get_default_session_options_new
)


def get_model_path(model_type="Baseline"):
    if model_type == "Baseline":
        model_dict = {
            "2022preEE": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2022preEE/NN.onnx",
            ),
            "2022postEE": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2022postEE/NN.onnx",
            ),
            "2023preBPix": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2023preBPix/NN.onnx",
            ),
            "2023postBPix": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2023postBPix/NN.onnx",
            ),
        }
    elif model_type == "Minimal":
        model_dict = {
            "2022preEE": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2022preEE/Model_traced__Minimal.pt",
            ),
            "2022postEE": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2022postEE/Model_traced__Minimal.pt",
            ),
            "2023preBPix": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2023preBPix/Model_traced__Minimal.pt",
            ),
            "2023postBPix": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2023postBPix/Model_traced__Minimal.pt",
            ),
        }
    elif model_type == "nTrigEle":
        model_dict = {
            "2022preEE": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2022preEE/Model_traced_nTrigEle.pt",
            ),
            "2022postEE": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2022postEE/Model_traced_nTrigEle.pt",
            ),
            "2023preBPix": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2023preBPix/Model_traced_nTrigEle.pt",
            ),
            "2023postBPix": os.path.join(
                os.path.dirname(__file__),
                "../tools/lowmass_dykiller/2023postBPix/Model_traced_nTrigEle.pt",
            ),
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_dict


def get_variable_list(model_type="Baseline"):
    # * varible list could have:
    # * - direct variable name, e.g., sigma_wv
    # * - varible in subfield, e.g., photon_lead.eta
    if model_type == "Baseline":
        variable_list = [
            "ptom",
            "pho_lead.s4",
            "pho_lead.sieip",
            "pho_lead.sipip",
            "pho_lead.sieie",
            "pho_lead.phi",
            "pho_lead.phiWidth",
            "pho_lead.ptom",
            "pho_lead.r9",
            "pho_lead.mvaID",
            "pho_sublead.s4",
            "pho_sublead.sieip",
            "pho_sublead.sipip",
            "pho_sublead.sieie",
            "pho_sublead.phi",
            "pho_sublead.phiWidth",
            "pho_sublead.ptom",
            "pho_sublead.r9",
            "pho_sublead.mvaID",
            "PV_log_score",
        ]
    elif model_type == "Minimal":
        variable_list = [
            "ptom",
            "pho_lead.s4",
            "pho_lead.ptom",
            "pho_lead.r9",
            "pho_lead.mvaID",
            "pho_sublead.s4",
            "pho_sublead.ptom",
            "pho_sublead.r9",
            "pho_sublead.mvaID",
            "PV_log_score",
        ]
    elif model_type == "nTrigEle":
        variable_list = [
            "ptom",
            "pho_lead.s4",
            "pho_lead.ptom",
            "pho_lead.r9",
            "pho_lead.mvaID",
            "pho_lead.pfRelIso03_chg_quadratic",
            "pho_sublead.s4",
            "pho_sublead.ptom",
            "pho_sublead.r9",
            "pho_sublead.mvaID",
            "pho_sublead.pfRelIso03_chg_quadratic",
            "nTrigEle",
            "PV_log_score",
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return variable_list


def eval_dykiller_for_lowmass(diphotons, year="2022postEE"):
    # adding log PV_score
    diphotons["PV_log_score"] = np.log(diphotons["PV_score"])

    # no need to load the models for zero length
    if len(diphotons) == 0:
        for model_type in ["Baseline", "Minimal", "nTrigEle"]:
            diphotons[f"dykiller_{model_type}"] = np.zeros(len(diphotons), dtype=np.float32)
        return diphotons

    for model_type in ["Baseline", "Minimal", "nTrigEle"]:
        model_dict = get_model_path(model_type)

        # model input variables
        variable_list = get_variable_list(model_type)
        dict_inputs = {
            var: ak.to_numpy(diphotons[tuple(var.split(".")) if "." in var else var])
            for var in variable_list
        }

        df_inputs = pd.DataFrame(dict_inputs)

        # ! Input df_inputs should not contain infinity or a value too large for dtype('float32')
        df_inputs = df_inputs.clip(
            np.finfo(np.float32).min + 1, np.finfo(np.float32).max - 1
        )

        if model_type == "Baseline":
            # ONNX model evaluation
            ort_session = onnxruntime.InferenceSession(f"{model_dict[year]}")
            input_name = ort_session.get_inputs()[0].name

            # ONNX model inference
            predictions = ort_session.run(
                None, {input_name: df_inputs.to_numpy(dtype=np.float32)}
            )
            # sigmoid function
            preds = 1 / (1 + np.exp(-predictions[0]))

            # add dykiller score
            diphotons[f"dykiller_{model_type}"] = preds.reshape(-1)
        else:
            # Load traced PyTorch model
            model = torch.jit.load(f"{model_dict[year]}")
            model.eval()

            # Disable gradient computation for efficiency
            with torch.no_grad():
                # Your evaluation code here
                out_eval = model(torch.FloatTensor(df_inputs.to_numpy())).flatten()
                out_score = 1 / (1 + np.exp(-out_eval))
            diphotons[f"dykiller_{model_type}"] = out_score.numpy()

    return diphotons
