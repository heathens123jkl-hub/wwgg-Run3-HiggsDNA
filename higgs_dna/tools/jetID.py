import awkward as ak
import correctionlib
import os
import logging

logger = logging.getLogger(__name__)


def add_jetId(jets, nano_version, year, flattenUnflatten=False):
    """
    Add (or recompute) jet ID to the jets object based on the NanoAOD version.
    """
    abs_eta = abs(jets.eta)

    # Return the existing jetId for NanoAOD versions below 12
    if nano_version < 12:
        return jets.jetId

    # For NanoAOD version 12 and above, we recompute the jet ID criteria
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
    else:
        if nano_version == 12:
            # Default tight
            passJetIdTight = ak.where(
                abs_eta <= 2.7,
                (jets.jetId & (1 << 1)) > 0,  # Tight criteria for abs_eta <= 2.7
                ak.where(
                    (abs_eta > 2.7) & (abs_eta <= 3.0),
                    ((jets.jetId & (1 << 1)) > 0) & (jets.neHEF < 0.99),  # Tight criteria for 2.7 < abs_eta <= 3.0
                    ((jets.jetId & (1 << 1)) > 0) & (jets.neEmEF < 0.4)  # Tight criteria for 3.0 < abs_eta
                )
            )

            # Default tight lepton veto
            passJetIdTightLepVeto = ak.where(
                abs_eta <= 2.7,
                passJetIdTight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8),  # add lepton veto for abs_eta <= 2.7
                passJetIdTight  # No lepton veto for 2.7 < abs_eta
            )

            return (passJetIdTight * (1 << 1)) | (passJetIdTightLepVeto * (1 << 2))
        elif nano_version == 15 and year in ["2016preVFP", "2016postVFP"]:
            # Run 2 NanoAOD v15 2016 recipe
            passJetIdTight = ak.where(
                abs_eta <= 2.4,
                (jets.neHEF < 0.9)
                & (jets.neEmEF < 0.9)
                & (jets.chMultiplicity + jets.neMultiplicity > 1)
                & (jets.chHEF > 0.0)
                & (jets.chMultiplicity > 0),
                ak.where(
                    (abs_eta > 2.4) & (abs_eta <= 2.7),
                    (jets.neHEF < 0.98)
                    & (jets.neEmEF < 0.99),
                    ak.where(
                        (abs_eta > 2.7) & (abs_eta <= 3.0),
                        jets.neMultiplicity >= 1,
                        (jets.neMultiplicity > 2)
                        & (jets.neEmEF < 0.9)
                    )
                )
            )
            passJetIdTightLepVeto = ak.where(
                abs_eta <= 2.4,
                passJetIdTight
                & (jets.muEF < 0.8)
                & (jets.chEmEF < 0.8),
                passJetIdTight
            )
            return (passJetIdTight * (1 << 1)) | (passJetIdTightLepVeto * (1 << 2))
        elif nano_version == 15 and year in ["2017", "2018"]:
            # Run 2 NanoAOD v15 2017/2018
            passJetIdTight = ak.where(
                abs_eta <= 2.6,
                (jets.neHEF < 0.9)
                & (jets.neEmEF < 0.9)
                & (jets.chMultiplicity + jets.neMultiplicity > 1)
                & (jets.chHEF > 0.0)
                & (jets.chMultiplicity > 0),
                ak.where(
                    (abs_eta > 2.6) & (abs_eta <= 2.7),
                    (jets.neHEF < 0.90)
                    & (jets.neEmEF < 0.99),
                    ak.where(
                        (abs_eta > 2.7) & (abs_eta <= 3.0),
                        jets.neHEF < 0.9999,
                        (jets.neMultiplicity > 2)
                        & (jets.neEmEF < 0.9)
                    )
                )
            )
            passJetIdTightLepVeto = ak.where(
                abs_eta <= 2.7,
                passJetIdTight
                & (jets.muEF < 0.8)
                & (jets.chEmEF < 0.8),
                passJetIdTight
            )
            return (passJetIdTight * (1 << 1)) | (passJetIdTightLepVeto * (1 << 2))
        else:
            # Example code: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jetidExample.py?ref_type=heads
            # Load CorrectionSet
            fallback_years = ["2025"]
            if year in fallback_years:
                logger.warning(f"There is no dedicated {year} jetID. As the {year} PUPPI tune is the same as the 2024 one, the 2024 jetID used! ")
                year = "2024"

            jerc_json = {
                "2022preEE": os.path.join(
                    os.path.dirname(__file__),
                    "../systematics/JSONs/POG/JME/2022_Summer22/jetid.json.gz",
                ),
                "2022postEE": os.path.join(
                    os.path.dirname(__file__),
                    "../systematics/JSONs/POG/JME/2022_Summer22EE/jetid.json.gz",
                ),
                "2023preBPix": os.path.join(
                    os.path.dirname(__file__),
                    "../systematics/JSONs/POG/JME/2023_Summer23/jetid.json.gz",
                ),
                "2023postBPix": os.path.join(
                    os.path.dirname(__file__),
                    "../systematics/JSONs/POG/JME/2023_Summer23BPix/jetid.json.gz",
                ),
                "2024": os.path.join(
                    os.path.dirname(__file__),
                    "../systematics/JSONs/POG/JME/2024_Summer24/jetid.json.gz",
                ),
            }

            cset = correctionlib.CorrectionSet.from_file(jerc_json[year])

            if flattenUnflatten:
                counts = ak.num(jets)
                jets = ak.flatten(jets, axis=1)

            eval_dict = {
                "eta": jets.eta,
                "chHEF": jets.chHEF,
                "neHEF": jets.neHEF,
                "chEmEF": jets.chEmEF,
                "neEmEF": jets.neEmEF,
                "muEF": jets.muEF,
                "chMultiplicity": jets.chMultiplicity,
                "neMultiplicity": jets.neMultiplicity,
                "multiplicity": jets.chMultiplicity + jets.neMultiplicity
            }

            ## Default tight for NanoAOD version 13 and above
            idTight = cset["AK4PUPPI_Tight"]
            inputsTight = [eval_dict[input.name] for input in idTight.inputs]
            idTight_value = idTight.evaluate(*inputsTight) * 2  # equivalent to bit2

            # Default tight lepton veto
            idTightLepVeto = cset["AK4PUPPI_TightLeptonVeto"]
            inputsTightLepVeto = [eval_dict[input.name] for input in idTightLepVeto.inputs]
            idTightLepVeto_value = idTightLepVeto.evaluate(*inputsTightLepVeto) * 4  # equivalent to bit3

            # Default jet ID
            id_value = idTight_value + idTightLepVeto_value

            if flattenUnflatten:
                return ak.unflatten(id_value, counts)
            else:
                return id_value
