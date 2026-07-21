from coffea.jetmet_tools.CorrectedMETFactory import corrected_polar_met
import awkward as ak


def apply_type1_met_correction(
    met: ak.Array,
    objects: tuple,
    raw_pt_name: str = "pt_nano"
) -> ak.Array:
    """
    Apply a Type-1 MET correction by propagating the pt-shifts
    from each object in `objects` into the MET.

    Parameters
    ----------
    met : awkward.Array
        Input MET record with fields `.pt` and `.phi`.
    objects : tuple of awkward.Array
        Any number of collections (jets, photons, electrons, muons, ...),
        each with fields:
            - .pt
            - .phi
            - .<raw_pt_name>   (the original, uncorrected pt)
    raw_pt_name : str, optional
        The field name on each object that stores its original (nanoAOD)
        pt before corrections, by default "pt_nano".

    Returns
    -------
    awkward.Array
        A new record with fields `.pt` and `.phi` containing the cumulatively corrected MET.
    """
    # Start from the input MET
    corr = met
    # Loop over each object collection in turn
    for obj in objects:
        corr = corrected_polar_met(
            met_pt=corr.pt,
            met_phi=corr.phi,
            jet_pt=obj.pt,
            jet_phi=obj.phi,
            jet_pt_orig=getattr(obj, raw_pt_name),
        )
    return corr


def MET_syst_Unclustered(ptphi, *, events=None, year=None):
    """
    ptphi: bundle of fields listed in `what` (here pt & phi), already flattened by Coffea.
           Shape = (M,) where M = len(ak.flatten(collection_you_called_add_systematic_on))
    Return: record-of-arrays with fields 'pt' and 'phi', each shaped (M, 2)
            [:,0] = Up, [:,1] = Down
    """
    # ptphi.pt and ptphi.phi are flat arrays of length N
    pt_var = ak.concatenate(
        [events.PuppiMET.ptUnclusteredUp[:,None], events.PuppiMET.ptUnclusteredDown[:,None]], axis=1
    )
    phi_var = ak.concatenate(
        [events.PuppiMET.phiUnclusteredUp[:,None], events.PuppiMET.phiUnclusteredDown[:,None]], axis=1
    )
    return ak.zip({"pt": pt_var, "phi": phi_var}, depth_limit=1)
