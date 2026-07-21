import awkward as ak
import numba
import numpy as np
import vector

vector.register_awkward()


def choose_jet(jets_variable, n, fill_value):
    """
    this helper function is used to create flat jets from a jagged collection,
    parameters:
    * jet_variable: (ak array) selected variable from the jet collection
    * n: (int) nth jet to be selected
    * fill_value: (float) value with wich to fill the padded none.
    """
    leading_jets_variable = jets_variable[
        ak.local_index(jets_variable) == n
    ]
    leading_jets_variable = ak.pad_none(
        leading_jets_variable, 1
    )
    leading_jets_variable = ak.flatten(
        ak.fill_none(leading_jets_variable, fill_value)
    )
    return leading_jets_variable


def rapidity_from_pt_eta_mass(pt, eta, mass, fill_value=-999.0):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        pz = pt * np.sinh(eta)
        pz = ak.where(eta == fill_value, fill_value, pz)
        energy = np.sqrt((pt**2 * np.cosh(eta)**2) + mass**2)
        rapidity = 0.5 * np.log((energy + pz) / (energy - pz))

    rapidity = ak.fill_none(rapidity, fill_value)
    rapidity = ak.where(np.isnan(rapidity), fill_value, rapidity)
    rapidity = ak.where(np.isinf(rapidity), fill_value, rapidity)
    rapidity = ak.where(eta == fill_value, fill_value, rapidity)
    return rapidity


def add_pnet_prob(
    self,
    jets: ak.highlevel.Array
):
    """
    this helper function is used to add to the jets from the probability of PNet
    calculated starting from the standard scores contained in the JetMET nAODs
    """

    jet_pn_b = jets.particleNetAK4_B

    jet_pn_c = jets.particleNetAK4_B * jets.particleNetAK4_CvsB / (ak.ones_like(jets.particleNetAK4_B) - jets.particleNetAK4_CvsB)
    jet_pn_c = ak.where(
        (jets.particleNetAK4_CvsB >= 0) & (jets.particleNetAK4_CvsB < 1),
        jet_pn_c,
        -1
    )

    # Use ak.where to constrain the values within [0, 1]
    pn_uds_base = ak.ones_like(jet_pn_b) - jet_pn_b - jet_pn_c
    pn_uds_clipped = ak.where(pn_uds_base < 0, 0, ak.where(pn_uds_base > 1, 1, pn_uds_base))
    jet_pn_uds = pn_uds_clipped * jets.particleNetAK4_QvsG
    jet_pn_uds = ak.where(
        (jets.particleNetAK4_QvsG >= 0) & (jets.particleNetAK4_QvsG < 1),
        jet_pn_uds,
        -1
    )

    jet_pn_g_base = ak.ones_like(jet_pn_b) - jet_pn_b - jet_pn_c - jet_pn_uds
    jet_pn_g = ak.where(jet_pn_g_base < 0, 0, ak.where(jet_pn_g_base > 1, 1, jet_pn_g_base))
    jet_pn_g = ak.where(
        (jets.particleNetAK4_QvsG >= 0) & (jets.particleNetAK4_QvsG < 1),
        jet_pn_g,
        -1
    )

    jet_pn_b_plus_c = jet_pn_b + jet_pn_c
    jet_pn_b_vs_c = jet_pn_b / jet_pn_b_plus_c

    jets["pn_b"] = jet_pn_b
    jets["pn_c"] = jet_pn_c
    jets["pn_uds"] = jet_pn_uds
    jets["pn_g"] = jet_pn_g
    jets["pn_b_plus_c"] = jet_pn_b_plus_c
    jets["pn_b_vs_c"] = jet_pn_b_vs_c

    return jets


def evaluate_ctag_wp(ctag_wps, nth_jet_pn_b_plus_c, nth_jet_pn_b_vs_c):
    """ParticleNetAK4 -- exclusive b- and c-tagging categories
    5x: b-tagged; 5x: c-tagged; 0: light

    Returns
    -------
    wp: ak.Array
        working point category for each jet (-1 for invalid scores)
    """
    wp = ak.ones_like(nth_jet_pn_b_plus_c) * -1  # Initialize with -1 for invalid scores
    for wp_cfg in ctag_wps:
        wp_ids = ak.ones_like(nth_jet_pn_b_plus_c) * wp_cfg[0]
        wp = ak.where(
            (wp_cfg[1][0] < nth_jet_pn_b_plus_c) & (nth_jet_pn_b_plus_c <= wp_cfg[1][1]) & (wp_cfg[2][0] < nth_jet_pn_b_vs_c) & (nth_jet_pn_b_vs_c <= wp_cfg[2][1]),
            wp_ids,
            wp
        )

    return wp


def evaluate_btag_sf_eff_multiwp(variation, wp, flat_jet_hFlav, flat_jet_eta, flat_jet_pt, jet_counts, sf_evaluator, eff_evaluator, dataset_name):
    """Evaluate the b-tagging scale factor and efficiency for each jet in multi-WP case"""
    sf = ak.unflatten(
        sf_evaluator.evaluate(
            variation,
            wp,
            flat_jet_hFlav,
            flat_jet_eta,
            flat_jet_pt,
        ),
        jet_counts
    )
    eff = ak.unflatten(
        eff_evaluator.evaluate(
            dataset_name,
            wp,
            flat_jet_hFlav,
            flat_jet_pt
        ),
        jet_counts
    )
    return sf, eff


def compute_btag_multiwp(jets_by_region, _sf_pass, _sf_fail, _btagEff_pass, _btagEff_fail, is_central):
    """Compute the b-tagging weight for light/heavy jets in the multiwp case"""
    w_btag_partial = None

    for key in jets_by_region:
        wp_pass, wp_fail = key
        if wp_pass is not None and wp_fail is not None:
            # jets in (J, J+1): use multi-WP formula
            numerator = _sf_pass[key] * _btagEff_pass[key] - _sf_fail[key] * _btagEff_fail[key]
            denominator = _btagEff_pass[key] - _btagEff_fail[key]
            term = numerator / denominator
            if is_central:
                # Avoid the unphysical case: https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/
                term = ak.where((numerator < 0), 1.0, term)
        elif wp_pass is not None:
            # region e.g. (T, None)
            term = _sf_pass[key]
        else:
            # region e.g. (None, L)
            numerator = 1 - _sf_fail[key] * _btagEff_fail[key]
            denominator = 1 - _btagEff_fail[key]
            term = numerator / denominator

        term_prod = ak.prod(term, axis=1)

        if w_btag_partial is None:
            w_btag_partial = term_prod
        else:
            w_btag_partial = w_btag_partial * term_prod

    return w_btag_partial


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64),
    ]
)
def delta_phi(a, b):
    """Compute difference in angle given two angles a and b

    Returns a value within [-pi, pi)
    """
    return (a - b + np.pi) % (2 * np.pi) - np.pi


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32, numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64, numba.float64, numba.float64),
    ]
)
def delta_r(eta1, phi1, eta2, phi2):
    r"""Distance in (eta,phi) plane given two pairs of (eta,phi)

    :math:`\sqrt{\Delta\eta^2 + \Delta\phi^2}`
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.hypot(deta, dphi)


def delta_r_with_ScEta(a, b):
    """Distance in (eta,phi) plane between two objects using `ScEta` insetad of `eta`"""
    return delta_r(a.ScEta, a.phi, b.eta, b.phi)


def trigger_match(
        offline_objects, trigobjs, pdgid, pt, filterbit, metric=lambda a, b: a.delta_r(b), dr=0.1
):
    """
    Matches offline objects  with online trigger objects using dR < dr criterion
    The filterbit corresponds to the trigger we want our offline objects to have fired
    """
    pass_pt = trigobjs.pt > pt
    pass_id = abs(trigobjs.id) == pdgid
    pass_filterbit = (trigobjs.filterBits & (0x1 << filterbit)) != 0
    trigger_cands = trigobjs[pass_pt & pass_id & pass_filterbit]
    delta_r = offline_objects.metric_table(trigger_cands, metric=metric)
    pass_delta_r = delta_r < dr
    n_of_trigger_matches = ak.sum(pass_delta_r, axis=2)
    trig_matched_locs = n_of_trigger_matches >= 1

    return trig_matched_locs


def DPhiV1V2(vec1, vec2):
    """
    Compute the generalized azimuthal angle difference Δφ between two objects
    based on their transverse directions and spatial separation.

    This observable captures the relative azimuthal orientation of two objects
    and can be used to study symmetries and angular correlations in a variety
    of systems.

    The definition used is:

        Δφ = sign_factor * arccos(vt1_hat ⋅ vt2_hat)

    where:

        sign_factor = sign((vt1_hat × vt2_hat) ⋅ z_hat) * sign((v1 - v2) ⋅ z_hat)

    Parameters:
        v1 (np.ndarray): 3D vector representing the position or momentum of the first object.
        v2 (np.ndarray): 3D vector representing the position or momentum of the second object.
        vt1_hat (np.ndarray): Unit vector representing the transverse component of the first object.
        vt2_hat (np.ndarray): Unit vector representing the transverse component of the second object.
        z_hat (np.ndarray): Unit vector defining the reference z-axis direction.

    Returns:
        float: The permutation invariant azimuthal angle difference Δφ in radians.

    Notes:
        - This definition is frame-independent as long as the transverse plane and z-axis are consistently defined.
        - Useful in contexts involving angular distributions, symmetry studies, and CP-violation-sensitive observables.
    """
    # Extract 3D direction vectors
    j1dir = ak.zip({"x": vec1.px, "y": vec1.py, "z": vec1.pz}, with_name="Vector3D")
    j2dir = ak.zip({"x": vec2.px, "y": vec2.py, "z": vec2.pz}, with_name="Vector3D")

    # Project to transverse plane (z = 0)
    jt1 = ak.zip({"x": vec1.px, "y": vec1.py, "z": ak.zeros_like(vec1.px)}, with_name="Vector3D")
    jt2 = ak.zip({"x": vec2.px, "y": vec2.py, "z": ak.zeros_like(vec2.px)}, with_name="Vector3D")

    # Normalize transverse vectors
    jt1_unit = jt1.unit()
    jt2_unit = jt2.unit()

    # z-axis unit vector
    z = ak.zip({
        "x": ak.zeros_like(vec1.px),
        "y": ak.zeros_like(vec1.px),
        "z": ak.ones_like(vec1.px),
    }, with_name="Vector3D")

    # Sign from cross and difference
    cross_sign = ak.where(jt1_unit.cross(jt2_unit).dot(z) > 0, 1.0, ak.where(jt1_unit.cross(jt2_unit).dot(z) <= 0, -1.0, 0.0))

    diff_sign = ak.where((j1dir - j2dir).dot(z) > 0, 1.0, ak.where((j1dir - j2dir).dot(z) <= 0, -1.0, 0.0))

    # Dot product
    dot = jt1_unit.dot(jt2_unit)

    # Valid range for acos is [-1, 1]
    valid = (dot >= -1.0) & (dot <= 1.0)

    # Compute acos only for valid entries
    with np.errstate(over='ignore', invalid='ignore'):
        dphi = ak.where(valid, np.arccos(dot) * diff_sign * cross_sign, -999.0)

    return dphi
