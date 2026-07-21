import awkward as ak
import numpy as np
from higgs_dna.selections.HHbbgg_selections import DeltaR
from higgs_dna.selections.object_selections import delta_r_mask
import logging

logger = logging.getLogger(__name__)

# Default particle-type definitions for associated decay classification
DEFAULT_ASSOCIATED_DECAY_PARTICLE_MAP = {
    "n_lep": (11, 13, 15),
    "n_nu": (12, 14, 16),
    "n_q": tuple(range(1, 9)),
}


def get_fiducial_flag(events: ak.Array, flavour: str = "Geometric") -> ak.Array:
    """
    Calculate the fiducial flag for events based on photon kinematics and geometric criteria at particle level.

    The function processes the events, identifying those that meet
    specific criteria based on the properties of the leading and subleading photons.
    The fiducial flag is determined based on transverse momentum (pt), mass, and pseudorapidity (eta)
    of the photon pairs, applying either 'Geometric' or 'Classical' selection criteria.

    Parameters:
    - events (ak.Array): An Awkward Array containing event data with GenIsolatedPhoton fields.
    - flavour (str, optional): The selection criterion to apply. Defaults to "Geometric".
      Can be "Geometric" for geometric mean based selection (https://arxiv.org/abs/2106.08329) or "Classical" for classical CMS pt/mass scaled cuts.

    Returns:
    - ak.Array: An Awkward Array of boolean flags, where True indicates an event meets the fiducial
      selection criteria.

    Note:
    - The function pads GenIsolatedPhoton fields to ensure at least two photons are present per event,
      filling missing values with None.
    - If the GenPart_iso branch is not included in the NanoAOD, the GenIsolatedPhotons collection is used
    """
    if 'iso' in events.GenPart.fields:
        sel_pho = (events.GenPart.pdgId == 22) & (events.GenPart.status == 1) & (events.GenPart.iso * events.GenPart.pt < 10)
        photons = events.GenPart[sel_pho]
        photons = photons[ak.argsort(photons.pt, ascending=False)]
        GenIsolatedPhotons = ak.pad_none(photons, 2)
    else:
        # Extract and pad the gen isolated photons
        GenIsolatedPhotons = events.GenIsolatedPhoton
        GenIsolatedPhotons = ak.pad_none(GenIsolatedPhotons, 2)

    # Separate leading and subleading photons
    lead_pho = GenIsolatedPhotons[:, 0]
    sublead_pho = GenIsolatedPhotons[:, 1]

    # Calculate diphoton system four vector
    diphoton = lead_pho + sublead_pho

    # Apply selection criteria based on the specified flavour
    if flavour == 'Geometric':
        # Geometric mean of pt criterion
        lead_mask = np.sqrt(lead_pho.pt * sublead_pho.pt) / diphoton.mass > 1 / 3
    elif flavour == 'Classical':
        # Classical pt/mass ratio criterion
        lead_mask = lead_pho.pt / diphoton.mass > 1 / 3

    # Subleading photon criterion always the same
    sublead_mask = sublead_pho.pt / diphoton.mass > 1 / 4

    # Pseudorapidity criteria for leading and subleading photons
    # Within tracker acceptance and remove the gap region
    # Note: Based on classical eta, not SC eta here
    lead_eta_mask = (np.abs(lead_pho.eta) < 1.4442) | ((np.abs(lead_pho.eta) < 2.5) & (np.abs(lead_pho.eta) > 1.566))
    sublead_eta_mask = (np.abs(sublead_pho.eta) < 1.4442) | ((np.abs(sublead_pho.eta) < 2.5) & (np.abs(sublead_pho.eta) > 1.566))

    # Combine all selection masks to form the fiducial flag
    fiducial_flag = lead_mask & sublead_mask & lead_eta_mask & sublead_eta_mask
    # Fill None values with False
    # Note: These values result from the padding
    # Only occurs for events that did not have two GenIsolatedPhoton origin
    fiducial_flag = ak.fill_none(fiducial_flag, False)

    return fiducial_flag


def get_genJets(
    events: ak.Array,
    pt_cut,
    eta_cut,
    jet_pho_min_dr=0.4,
    jet_ele_min_dr=0.4,
    jet_muo_min_dr=0.4,
    electron_pt_threshold=15.0,
    electron_max_eta=2.5,
    muon_pt_threshold=10.0,
    muon_max_eta=2.4,
) -> ak.Array:
    # We decide to clean based on dR criteria and not use partonFlavour as this is easier to reproduce
    # The commented option below is also interesting, removing jets that have not been matched to a coloured parton...
    # GenJets = GenJets[GenJets.partonFlavour != 0]
    GenJets = events.GenJet

    if 'iso' in events.GenPart.fields:
        # Note: iso is a relative quantity
        sel_pho = (events.GenPart.pdgId == 22) & (events.GenPart.status == 1) & (events.GenPart.iso * events.GenPart.pt < 10)
        photons = events.GenPart[sel_pho]
        photons = photons[ak.argsort(photons.pt, ascending=False)]
        GenIsolatedPhotons = ak.pad_none(photons, 2)
    else:
        # Extract and pad the gen isolated photons
        GenIsolatedPhotons = events.GenIsolatedPhoton
        GenIsolatedPhotons = ak.pad_none(GenIsolatedPhotons, 2)

    # Separate leading and subleading photons
    lead_pho = GenIsolatedPhotons[:, 0]
    sublead_pho = GenIsolatedPhotons[:, 1]
    diphotons = lead_pho + sublead_pho

    if (ak.num(diphotons.pt, axis=0) > 0):
        lead = ak.zip(
            {
                "pt": lead_pho.pt,
                "eta": lead_pho.eta,
                "phi": lead_pho.phi,
                "mass": lead_pho.mass,
            }
        )
        lead = ak.with_name(lead, "PtEtaPhiMCandidate")
        sublead = ak.zip(
            {
                "pt": sublead_pho.pt,
                "eta": sublead_pho.eta,
                "phi": sublead_pho.phi,
                "mass": sublead_pho.mass,
            }
        )
        sublead = ak.with_name(sublead, "PtEtaPhiMCandidate")
        dr_pho_lead_cut = delta_r_mask(GenJets, lead, jet_pho_min_dr)
        dr_pho_sublead_cut = delta_r_mask(GenJets, sublead, jet_pho_min_dr)
    else:
        dr_pho_lead_cut = GenJets.pt > -1
        dr_pho_sublead_cut = GenJets.pt > -1

    # Lepton selection for overlap removal
    GenLeptons = events.GenPart[(abs(events.GenPart.pdgId) == 11) | (abs(events.GenPart.pdgId) == 13) & (events.GenPart.status == 1)]
    # # 11: Electron, 13: Muon
    if 'iso' in events.GenPart.fields:
        SelGenElectrons = GenLeptons[
            (abs(GenLeptons.pdgId) == 11)
            & (GenLeptons.pt > electron_pt_threshold)
            & (abs(GenLeptons.eta) < electron_max_eta)
            & (GenLeptons.iso < 0.2)
        ]
        SelGenMuons = GenLeptons[
            (abs(GenLeptons.pdgId) == 13)
            & (GenLeptons.pt > muon_pt_threshold)
            & (abs(GenLeptons.eta) < muon_max_eta)
            & (GenLeptons.iso < 0.2)
        ]
        dr_electrons_mask = delta_r_mask(GenJets, SelGenElectrons, jet_ele_min_dr)
        dr_muons_mask = delta_r_mask(GenJets, SelGenMuons, jet_muo_min_dr)
    else:
        logger.info("Careful: You are running over a sample that is nanoAOD v13 or older where the genPart collection does not contain the iso field")
        logger.info("Overlap removal for counting GenJets wrt to leptons will not be performed.")
        dr_electrons_mask = GenJets.pt > -1
        dr_muons_mask = GenJets.pt > -1

    GenJets = GenJets[(dr_pho_lead_cut) & (dr_pho_sublead_cut) & (dr_electrons_mask) & (dr_muons_mask)]

    # This is targeted primarly at photons from Higgs decay but also prompt electrons and muons in VH, TTH
    GenJets = GenJets[GenJets.pt > pt_cut]
    GenJets = GenJets[np.abs(GenJets.eta) < eta_cut]

    return GenJets


def get_higgs_gen_attributes(events: ak.Array) -> ak.Array:
    """
    Calculate the Higgs pt and y based on photon kinematics at particle level.

    Note:
    - The function pads GenIsolatedPhoton fields to ensure at least two photons are present per event,
      filling missing values with None.
    - If the GenPart_iso branch is not included in the NanoAOD, the GenIsolatedPhotons collection is used, to be consistent with get_fiducial_flag() above.
    """
    if 'iso' in events.GenPart.fields:
        sel_pho = (events.GenPart.pdgId == 22) & (events.GenPart.status == 1) & (events.GenPart.iso * events.GenPart.pt < 10)
        gen_photons = events.GenPart[sel_pho]
        gen_photons = gen_photons[ak.argsort(gen_photons.pt, ascending=False)]
        gen_photons = ak.pad_none(gen_photons, 2)
    else:
        # Extract and pad the gen isolated photons
        gen_photons = events.GenIsolatedPhoton
        gen_photons = ak.pad_none(gen_photons, 2)

    # Separate leading and subleading photons
    lead_pho = gen_photons[:, 0]
    sublead_pho = gen_photons[:, 1]
    gen_diphoton = lead_pho + sublead_pho

    # Diphoton Variables
    pt = gen_diphoton.pt
    y = 0.5 * np.log((gen_diphoton.energy + gen_diphoton.pz) / (gen_diphoton.energy - gen_diphoton.pz))
    phi = gen_diphoton.phi

    return (pt, y, phi, lead_pho, sublead_pho)


def match_jet(reco_jets, gen_jets, n, fill_value, jet_size=0.4, jet_flav=False):
    """
    this helper function is used to identify if a reco jet (or lepton) has a matching gen jet (lepton) for MC,
    -> Returns an array with 3 possible values:
        0 if reco not genMatched,
        1 if reco genMatched,
        -999 if reco doesn't exist
    parameters:
    * reco_jets: (ak array) reco_jet from the jets collection.
    * gen_jets: (ak array) gen_jet from the events.GenJet (or equivalent, e.g. events.Electron) collection.
    * n: (int) nth jet to be selected.
    * fill_value: (float) value with wich to fill the padded none if nth jet doesnt exist in the event.
    """
    if n is not None:
        reco_jets_i = reco_jets[ak.local_index(reco_jets, axis=1) == n]
    else:
        # This is for arrays already split into separate event slices, used for Higgs an bjet matching.
        reco_jets_i = ak.singletons(reco_jets)
    reco_jets_i = ak.pad_none(reco_jets_i, 1, clip=True)

    candidate_jet_matches = ak.cartesian({"reco": reco_jets_i, "gen": gen_jets}, axis=1)
    candidate_jet_matches["deltaR_jj"] = DeltaR(
        candidate_jet_matches["reco"], candidate_jet_matches["gen"]
    )

    matched_jets = ak.firsts(
        candidate_jet_matches[
            ak.argmin(candidate_jet_matches["deltaR_jj"], axis=1, keepdims=True)
        ], axis=1
    )
    matched_jets_bool = matched_jets["deltaR_jj"] < jet_size

    if jet_flav:
        matched_gen_flav = ak.where(
            matched_jets_bool, matched_jets["gen"].partonFlavour, fill_value
        )
        matched_gen_flav = ak.where(
            ~ak.is_none(ak.firsts(reco_jets_i)), matched_gen_flav, fill_value
        )
        return matched_gen_flav
    else:
        matched_jets_bool = ak.where(
            ~ak.is_none(ak.firsts(reco_jets_i)), matched_jets_bool, fill_value
        )
        return matched_jets_bool


def match_fatjet_hbb(reco_jets, gen_jets, n, fill_value, jet_size=0.8):
    reco_jets_i = reco_jets[ak.local_index(reco_jets, axis=1) == n]
    reco_jets_i = ak.pad_none(reco_jets_i, 1, clip=True)

    candidate_jet_matches = ak.cartesian({"reco": reco_jets_i, "gen": gen_jets}, axis=1)
    candidate_jet_matches["deltaR_jj"] = DeltaR(
        candidate_jet_matches["reco"], candidate_jet_matches["gen"]
    )

    # Count number of gen jets within jet_size for each reco jet
    close_matches = candidate_jet_matches["deltaR_jj"] < jet_size
    match_count = ak.sum(close_matches, axis=1)

    return ak.fill_none(match_count == 2, fill_value)


def match_jet_to_genpart(reco_jet, gen_parts, fill_value, jet_size=0.4):
    """
    Match a single reco jet to the closest gen particle within jet_size.
    No filtering on pdgId is applied; gen_parts should be pre-filtered
    (e.g. isLastCopy) before calling this function.

    Parameters:
    * reco_jet: (ak array) single reco jet (flat, e.g. result of ak.firsts)
    * gen_parts: (ak array) gen particles registered as PtEtaPhiMCandidate,
                 must carry fields 'pdgId' and 'mother_pdgId'
                 (mother_pdgId pre-computed from genPartIdxMother)
    * fill_value: value used when the reco jet is absent
    * jet_size: DeltaR cone size for matching

    Returns:
    * matched_pdgId: pdgId of closest gen particle within jet_size,
                     0 if reco jet exists but no match found, fill_value if absent
    * matched_mother_pdgId: pdgId of its mother under the same conditions
    """
    reco_jet_singleton = ak.singletons(reco_jet)
    reco_jet_singleton = ak.pad_none(reco_jet_singleton, 1, clip=True)

    pairs = ak.cartesian({"reco": reco_jet_singleton, "gen": gen_parts}, axis=1)
    pairs["deltaR"] = DeltaR(pairs["reco"], pairs["gen"])

    best_match = ak.firsts(
        pairs[ak.argmin(pairs["deltaR"], axis=1, keepdims=True)], axis=1
    )
    is_matched = ak.fill_none(best_match["deltaR"] < jet_size, False)
    reco_exists = ~ak.is_none(reco_jet)

    matched_pdgId = ak.values_astype(
        ak.where(
            reco_exists & is_matched,
            best_match["gen"].pdgId,
            ak.where(reco_exists, 0, fill_value),
        ),
        np.int32,
    )
    matched_mother_pdgId = ak.values_astype(
        ak.where(
            reco_exists & is_matched,
            best_match["gen"].mother_pdgId,
            ak.where(reco_exists, 0, fill_value),
        ),
        np.int32,
    )
    return matched_pdgId, matched_mother_pdgId


def _count_decay_products(
    decay_products: ak.Array,
    particle_type: str,
    particle_type_map: dict[str, tuple[int, ...]] | None = None,
) -> ak.Array:
    """Return per-event counts of decay products matching a requested particle type."""
    abs_pdg_ids = np.abs(decay_products.pdgId)

    particle_map = particle_type_map or DEFAULT_ASSOCIATED_DECAY_PARTICLE_MAP
    pdg_ids = particle_map.get(particle_type)
    if pdg_ids is None:
        raise ValueError(f"Unknown particle type requirement: {particle_type}")

    mask = np.isin(abs_pdg_ids, pdg_ids)
    return ak.sum(mask, axis=1)


def classify_associated_decay(
    events: ak.Array,
    associate: dict[str, any],
    decay: dict[str, any],
    categories: dict[str, dict[str, int]],
    n_higgs: int = 1,
    orthogonal_categories: bool = False,
    particle_type_map: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, ak.Array] | tuple[dict[str, ak.Array], dict[str, int]]:
    """
    Identify associated-production decay topologies using generator-level information.

    Parameters
    ----------
    events
        NanoEvents-like object with ``GenPart`` and ``distinctChildrenDeep`` available.
    associate
        Mapping describing the associated particle (``pdgId`` and expected ``multiplicity``).
    decay
        Mapping describing which decay products to inspect and their relationship to the associate.
    categories
        Mapping of output category name to particle-type multiplicity requirements.
    n_higgs
        Expected number of hard-process Higgs bosons in the event. Default (and the only tested value) is 1.
    orthogonal_categories
        If True, raise when more than one category matches within the same associate definition.
    particle_type_map
        Optional mapping from particle-type keys (e.g. ``n_lep``) to PDG IDs; defaults to
        ``DEFAULT_ASSOCIATED_DECAY_PARTICLE_MAP``.
    """

    if not hasattr(events, "GenPart"):
        raise AttributeError("Events must have GenPart collection for associated decay classification")

    if n_higgs < 1:
        raise ValueError("n_higgs must be at least 1")
    if n_higgs > 1:
        logger.warning("n_higgs > 1 is untested; please validate results carefully")

    # Get Higgs and associated particle(s)
    higgs = events.GenPart[(np.abs(events.GenPart.pdgId) == 25) & events.GenPart.hasFlags("isHardProcess")]
    has_higgs = ak.num(higgs) == n_higgs
    if not ak.any(has_higgs):
        logger.warning("No events with a hard-process Higgs found; skipping associated decay classification.")
        return {}
    higgs = ak.firsts(higgs)
    multiplicity = associate.get("multiplicity", 1)
    associate_particles = events.GenPart[
        (np.abs(events.GenPart.pdgId) == associate["pdgId"])
        & events.GenPart.hasFlags("isHardProcess")
        & (events.GenPart.genPartIdxMother == higgs.genPartIdxMother)
    ]
    n_associate = ak.num(associate_particles)
    valid_associate = n_associate == multiplicity
    if not ak.any(valid_associate):
        logger.warning("No events with the required associated particle multiplicity found; skipping associated decay classification.")
        return {}

    # Get decay products of the associated particle(s), possibly deeper in the tree
    if decay["relationship_to_associate"] == "self":
        decay_products = associate_particles.distinctChildrenDeep
        decay_products = ak.flatten(decay_products, axis=2)
    elif decay["relationship_to_associate"] == "child":
        children = ak.flatten(
            associate_particles.distinctChildrenDeep[
                np.abs(associate_particles.distinctChildrenDeep.pdgId) == decay["pdgId"]
            ],
            axis=2,
        )
        decay_products = children.distinctChildrenDeep
        decay_products = ak.flatten(decay_products, axis=2)
    else:
        raise NotImplementedError("Only 'self' or 'child' relationships are supported")

    # Keep only the direct children of the parent to avoid picking up radiation etc
    decay_product_parents = decay_products.parent
    is_last_copy_child = decay_product_parents.hasFlags("isLastCopy")
    decay_products = decay_products[is_last_copy_child]

    # Categorise
    categories_masks: dict[str, ak.Array] = {}
    for cat_name, reqs in categories.items():
        mask = valid_associate
        for particle_type, n_required in reqs.items():
            n_particles = _count_decay_products(
                decay_products, particle_type, particle_type_map
            )
            mask = mask & (n_particles == n_required)
        categories_masks[cat_name] = mask
    if orthogonal_categories and categories_masks:
        stack = ak.concatenate([mask[None, ...] for mask in categories_masks.values()], axis=0)
        overlap = ak.sum(stack, axis=0) > 1
        if ak.any(overlap):
            raise ValueError("Orthogonal categories requested but overlap found for associated decays")

    return categories_masks


def label_associated_decay(
    events: ak.Array,
    configs: list[dict[str, any]],
    *,
    default_label: str = "unclassified",
    raise_on_overlap: bool = False,
    particle_type_map: dict[str, tuple[int, ...]] | None = None,
) -> tuple[ak.Array, dict[str, ak.Array]]:
    """
    Run ``classify_associated_decay`` for multiple configurations and return a per-event label.

    Parameters
    ----------
    particle_type_map
        Optional mapping from particle-type keys (e.g. ``n_lep``) to PDG IDs; defaults to
        ``DEFAULT_ASSOCIATED_DECAY_PARTICLE_MAP`` when not provided.

    Returns
    -------
    labels
        Awkward Array of strings, one per event.
    all_masks
        Mapping of category name to boolean event masks for downstream use.
    """

    labels = ak.Array([default_label] * len(events.event))
    all_masks: dict[str, ak.Array] = {}
    for cfg in configs:
        masks = classify_associated_decay(
            events=events,
            associate=cfg["associate"],
            decay=cfg["decay"],
            categories=cfg["categories"],
            n_higgs=cfg.get("n_higgs", 1),
            orthogonal_categories=cfg.get("orthogonal_categories", False),
            particle_type_map=particle_type_map,
        )

        for cat_name, mask in masks.items():
            labels = ak.where(mask, ak.full_like(labels, cat_name), labels)
            all_masks[cat_name] = mask

    if all_masks:
        stack = ak.concatenate([mask[None, ...] for mask in all_masks.values()], axis=0)
        overlap = ak.sum(stack, axis=0) > 1
        n_overlap = int(ak.sum(overlap))
        if n_overlap > 0:
            msg = f"Associated decay categories overlap for {n_overlap} events."
            if raise_on_overlap:
                raise ValueError(msg)
            logger.warning(msg)

    return labels, all_masks
