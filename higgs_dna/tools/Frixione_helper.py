import awkward as ak
import numpy
import numba


@numba.njit
def cumsum_2d(arr, builder):
    for _a in arr:
        builder.begin_list()
        entry = 0
        for __a in _a:
            entry = entry + __a
            builder.real(entry)
        builder.end_list()
    return builder


def pass_approximate_frixione_isolation(
    gen_particles: ak.Array,
    gen_photon: ak.Array,
    frix_cone=0.05
) -> ak.Array:
    """
    https://arxiv.org/pdf/hep-ph/9801442
    Check if a photon passes the Frixione isolation criterion.
    The original frixione isolation is evaluated with the hadrons but here with approximate frixione isolation we astimate the frxione isolation with genparts.
    Parameters:
    gen_particles: ak Array of gen particles with fields 'status', 'pdgId', 'pt', 'eta', 'phi', 'et'.
    photon_idx: Index of the photon to evaluate. The function can handel only one photon per event but can not handel akward array of multiple photon per event.
    frix_cone: Maximum cone size for Frixione isolation.
    Returns:
    pass_frix: Boolean, True if photon passes the Frixione criterion, False otherwise.
    """
    # Create a mask for gen particles in which we are interested in
    if (len(gen_photon.pt) == 0):
        return ak.Array([])

    gen_mask = (
        ((gen_particles.statusFlags & (1 << 7)) != 0)  # isHardProcess
        & (
            (abs(gen_particles.pdgId) == 11)    # Electrons
            | (abs(gen_particles.pdgId) == 13)  # Muons
            | (abs(gen_particles.pdgId) == 15)  # Tau
            | (abs(gen_particles.pdgId) < 10)   # Quarks
            | (abs(gen_particles.pdgId) == 21)  # Gluons
        )
        & (gen_particles.status != 21)  # drop the two incoming partons
    )

    valid_gen_particles = gen_particles[gen_mask]
    # calulate deltaR between photon and valid gen particles since we need to remove all which have dR bigger than frix cone
    deltaR_genpart_pho = valid_gen_particles.metric_table(gen_photon)

    # create a mask to remove the particle which not inside the biggest frix cone
    # and with ak.all make the deltaR_mask same shape as valid_gen_particles to be able to apply it on valid_gen_particles
    deltaR_mask = ak.all(deltaR_genpart_pho < frix_cone , axis=-1)
    valid_gen_particles = valid_gen_particles[deltaR_mask]

    # define photon and genpart ET
    Et_valid_gen_particles = numpy.sqrt((valid_gen_particles.pt)**2 + (valid_gen_particles.mass)**2)
    gen_pho_Et = numpy.sqrt(gen_photon.pt**2 + gen_photon.mass**2)

    # replace the [] array of Et with [None], later we need to broadcast Et_valid_gen_particles, and this replacement will help
    Et_valid_gen_particles = ak.where(ak.num(Et_valid_gen_particles) == 0, [[None]], Et_valid_gen_particles)

    # define deltR between pho and genparts to have the same dimension as Photon because we need to check particles around given photon
    deltaR_pho_genPart = gen_photon.metric_table(valid_gen_particles)
    # since there are events with no photons, and corresponding deltaR_pho_gepart will be None, to replace None with []
    deltaR_pho_genPart = ak.fill_none(deltaR_pho_genPart, [[]], axis=0)
    deltaR_pho_genPart = ak.flatten(deltaR_pho_genPart, axis=1)  # since we are have passed only one photon in the fuction can plation the delatR_pho_genPart
    sorted_deltaR_pho_genPart = ak.sort(deltaR_pho_genPart, axis=-1)
    sorted_indices_deltaR_pho_genPart = ak.argsort(deltaR_pho_genPart, axis=-1)

    # sort the Et_valid_gen_particles_brodcasted according to shorted_indices_deltaR_pho_genPart to that we can calulate the cumsum of Et_valid_gen_particles_brodcasted
    sorted_Et_valid_gen_particles = Et_valid_gen_particles[sorted_indices_deltaR_pho_genPart]

    # calulate (1 - cos(deltaR)) / (1 - cos(frix_cone)) for each deltaR
    cone_vars = (1 - numpy.cos(sorted_deltaR_pho_genPart)) / (1 - numpy.cos(frix_cone))

    # calculate the cumulative sum of Et_valid_gen_particles
    cumsum_GenEt = cumsum_2d(sorted_Et_valid_gen_particles, ak.ArrayBuilder()).snapshot()

    # Photon ET prodcast is naccessary because sometimes there is no cumsum_GenEt but photn ET is there and we need to divide cumsum_GenEt by photon ET
    pho_Et_broadcasted, _ = ak.fill_none(ak.broadcast_arrays(gen_pho_Et, cumsum_GenEt), [], axis=1)
    cumsum_GenEt_over_phoET = cumsum_GenEt / pho_Et_broadcasted
    pass_frix = ak.all(cumsum_GenEt_over_phoET < cone_vars , axis=-1)

    has_gen_photon = ~ak.is_none(gen_photon.pt)

    # for events with no gen photon, pass_frix will be an empty array, so we need to replace an empty array with False
    pass_frix = ak.where(has_gen_photon, pass_frix, False)  # no gen photon set to False

    return pass_frix


def attach_geninfo_to_photons(
    photons: ak.Array
) -> ak.Array:
    """
    Attaches gen information to the photons.
    Parameters:
    photons: ak Array of photons with fields 'pt', 'eta', 'phi', 'et'.
    Returns:
    photons: ak Array of photons with additional fields for gen information.
    """
    Genmatched_photon = photons.matched_gen

    photons["statusFlags"] = Genmatched_photon.statusFlags
    ispromptfinalstate = (
        ((Genmatched_photon.statusFlags & (1 << 0)) != 0)
        & (Genmatched_photon.status == 1)
    )
    photons["ispromptfinalstate"] = ispromptfinalstate

    Genmatched_photon_parent = Genmatched_photon.parent
    photon_mom_genpdgId = Genmatched_photon_parent.pdgId
    photons["mom_statusFlags"] = Genmatched_photon_parent.statusFlags
    photons["mom_pdgId"] = Genmatched_photon_parent.pdgId

    Genmatched_photon_parent_parent = Genmatched_photon_parent.parent
    photon_mommom_genpdgId = Genmatched_photon_parent_parent.pdgId
    photons["mommom_pdgId"] = photon_mommom_genpdgId
    isHardProcess = (Genmatched_photon_parent.statusFlags & (1 << 7)) != 0
    photons["isHardProcess"] = isHardProcess

    isFromToptoWb = (
        (abs(photon_mom_genpdgId) == 24)
        | (abs(photon_mommom_genpdgId) == 24)
        | ((abs(photon_mom_genpdgId)) == 5 & (abs(photon_mommom_genpdgId) == 6))
    )
    photons["isFromToptoWb"] = isFromToptoWb
    isFromQuark = (
        (abs(photon_mom_genpdgId) == 21)
        | (abs(photon_mommom_genpdgId) == 21)
        | ((abs(photon_mom_genpdgId) <= 6) & (abs(photon_mommom_genpdgId) != 6) & (abs(photon_mommom_genpdgId) != 24))
    )
    photons["isFromQuark"] = isFromQuark
    isFromProton = (abs(photon_mom_genpdgId) == 2212)
    photons["isFromProton"] = isFromProton

    return photons


def attach_frixione_isolation_flag_to_diphotons(
    gen_parts: ak.Array,
    diphotons: ak.Array,
    frix_cones=[0.05]
) -> ak.Array:
    """
    Attaches Frixione isolation information to the diphotons.
    Parameters:
    photons: ak Array of photons with fields 'pt', 'eta', 'phi', 'et'.
    frix_cones: List of cone sizes for Frixione isolation.
    Returns:
    photons: ak Array of diphotons with additional fields for Frixione isolation flag, ispythia flag and ET sum in a given cone devided by photon pt.
    Every added field name will have the cone size in its name multiplied by 100.
    """

    for frix_cone in frix_cones:
        cone = f"{int(round(frix_cone * 100)):02d}"
        pass_frix_name = f"pass_frix_R{cone}"
        ispythia_name = f"ispythia_R{cone}"

        flag_pass_frix_lead = pass_approximate_frixione_isolation(gen_parts, diphotons["pho_lead"].matched_gen, frix_cone=frix_cone)
        flag_pass_frix_sublead = pass_approximate_frixione_isolation(gen_parts, diphotons["pho_sublead"].matched_gen, frix_cone=frix_cone)

        diphotons["pho_lead"] = ak.with_field(diphotons["pho_lead"], flag_pass_frix_lead, pass_frix_name)
        diphotons["pho_sublead"] = ak.with_field(diphotons["pho_sublead"], flag_pass_frix_sublead, pass_frix_name)
        lead_isPythia = (
            (~ ((diphotons["pho_lead"].isHardProcess) & (diphotons["pho_lead"].mom_pdgId == 22)))
            & diphotons["pho_lead"].ispromptfinalstate
            & (
                diphotons["pho_lead"].isFromToptoWb
                | (diphotons["pho_lead"].isFromQuark & ~diphotons["pho_lead"][pass_frix_name])
                | diphotons["pho_lead"].isFromProton
            )
        )
        sublead_isPythia = (
            (~ ((diphotons["pho_sublead"].isHardProcess) & (diphotons["pho_sublead"].mom_pdgId == 22)))
            & diphotons["pho_sublead"].ispromptfinalstate
            & (
                diphotons["pho_sublead"].isFromToptoWb
                | (diphotons["pho_sublead"].isFromQuark & ~diphotons["pho_sublead"][pass_frix_name])
                | diphotons["pho_sublead"].isFromProton
            )
        )

        # False for photons which do not have a matched gen photon
        lead_isPythia = ak.fill_none(lead_isPythia, False)
        sublead_isPythia = ak.fill_none(sublead_isPythia, False)
        diphotons["pho_lead"] = ak.with_field(diphotons["pho_lead"], lead_isPythia, ispythia_name)
        diphotons["pho_sublead"] = ak.with_field(diphotons["pho_sublead"], sublead_isPythia, ispythia_name)
    return diphotons
