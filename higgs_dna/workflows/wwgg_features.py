"""Flat, output-only WWgg observables. No cuts, weights or candidates are changed.

Every floating observable has a *_valid column; unavailable values are NaN.
Truth has a separate namespace and is never a classification input by default.
Only Awkward and NumPy are needed, allowing tests without importing Coffea.
"""
import awkward as ak
import numpy as np

SCHEMA_VERSION = "2"


def dphi(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def field(obj, name, default=np.nan):
    if name in ak.fields(obj):
        return ak.fill_none(obj[name], default)
    return ak.full_like(obj.pt, default, dtype=np.float64)


def p4(obj):
    pt, eta, phi, mass = (field(obj, k) for k in ("pt", "eta", "phi", "mass"))
    px, py, pz = pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)
    return dict(px=px, py=py, pz=pz, energy=np.sqrt(mass**2+pt**2+pz**2))


def add4(*objects):
    return {k: sum(o[k] for o in objects) for k in ("px", "py", "pz", "energy")}


def kinematics(v):
    pt = np.hypot(v['px'], v['py'])
    return dict(pt=pt, phi=np.arctan2(v['py'], v['px']),
                eta=np.arcsinh(v['pz']/ak.where(pt > 0, pt, np.nan)),
                mass=np.sqrt(np.maximum(0, v['energy']**2-pt**2-v['pz']**2)),
                energy=v['energy'])


def dr(a, b):
    return np.hypot(field(a, 'eta')-field(b, 'eta'), dphi(field(a, 'phi'), field(b, 'phi')))


def leptons(electrons, muons):
    """Common primitive records, without merging incompatible NanoAOD behaviors."""
    groups = []
    for collection, flavour, iso in ((electrons, 11, 'pfRelIso03_all'),
                                    (muons, 13, 'pfRelIso04_all')):
        values = {k: field(collection, k) for k in ('pt','eta','phi','mass','charge','dxy','dz')}
        values['flavour'] = ak.full_like(collection.pt, flavour, dtype=np.int32)
        values['reliso'] = field(collection, iso)
        for key, relevant in (('cutBased',11), ('mvaIso_WP80',11),
                              ('mvaIso_WP90',11), ('tightId',13), ('mediumId',13)):
            values[key] = field(collection,key) if flavour == relevant else ak.full_like(collection.pt,np.nan,dtype=np.float64)
        for key in ('genPartIdx','genPartFlav'):
            values[key] = field(collection, key, -1)
        groups.append(ak.zip(values))
    result = ak.concatenate(groups, axis=1)
    return result[ak.argsort(result.pt, axis=1, ascending=False, stable=True)]


def take_one(collection, indices):
    """Safe per-event local index lookup, including zero-length collections."""
    valid = ak.fill_none((indices >= 0) & (indices < ak.num(collection)), False)
    indices = ak.values_astype(ak.where(valid, indices, 0), np.int64)
    found = ak.firsts(ak.pad_none(collection, 1)[ak.singletons(indices)])
    return ak.mask(found, valid)


def build_features(events, diphoton, electrons, muons, jets, is_mc):
    """Arguments are the already selected event rows and their selected objects."""
    n = len(diphoton)
    out = {}

    def put(name, value, valid=None):
        value = np.asarray(ak.to_numpy(ak.fill_none(value, np.nan)), dtype=np.float64)
        good = np.isfinite(value)
        if valid is not None:
            good &= np.asarray(ak.to_numpy(ak.fill_none(valid, False)), dtype=bool)
        if value.shape != (n,):
            raise ValueError('Non-flat WWgg feature: '+name)
        out[name] = ak.Array(np.where(good, value, np.nan))
        out[name+'_valid'] = ak.Array(good)

    def flag(name, value):
        out[name] = ak.values_astype(ak.fill_none(value, False), np.bool_)

    def dump_object(prefix, obj, extras=()):
        flag(prefix+'_present', ~ak.is_none(obj.pt, axis=0))
        for key in ('pt','eta','phi','mass')+tuple(extras):
            put(prefix+'_'+key, field(obj, key))
        put(prefix+'_energy', p4(obj)['energy'])

    ls = leptons(electrons, muons)
    js = jets[ak.argsort(jets.pt, axis=1, ascending=False, stable=True)]
    lp = ak.pad_none(ls, 2)
    jp = ak.pad_none(js, 4)
    l1, l2 = lp[:,0], lp[:,1]
    photons = [diphoton.pho_lead, diphoton.pho_sublead]
    for i, photon in enumerate(photons, 1):
        name = 'feat_pho'+str(i)
        dump_object(name, photon, ('mvaID','r9'))
        put(name+'_pt_over_mgg', photon.pt/diphoton.mass)
        put(name+'_energy_over_mgg', p4(photon)['energy']/diphoton.mass)
    for i, lepton in enumerate((l1,l2), 1):
        dump_object('feat_lep'+str(i), lepton,
                    ('charge','flavour','reliso','dxy','dz','cutBased','mvaIso_WP80',
                     'mvaIso_WP90','tightId','mediumId'))
    for i in range(4):
        dump_object('feat_jet'+str(i+1), jp[:,i], ('btagDeepFlavB',))

    for key, value in (('nlep',ak.num(ls)), ('nele',ak.num(electrons)),
                       ('nmu',ak.num(muons)), ('njet',ak.num(js)),
                       ('ht',ak.sum(js.pt, axis=1))):
        put('feat_'+key, value)
    for key in ('pt','eta','phi','mass'):
        put('feat_dipho_'+key, diphoton[key])
    put('feat_photon_id_min', np.minimum(photons[0].mvaID, photons[1].mvaID))
    put('feat_photon_id_max', np.maximum(photons[0].mvaID, photons[1].mvaID))
    put('feat_dphi_gg', abs(dphi(photons[0].phi, photons[1].phi)))
    put('feat_dr_gg', dr(*photons))

    b = field(js, 'btagDeepFlavB')
    b = b[np.isfinite(b)]
    b = ak.pad_none(b[ak.argsort(b, axis=1, ascending=False)], 2)
    put('feat_bscore_max', b[:,0])
    put('feat_bscore_second', b[:,1])
    put('feat_bscore_top2_sum', b[:,0]+b[:,1])
    for name, objs in (('ll',(l1,l2)), ('jj',(jp[:,0],jp[:,1])),
                       ('jj34',(jp[:,2],jp[:,3])),
                       ('4j',tuple(jp[:,i] for i in range(4)))):
        v = add4(*(p4(obj) for obj in objs))
        for key, value in kinematics(v).items():
            put('feat_'+name+'_'+key, value)
    put('feat_dr_ll', dr(l1,l2))
    put('feat_dphi_ll', abs(dphi(l1.phi,l2.phi)))
    have_ll = ak.num(ls) >= 2
    put('feat_ll_os', l1.charge*l2.charge < 0, have_ll)
    put('feat_ll_sf', l1.flavour == l2.flavour, have_ll)
    put('feat_dr_jj', dr(jp[:,0],jp[:,1]))
    fourjet = kinematics(add4(*(p4(jp[:,i]) for i in range(4))))
    put('feat_dphi_dipho_4j',abs(dphi(diphoton.phi,fourjet['phi'])))
    put('feat_dr_dipho_4j',np.hypot(diphoton.eta-fourjet['eta'],dphi(diphoton.phi,fourjet['phi'])))
    pairs = ak.combinations(js, 2, fields=['a','b'])
    for label, values in (
        ('jet_jet', dr(pairs.a,pairs.b)),
        ('jet_photon', ak.concatenate([dr(js, g) for g in photons], axis=1))):
        put('feat_dr_'+label+'_min', ak.min(values, axis=1, mask_identity=True))
        put('feat_dr_'+label+'_max', ak.max(values, axis=1, mask_identity=True))
    # Both photon pairings for each leading selected lepton, to permit explicit
    # alternative e-gamma veto definitions later. Never silently replace Z_megamma.
    for i, lepton in enumerate((l1,l2),1):
        for j, photon in enumerate(photons,1):
            mass = kinematics(add4(p4(lepton),p4(photon)))['mass']
            put(f'feat_lep{i}_pho{j}_mass', mass)
            put(f'feat_ele{i}_pho{j}_abs_dmZ', abs(mass-91.2), lepton.flavour == 11)
    eg = ak.concatenate([kinematics(add4(p4(electrons),p4(g)))['mass']
                         for g in photons], axis=1)
    put('feat_eg_abs_dmZ_min_all', ak.min(abs(eg-91.2), axis=1, mask_identity=True))

    met = events.PuppiMET
    for key in ('pt','phi','sumEt','significance','covXX','covXY','covYY'):
        put('feat_met_'+key, field(met,key))
    metpt, metphi = field(met,'pt'), field(met,'phi')
    for i, lepton in enumerate((l1,l2),1):
        angle = abs(dphi(lepton.phi,metphi))
        put(f'feat_dphi_met_lep{i}', angle)
        put(f'feat_mt_lep{i}_met', np.sqrt(np.maximum(0,2*lepton.pt*metpt*(1-np.cos(angle)))))
    ll = kinematics(add4(p4(l1),p4(l2)))
    put('feat_dphi_met_ll', abs(dphi(metphi,ll['phi'])))
    put('feat_dphi_met_dipho', abs(dphi(metphi,diphoton.phi)))
    put('feat_dphi_met_jet_min', ak.min(abs(dphi(js.phi,metphi)), axis=1, mask_identity=True))
    nearest = ak.min(abs(dphi(ls.phi,metphi)), axis=1, mask_identity=True)
    put('feat_projected_met', metpt*np.sin(np.minimum(nearest,np.pi/2)))
    llv = add4(p4(l1),p4(l2))
    mt2 = (np.sqrt(ll['mass']**2+ll['pt']**2)+metpt)**2 - (
        (llv['px']+metpt*np.cos(metphi))**2+(llv['py']+metpt*np.sin(metphi))**2)
    put('feat_mt_ll_met', np.sqrt(np.maximum(0,mt2)))

    # Source truth is descriptive only: no truth-dependent cuts, no fake label
    # inferred from a missing match. genPartIdx uses the NanoAOD local index.
    truth_available = is_mc and 'GenPart' in ak.fields(events)
    out['truth_available'] = ak.Array(np.full(n,truth_available,dtype=bool))
    gen = events.GenPart if truth_available else None
    for name, obj in [('pho1',photons[0]),('pho2',photons[1]),('lep1',l1),('lep2',l2)]:
        prefix = 'truth_'+name
        idx = field(obj,'genPartIdx',-1)
        put(prefix+'_genPartFlav', field(obj,'genPartFlav'),
            (field(obj,'genPartFlav',-1)>=0) & truth_available)
        match = take_one(gen,idx) if truth_available else None
        valid = ~ak.is_none(match.pt,axis=0) if truth_available else np.zeros(n,dtype=bool)
        flag(prefix+'_matched', valid)
        for key in ('pdgId','status','statusFlags','pt','eta','phi'):
            put(prefix+'_'+key, field(match,key) if truth_available else np.full(n,np.nan),valid)
        if truth_available:
            mother = take_one(gen,field(match,'genPartIdxMother',-1))
            put(prefix+'_mother_pdgId',field(mother,'pdgId'))
            put(prefix+'_dr',dr(obj,match),valid)
            status = ak.values_astype(field(match,'statusFlags',0),np.int64)
            for label, bit in (('prompt',0),('direct_prompt_tau_decay',5)):
                put(prefix+'_'+label,(status & (1<<bit))!=0,valid)
        else:
            for key in ('mother_pdgId','dr','prompt','direct_prompt_tau_decay'):
                put(prefix+'_'+key,np.full(n,np.nan))
    for key in ('pt','phi'):
        value = field(events.GenMET,key) if is_mc and 'GenMET' in ak.fields(events) else np.full(n,np.nan)
        put('truth_genmet_'+key,value)
    if truth_available:
        last = (gen.statusFlags & (1<<13)) != 0
        nh = ak.sum(last & (abs(gen.pdgId)==25),axis=1)
    else:
        nh = np.full(n,np.nan)
    put('truth_n_higgs_lastcopy',nh)
    put('truth_genWeight',events.genWeight if is_mc else np.full(n,np.nan))
    # Selected filters are recorded for provenance, not used to change selection.
    for key in ('goodVertices','globalSuperTightHalo2016Filter','EcalDeadCellTriggerPrimitiveFilter',
                'BadPFMuonFilter','BadPFMuonDzFilter','hfNoisyHitsFilter','eeBadScFilter','ecalBadCalibFilter'):
        value = events.Flag[key] if 'Flag' in ak.fields(events) and key in ak.fields(events.Flag) else np.full(n,np.nan)
        put('audit_filter_'+key,value)
    return out
