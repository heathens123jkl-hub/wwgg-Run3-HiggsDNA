"""Output-only DeepTau trial on existing WWgg-selected objects."""
import awkward as ak
import numpy as np


def dr2(a, b):
    dphi = np.arctan2(np.sin(a.phi-b.phi), np.cos(a.phi-b.phi))
    return (a.eta-b.eta)**2 + dphi**2


def require_fields(array, fields, label):
    missing = set(fields)-set(ak.fields(array))
    if missing:
        raise ValueError(label+' missing fields: '+', '.join(sorted(missing)))


def evaluate(taus, diphoton, electrons, muons, cfg):
    if cfg['mode'] not in ('store_only', 'apply'):
        raise ValueError('This trial only supports store_only')
    names = ['idDeepTau2018v2p5'+s for s in ('VSjet', 'VSe', 'VSmu')]
    require_fields(taus, ['pt','eta','phi','dz','charge','decayMode']+names, 'Tau')
    for obj, label in ((electrons,'Electron'),(muons,'Muon')):
        require_fields(obj, ['eta','phi','charge'], label)
    # Preserve local object indexes so saved accepted pairs can be traced back.
    taus = ak.with_field(taus, ak.local_index(taus, axis=1), '_trial_index')
    electrons = ak.with_field(electrons, ak.local_index(electrons, axis=1), '_trial_index')
    muons = ak.with_field(muons, ak.local_index(muons, axis=1), '_trial_index')
    finite = ak.ones_like(taus.pt, dtype=bool)
    for key in ['pt','eta','phi','dz','charge','decayMode']:
        finite = finite & np.isfinite(taus[key])
    valid_id = ak.ones_like(taus.pt, dtype=bool)
    for key, maximum in zip(names, (8,8,4)):
        x = taus[key]
        valid_id = valid_id & np.isfinite(x) & (x >= 0) & (x <= maximum) & (x == np.floor(x))
    zero_id = ak.zeros_like(taus.pt, dtype=bool)
    for key in names:
        zero_id = zero_id | (taus[key] == 0)
    kin = finite & (taus.pt > cfg['tau_pt_min']) & (abs(taus.eta) < cfg['tau_abs_eta_max'])
    kin = kin & (abs(taus.dz) < cfg['tau_abs_dz_max']) & (abs(taus.charge) == 1)
    for dm in cfg['excluded_decay_modes']:
        kin = kin & (taus.decayMode != dm)
    # Only the saved diphoton candidate is used. No mutation or jet cleaning.
    for photon in (diphoton.pho_lead, diphoton.pho_sublead):
        kin = kin & (dr2(taus, photon) > cfg['tau_photon_dr_min']**2)
    kin = ak.fill_none(kin, False)
    out = {
        'tau_trial_n_raw': ak.num(taus, axis=1),
        'tau_trial_n_kin': ak.sum(kin, axis=1),
        'tau_trial_n_invalid': ak.sum(~ak.fill_none(finite & valid_id, False), axis=1),
        'tau_trial_n_zero_id_kin': ak.sum(kin & zero_id, axis=1),
    }
    for key in ('pt','eta','phi','dz','charge','decayMode') + tuple(names):
        out['tau_trial_tau_'+key] = taus[key]
    for key in ('genPartFlav','genPartIdx'):
        # Synthetic tests can omit truth; real MC runner requires it explicitly.
        out['tau_trial_tau_'+key] = taus[key] if key in taus.fields else ak.full_like(taus.pt,-999,dtype=np.int32)
    out['tau_trial_tau_pass_kin'] = kin
    for label,objects in (('electron',electrons),('muon',muons)):
        for key in ('eta','phi','charge','pt','genPartFlav','genPartIdx'):
            out['tau_trial_'+label+'_'+key] = objects[key] if key in objects.fields else ak.full_like(objects.eta,-999,dtype=np.int32)
    for kind, wps in cfg['working_points'].items():
        cut = kin & valid_id
        for axis, threshold in wps.items():
            maximum = 4 if axis == 'VSmu' else 8
            if not isinstance(threshold, int) or not 1 <= threshold <= maximum:
                raise ValueError('Invalid WP: '+axis)
            cut = cut & (taus['idDeepTau2018v2p5'+axis] >= threshold)
        selected = taus[ak.fill_none(cut,False)]
        out['tau_trial_n_'+kind] = ak.num(selected, axis=1)
        if kind == 'tautau':
            pairs = ak.combinations(selected, 2, fields=['a','b'], axis=1)
        else:
            partner = electrons if kind == 'etau' else muons
            pairs = ak.cartesian({'a':selected,'b':partner}, axis=1)
        separated = dr2(pairs.a, pairs.b) > cfg['pair_dr_min']**2
        os = separated & (pairs.a.charge * pairs.b.charge < 0)
        out['tau_trial_'+kind+'_os'] = ak.fill_none(ak.any(os,axis=1),False)
        decision = os if cfg['require_opposite_sign'] else separated
        out['tau_trial_'+kind] = ak.fill_none(ak.any(decision,axis=1),False)
        accepted = pairs[ak.fill_none(decision,False)]
        out['tau_trial_'+kind+'_pair_tau_index'] = accepted.a._trial_index
        out['tau_trial_'+kind+'_pair_partner_index'] = accepted.b._trial_index
        out['tau_trial_'+kind+'_pair_opposite_sign'] = accepted.a.charge * accepted.b.charge < 0
    out['tau_trial_veto'] = out['tau_trial_etau'] | out['tau_trial_mutau'] | out['tau_trial_tautau']
    out['tau_trial_veto_os'] = out['tau_trial_etau_os'] | out['tau_trial_mutau_os'] | out['tau_trial_tautau_os']
    return out


def efficiency(weights, rejected):
    """Signed-weight fraction; covariance accounted via disjoint pass/fail sums."""
    w = np.asarray(weights, dtype=float)
    reject = np.asarray(rejected, dtype=bool)
    if w.shape != reject.shape or not np.all(np.isfinite(w)):
        raise ValueError('Invalid weights or mask')
    a,b = w[reject],w[~reject]
    sa,sb = float(a.sum()),float(b.sum())
    va,vb = float(a@a),float(b@b)
    total = sa+sb
    frac = sa/total if total > 0 else None
    status = 'VALID' if frac is not None and 0 <= frac <= 1 else 'NONPHYSICAL_OR_EMPTY'
    error = np.sqrt(((1-frac)**2*va + frac**2*vb)/total**2) if status=='VALID' else None
    # An all-pass/all-fail finite sample cannot establish zero uncertainty.
    if len(a)==0 or len(b)==0:
        error = None
        if status=='VALID': status='BOUNDARY_NO_GAUSSIAN_ERROR'
    return dict(rows=len(w),rejected_rows=len(a),negative_weight_rows=int((w<0).sum()),
                sumw_before=total,sumw_rejected=sa,sumw_retained=sb,
                sumw2_before=va+vb,sumw2_rejected=va,
                veto_fraction=frac,retention_fraction=1-frac if frac is not None else None,
                stat_error=float(error) if error is not None else None,status=status)
