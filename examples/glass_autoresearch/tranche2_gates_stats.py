"""Fit-only whitening and barrier standardization; independent parent-level scoring."""
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import t
from fifteen_dynamics_closure import sha, store

N = 8000
NAMES = ['A_density', 'B_density', 'owner_energy', 'period_two_vacancy', 'label_0', 'label_1', 'label_2']


def interval(point, influence):
    v = np.asarray(influence); se = v.std(ddof=1)/np.sqrt(len(v))
    def bounds(q):
        h = t.ppf(q, len(v)-1)*se
        return [float(point-h), float(point+h)]
    return dict(estimate=float(point), parent_SE=float(se), parent95=bounds(.975), two_primary97_5=bounds(.9875))


def solve(F, D, k):
    cov = np.cov(F[:, :k], rowvar=False); scales = np.sqrt(np.diag(cov)); use = scales > 0
    if not use.any():
        raise ValueError('zero observable variance')
    x = scales[use]; sub = np.ix_(use, use)
    eig, U = np.linalg.eigh(cov[sub]/(x[:, None]*x[None, :]))
    keep = eig > 1e-10*eig[-1]; W = (U[:, keep]/np.sqrt(eig[keep]))/x[:, None]
    mat = W.T @ D.mean(0)[:k, :k][sub] @ W
    vals, V = np.linalg.eigh((mat+mat.T)*.5); v = np.zeros(7); v[np.flatnonzero(use)] = W@V[:, 0]
    if v[np.argmax(abs(v))] < 0:
        v *= -1
    return dict(coef=v.tolist(), fit_rate=float(vals[0]), quotient_spectrum=vals.tolist(), correlation_eigenvalues=eig.tolist(),
                retained_dimensions=int(keep.sum()), raw_dimensions=k, fit_variance=float(v[:k]@cov@v[:k]))


def sparse(data, member):
    return csr_matrix((data[member], data['keys'], data['indptr']), shape=(len(data['gate']), int(data['ncols'])))


def fit_locks(out, fields, dmat, data, B):
    models = {str(n): {name:solve(fields[:n], dmat[:n], k) for name, k in [('conventional', 4), ('augmented', 7)]}
              for n in (256, 1024, 4096)}
    store(out/'19-fit-model.json', dict(field_order=NAMES, convention='Fit covariance ddof=1; unit fit variance, largest absolute loading oriented positive.', models=models))
    counts = np.asarray(sparse(data, 'counts').sum(axis=0)).reshape(-1, 4); kept = np.flatnonzero(counts.min(axis=1) >= 32)
    w = counts[kept].sum(axis=1); mass = w.sum()/counts.sum()
    store(out/'20-fit-barriers.json', dict(key_formula='128*max(dE_e,0)+dE_f, f initially closed', threshold=32,
          barrier_key=kept.tolist(), pair=[[int(z//B), int(z%B)] for z in kept], pooled_mass=(w/w.sum()).tolist(),
          counts_by_label_0_to_3=counts[kept].tolist(), fit_fraction_closed_eligible_retained=float(mass), all_fit_closed_eligible=int(counts.sum())))


def field_metrics(F, D, vector):
    v = np.asarray(vector); fv = F@v; d = np.einsum('i,pij,j->p', v, D, v, optimize=True)
    n = len(fv); d0 = d.mean(); centered = fv-fv.mean(); vs = (n/(n-1))*centered**2; v0 = vs.mean()
    if not (d0 >= 0 and v0 > 0):
        raise ValueError('invalid frozen-field Dirichlet/variance')
    r = d0/v0; rd = (d-r*vs)/v0
    cov = np.cov(np.column_stack([F[:, :2], fv]), rowvar=False); overlap = cov[:2, 2]
    proj = float(overlap@np.linalg.pinv(cov[:2, :2], hermitian=True)@overlap/v0)
    table = dict(field_mean=float(fv.mean()), df=n-1, variance=interval(v0, vs), dirichlet_per_attempt=interval(d0, d),
                 rayleigh_per_attempt=interval(r, rd), rayleigh_times_N_per_sweep=interval(N*r, N*rd),
                 variance_projection_R2_on_A_B_waves=proj, corr_with_A_B_waves=(overlap/np.sqrt(np.diag(cov)[:2]*v0)).tolist())
    return table, r, rd


def ratios(num, den):
    mean = den.mean()
    if mean <= 0:
        raise ValueError('unidentified zero denominator')
    r = num.mean()/mean
    return r, (num-r*den)/mean


def contrast(a, b):
    return a[0]-b[0], a[1]-b[1]


def decision(obj, floor):
    lo, hi = obj['parent95']
    return 'resolved above practical floor' if lo > floor else 'resolved below practical floor' if hi < floor else 'interval includes practical floor'


def compare_labels(data, lock):
    n = sparse(data, 'counts'); flux = sparse(data, 'flux')
    keys = np.asarray(lock['barrier_key']); w = np.asarray(lock['pooled_mass'])
    cols = 4*keys[:, None]+np.arange(4); shape = (n.shape[0], len(keys), 4)
    c = n[:, cols.ravel()].toarray().reshape(shape); y = flux[:, cols.ravel()].toarray().reshape(shape)
    m = c.sum(0); supported = m.min(1) > 0
    if not supported.any():
        return dict(status='unidentified: no four-label common support')
    mass = float(w[supported].sum()); w = w[supported]/mass; c = c[:, supported]; y = y[:, supported]
    theta = y.sum(0)/m[supported]; by = w@theta
    inf = np.einsum('pls,l->ps', (y-c*theta)/c.mean(0), w)
    return dict(standardized_per_initially_closed_eligible_by_label=[interval(q, inf[:, j]) for j, q in enumerate(by)],
                label_3_minus_0=interval(by[3]-by[0], inf[:, 3]-inf[:, 0]), fitted_target_mass_supported=mass,
                supported_strata=int(supported.sum()), fit_selected_strata=len(keys), split_fraction_closed_eligible_retained=float(c.sum()/n.sum()),
                sparse_total_closed_eligible=int(n.sum()), arm_closed_eligible=c.sum(axis=(0,1)).tolist(),
                estimand='Conditional mean a_e*Iopen, initially closed only, common exact positive f barrier and clipped e barrier mass fixed on fit; not an independent causal effect.')


def report(out, stage, fields, dmat, data, lineage, timing):
    models_path = out/'19-fit-model.json'; models = json.loads(models_path.read_text())['models']
    ladders = {}
    for n, mod in models.items():
        base, r0, dr0 = field_metrics(fields, dmat, mod['conventional']['coef'])
        aug, r1, dr1 = field_metrics(fields, dmat, mod['augmented']['coef'])
        effect = interval(1-r1/r0, r1/r0**2*dr0-dr1/r0)
        ladders[n] = dict(conventional=base, augmented=aug, relative_improvement=effect,
                          absolute_improvement_per_attempt=interval(r0-r1, dr0-dr1),
                          absolute_improvement_times_N_per_sweep=interval((r0-r1)*N, (dr0-dr1)*N), materiality=decision(effect, .05))
    forms = {name: field_metrics(fields, dmat, np.eye(len(NAMES))[j])[0] for j, name in enumerate(NAMES)}
    core = dict(stage=stage, parents=len(fields), evidence_sha256=sha(out/f'19-{stage}-parents.npz'), lineage=lineage, timing_controls=timing,
                uncertainty='Independent whole-parent paired delta-method t intervals, condition on fit coefficients/weights for separate evaluation; fit-only inference descriptive.')
    store(out/f'19-{stage}.json', dict(**core, item=19, edges_per_parent=3*N, exhaustive_attempts=3*N*len(fields),
          fit_model_sha256=sha(models_path), field_order=NAMES, scores_by_n_fit=ladders, single_field_scores=forms,
          primary_model='4096', relative_improvement_floor=.05, prediction='below floor',
          interpretation='Ideal reversible random NN exchange: population D/Var upper-bounds the spectral gap per single attempt. Reported numbers are finite-parent estimators. Multiplication by N is just a unit conversion, not a finite-lag or sweep probability. Density k != 0 is not exactly conserved, but a large density-span R2 warns against a distinct glass-mode interpretation. Fluctions in invariant zero-wavevector inventories are exactly zero.',
          citations=['https://arxiv.org/abs/1211.7103', 'https://arxiv.org/abs/1407.8083','https://doi.org/10.1073/pnas.2427246122']))
    col = dict(zip(data['columns'].tolist(), data['gate'].T)); eligible = col['eligible']
    op = ratios(col['open_flux'], eligible); cl = ratios(col['close_flux'], eligible)
    lock = out/'20-fit-barriers.json'; labels = compare_labels(data, json.loads(lock.read_bytes()))
    opp = interval(*op)
    totals = {k:(int(v.sum()) if k in ('eligible', 'open_count', 'close_count', 'far_eligible', 'far_crossings',
                     'changed_acceptance_count', 'far_changed_acceptance_count') else float(v.sum())) for k,v in col.items()}
    store(out/f'20-{stage}.json', dict(**{**core, 'evidence_sha256':sha(out/f'20-{stage}-parents.npz')},item=20,
          fit_barriers_sha256=sha(lock), anchor_opportunities=N*len(fields), totals=totals, eligible_fraction=interval(eligible.mean()/N, eligible/N),
          opening_flux_per_eligible=opp, closing_flux_per_eligible=interval(*cl), signed_open_minus_close_per_eligible=interval(*contrast(op, cl)),
          acceptance_e_per_eligible=interval(*ratios(col['accepted_e'], eligible)),
          opening_flux_per_named_x_edge_attempt=interval(col['open_flux'].mean()/N, col['open_flux']/N),
          opening_flux_per_any_native_attempt_counting_named_x_geometry_only=interval(col['open_flux'].mean()/(3*N), col['open_flux']/(3*N)),
          increase_on_openings_per_eligible=interval(*ratios(col['open_magnitude_flux'], eligible)),
          decrease_on_closings_per_eligible=interval(*ratios(col['close_magnitude_flux'], eligible)),
          acceptance_increase_given_weighted_opening=interval(*ratios(col['open_magnitude_flux'], col['open_flux'])),
          acceptance_decrease_given_weighted_closing=interval(*ratios(col['close_magnitude_flux'], col['close_flux'])),
          barrier_matched_label_result=labels, practical_floor=.001, prediction='above floor', materiality=decision(opp, .001),
          far_control='Same parallel orientation and all anchor translations; distance 5 vs 1. No shared owner terms: exact zero mixed barrier and zero opening or closing. Explicit 12-owner scalar substitution verified periodically.',
          reverse_balance='The paired e reverse has dE = -dE_e and reversed gate crossing. exp(-beta*dE_e)*a(-dE_e)=a(dE_e). The parent-estimated difference O-C above is a stationarity diagnostic, not signed creation. Raw counts are unweighted and need not balance.',
          citations=['https://doi.org/10.1103/PhysRevLett.132.067101','https://doi.org/10.1073/pnas.2416800122']))
