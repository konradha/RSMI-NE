"""Dense, validation-only precision follow-through for the original additive residual."""
import argparse
import csv
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zipfile

for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[var] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy.stats import t

from rate import source_rows
from sectors import EXAMPLES, INDICES, L, PACK, STATIC, VIS, VT, dump, save, sha
from l20_operators_static import CODE, dense, softmax
from l20_pairs import partition

ROOT = EXAMPLES.parent / 'results/l20-autoresearch'
DESIGN = ROOT / 'dense-range/design.json'
LAGS = np.array([0, 3, 4, 5, 6, 7, 8, 9])
CHANNELS = ('residual_0', 'residual_1', 'residual_2', 'residual_3', 'residual_trace', 'A')
VOLUME = L**3
N = 2048


def require(value, message):
    if not value:
        raise ValueError(message)


class IdentityResidual:
    """Static encoder plus the sealed sectors.Fields projection, identity only."""
    def __init__(self, template, prior, design):
        archive = STATIC / 'fit-receipts.zip'
        require(sha(archive) == prior['static_archive_sha256'], 'frozen fit archive changed')
        require(sha(template) == prior['template_sha256'] == design['additive_sha256'], 'additive template changed')
        with zipfile.ZipFile(archive) as z:
            payload = z.read('fits/' + design['model'] + '.npz')
        require(hashlib.sha256(payload).hexdigest() == design['fit_npz_sha256'], 'fit member changed')
        with np.load(io.BytesIO(payload), allow_pickle=False) as f:
            self.w, self.b, self.u, self.a, self.c = (f['enc' + str(i)] for i in range(5))
        with np.load(template, allow_pickle=False) as f:
            key = json.loads(str(f['key']))
            require(key == {k: prior[k] for k in ('nonlinear', 'affine_sha256', 'affine_seed', 'static_archive_sha256', 'train_source')}, 'wrong template training provenance')
            self.coef = f['coef']
        self.h = len(self.b)
        self.weights = np.concatenate((self.w, self.a, self.coef[:-1]), axis=1)
        self.bias = np.concatenate((self.b, self.c, self.coef[-1]))

    def __call__(self, state):
        # One SGEMM for all preactivations of all 8000 blocks, no orbit construction.
        visible = np.take(state, VT, mode='clip')
        x = np.take(CODE, visible, axis=0, mode='clip').reshape(VOLUME, -1)
        z = x @ self.weights
        z += self.bias
        h = self.h
        np.tanh(z[:, :h], out=z[:, :h])
        residual = softmax(z[:, :h] @ self.u + z[:, h:h+4]) - z[:, h+4:]
        out = np.empty((5, VOLUME), dtype=np.float64)
        out[:4] = residual.T
        out[4] = state == 1
        return out.reshape(5, L, L, L)

    def direct_error(self, state, field):
        # Independent periodic addressing at the box seams; no VT or combined weights.
        centers = np.array([(0,0,0), (19,19,19), (0,19,7), (11,0,19), (5,7,9), (19,4,0)])
        coords = np.moveaxis((centers[:,None] + VIS) % L, -1, 0)
        x = dense(state.reshape((L,)*3)[tuple(coords)])
        ref = softmax(np.tanh(x @ self.w + self.b) @ self.u + x @ self.a + self.c)
        ref -= x @ self.coef[:-1] + self.coef[-1]
        return np.max(np.abs(ref - field[(slice(None,4), *centers.T)].T))


def axial(spectrum):
    # conj(FFT(x))*FFT(y) -> sum_x f(x)g(x+r); FFT inverse already normalizes.
    cube = fft.irfftn(spectrum, s=(L,L,L), axes=(1,2,3), workers=1) / VOLUME
    return np.stack((cube[:, LAGS, 0, 0], cube[:, 0, LAGS, 0], cube[:, 0, 0, LAGS]), axis=0).transpose(0,2,1)


def literal(x, y):
    return np.array([[np.mean(x * np.roll(y, -r, axis=axis), axis=(1,2,3)) for r in LAGS] for axis in (1,2,3)])


def packet(states, field, verify):
    values = [field(row) for row in states]
    spectra = [fft.rfftn(x, axes=(1,2,3), workers=1) for x in values]
    same = np.empty((PACK, len(LAGS), 5))
    mismatch = np.empty_like(same)
    errors = np.zeros(3)
    for i, (x, s) in enumerate(zip(values, spectra)):
        j = (i+1) % PACK
        conjugate = s.conj()
        auto = axial(conjugate*s)
        cross = axial(conjugate*spectra[j])
        same[i] = auto.mean(0)
        mismatch[i] = cross.mean(0)
        if verify and i < 3:
            errors = np.maximum(errors, (field.direct_error(states[i], x),
                                np.max(np.abs(auto-literal(x,x))), np.max(np.abs(cross-literal(x,values[j])))))
    require(errors[0] < 3e-6 and np.max(errors[1:]) < 1e-12, f'identity/periodicity checks failed: {errors}')
    counts = np.stack([np.count_nonzero(states == s, axis=1) for s in (1,2)], axis=1)
    return dict(raw_same=same, raw_mismatch=mismatch, spatial_mean=np.array([x.mean((1,2,3)) for x in values]), counts=counts, check_errors=errors)


def compute(index, data, field, old, work, key, workers):
    (lookup, selected), meta = source_rows(data, index, (1,))
    require(meta['sha256'] == old['sha256'] and meta['lineage_sha256']['1'] == old['group_lineage_sha256'][1], 'endpoint or validation lineage changed')
    require(meta['beta'] == old['beta'] and meta['selected_folds'] == [1], 'wrong temperature or fold')
    (_, ids, _), _ = partition(index)
    require(len(ids) == N and np.all(lookup[ids] >= 0), 'incomplete validation group')
    rows = selected[lookup[ids]]
    del selected
    def run(p):
        path = work / f'{key}-b{index:02}-p{p:03}.npz'
        if not path.exists():
            save(path, packet(rows[p*PACK:(p+1)*PACK], field, p == 0))
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(run, range(N//PACK)))
    stats = {k: np.stack([v[k] for v in parts]) for k in parts[0]}
    counts = stats['counts']
    require(np.all(counts == counts[0,0]), 'A or B composition fluctuates')
    stats['parent_ids'] = ids.reshape(-1, PACK)
    return meta, stats


def connected(stats):
    mean = stats['spatial_mean']
    # Baseline is unbiased for the square of the ensemble mean, not mean_i(m_i^2).
    base = (mean*(mean.sum(axis=1,keepdims=True)-mean)).sum(1)/(PACK*(PACK-1))
    def channels(a):
        return np.concatenate((a[...,:4], a[...,:4].sum(-1,keepdims=True), a[...,4:]), axis=-1)
    cov = {kind:channels(stats['raw_'+kind].mean(1) - base[:,None,:]) for kind in ('same','mismatch')}
    # Variance of each spatial mean estimates Var(physical k=0); no per-parent centering at finite r.
    zero = channels(VOLUME * np.var(mean, axis=1, ddof=1))
    abund = stats['counts'][:,0,0] / VOLUME
    error = np.max(np.abs(cov['same'][:,0,-1] - abund*(1-abund)))
    require(error < 1e-12, f'incorrect A contact covariance: {error}')
    return cov, zero, error


def interval(groups, multiplicity=1):
    n = len(groups)
    m = groups.mean(0)
    hw = (groups.std(0, ddof=1)/np.sqrt(n)) * t.isf(.025/multiplicity, n-1)
    return np.stack((m,m-hw,m+hw), axis=-1)


def equiv(iv, delta):
    if iv[1] > -delta and iv[2] < delta:
        return 'within_margin'
    if iv[1] > delta or iv[2] < -delta:
        return 'beyond_margin'
    return 'unresolved'


def overlap(a,b):
    if a[2] < b[1] or b[2] < a[1]: return 'disjoint'
    if a[1] >= b[1] and a[2] <= b[2]: return 'dense_inside_sparse'
    if b[1] >= a[1] and b[2] <= a[2]: return 'sparse_inside_dense'
    return 'partial_overlap'


def csv_out(path, rows):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def summarize(index, stats, old, delta, ax, rows, modes):
    cov, zero, contact_err = connected(stats)
    focal = int(np.flatnonzero(LAGS==6)[0]); trace = CHANNELS.index('residual_trace')
    for kind, x in cov.items():
        for k, ch in enumerate(CHANNELS):
            family = 126 if kind=='mismatch' else (21 if ch=='A' else 105)
            singles = interval(x[:,:,k]); bands = interval(x[:,:,k], family)
            for j, r in enumerate(LAGS):
                iv, band = singles[j], bands[j]
                rows.append(dict(index=index, beta=old['sources'][str(index)]['beta'], pairing=kind, channel=ch, separation=r, empty_site_gap=r-3 if r else '', parents=N, packets=N//PACK, mean=iv[0], lo95=iv[1], hi95=iv[2], equivalent95=equiv(iv,delta), family_size=family if r else '', simultaneous_lo95=band[1] if r else '', simultaneous_hi95=band[2] if r else '', simultaneous_equivalence=equiv(band,delta) if r else ''))
    for label, groups in zip((*CHANNELS, 'B'), (*zero.T, VOLUME*np.var(stats['counts'][:,:,1]/VOLUME,axis=1,ddof=1))):
        m,l,h = interval(groups)
        modes.append(dict(index=index, channel=label, mean=m, lo95=l, hi95=h))
    require(np.max(np.abs(zero[:,-1])) < 1e-20, 'fixed A k=0 check failed')
    bands = interval(cov['same'][:,:,trace],105)
    points = interval(cov['same'][:,:,trace]); point = points[focal]
    gaps = LAGS[1:]-3; m,lo,hi=bands[1:].T
    ax.fill_between(gaps,lo,hi,alpha=.18,color='#126379',label='95% simultaneous (105 cells)')
    ax.errorbar(gaps,m,yerr=np.stack((m-points[1:,1],points[1:,2]-m)),fmt='o-',markersize=3,color='#126379',capsize=2,label='Trace ± pointwise 95%')
    ax.axhline(0,lw=.7,color='.3'); ax.axhline(delta,color='.45',lw=.7,ls='--'); ax.axhline(-delta,color='.45',lw=.7,ls='--')
    ax.set(xlabel='Empty-site gap (r − 3)',xticks=gaps,title=f"β = {old['sources'][str(index)]['beta']:g}")
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    sparse = old['calibration'][str(index)]['residual_at_six']
    dense6 = interval(cov['same'][:,focal,trace],6)
    hw = (point[2]-point[1])/2
    null_band = interval(cov['mismatch'][:,focal,trace])
    return dict(contact_trace=points[0].tolist(), r6_trace_pointwise=point.tolist(), r6_pointwise_halfwidth=hw,
                meets_previous_precision_halfwidth_0_0005=bool((dense6[2]-dense6[1])/2 <= .0005),
                r6_trace_simultaneous=bands[focal].tolist(), r6_pointwise_equivalence=equiv(point,delta),
                r6_simultaneous_equivalence=equiv(bands[focal],delta), r6_pointwise_resolves_sign=bool(point[1]>0 or point[2]<0),
                simultaneous_trace_gaps_within_margin=gaps[(lo>-delta)&(hi<delta)].tolist(),
                r6_sparse_original=sparse, r6_dense_at_original_6look_coverage=dense6.tolist(),
                interval_overlap_matched_coverage=overlap(dense6,sparse), interval_overlap_pointwise_vs_sparse=overlap(point,sparse),
                width_improvement_matched_coverage=(sparse[2]-sparse[1])/(dense6[2]-dense6[1]),
                r6_mismatched_trace_pointwise=null_band.tolist(), mismatch_zero_inside_r6_pointwise=bool(null_band[1]<=0<=null_band[2]),
                A_r6_pointwise=interval(cov['same'][:,focal,-1]).tolist(),
                residual_k0_trace=interval(zero[:,trace]).tolist(), A_B_count_per_parent=stats['counts'][0,0].tolist(),
                A_contact_max_error=contact_err, direct_checks_max_error=dict(zip(('identity_field','same_roll','mismatch_roll'), stats['check_errors'][0].tolist())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--out',type=Path,default=DESIGN.parent)
    p.add_argument('--workers',type=int,choices=range(1,9),default=2)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True); a.work.mkdir(parents=True,exist_ok=True)
    design=json.loads(DESIGN.read_bytes()); spec=design['prior']; prior_path=ROOT / spec['path']
    require(sha(prior_path)==spec['sha256'], 'sparse reference receipt changed')
    old=json.loads(prior_path.read_bytes()); field=IdentityResidual(a.template,old['sources']['models'],spec)
    provenance=dict(design_sha256=sha(DESIGN), script_sha256=sha(Path(__file__)), prior_sha256=spec['sha256'],
                    source_loader_sha256=sha(Path(__file__).with_name('rate.py')), sectors_sha256=sha(Path(__file__).with_name('sectors.py')),
                    split_code_sha256=sha(EXAMPLES/'l20_pairs.py'), field_utilities_sha256=sha(EXAMPLES/'l20_operators_static.py'),
                    fit_member_sha256=spec['fit_npz_sha256'], models=old['sources']['models'])
    key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()[:24]
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.1),layout='constrained')
    rows=[]; modes=[]; results={}; sources={}; arrays=dict(lags=LAGS,raw_channels=np.array(('residual_0','residual_1','residual_2','residual_3','A')))
    for index,ax in zip(INDICES,axes,strict=True):
        sources[index],stats=compute(index,a.data,field,old['sources'][str(index)],a.work,key,a.workers)
        arrays.update({f'b{index:02}_{k}':v for k,v in stats.items()})
        results[index]=summarize(index,stats,old,design['equivalence_margin'],ax,rows,modes)
        print(f"β-index {index}: r6 [estimate, lo95, hi95] = {results[index]['r6_trace_pointwise']}",flush=True)
    axes[0].set_ylabel('Signed residual trace covariance [probability²]'); axes[0].legend(fontsize=7,frameon=False)
    fig.suptitle('Frozen additive residual · all translations, three axes · dashed lines at ±0.001')
    fig.savefig(a.out/'range.png',dpi=160); plt.close(fig)
    save(a.out/'parent-moments.npz',arrays); csv_out(a.out/'covariance.csv',rows); csv_out(a.out/'zero-mode.csv',modes)
    dump(a.out/'summary.json',dict(provenance=provenance,sources=sources,parents_per_index=N,independent_packets=N//PACK,
         interval_order=['estimate','lo','hi'],equivalence_margin=design['equivalence_margin'],results=results,
         outputs_sha256={n:sha(a.out/n) for n in ('parent-moments.npz','range.png','covariance.csv','zero-mode.csv')},
         interpretation='Same validation parents, improved spatial sampling of a fixed lab-frame field. Interval overlap is descriptive, not a paired significance test or an independent replication. Bounds are signed covariances at measured axial separations in one finite box, not a correlation length or evidence of thermodynamic order.'))


if __name__=='__main__':
    main()
