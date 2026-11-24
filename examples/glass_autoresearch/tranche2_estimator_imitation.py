"""Auxiliary capacity diagnostic: copy historical labels using ONLY native TRAIN-fit tile frequencies, never score on held-out faces."""
import tranche2_estimator as e

import time
from types import SimpleNamespace

import numpy as np


def run():
    start = time.monotonic()
    v, _ = e.extract(0,4096)
    freq = np.bincount(v.ravel(), minlength=e.V).astype('f4')/v.size
    ch = SimpleNamespace(p=freq.astype('f4'))
    source=e.ROOT/'results/l20-autoresearch/faces/optimized.npz'
    with np.load(source) as arch: teacher=arch['discrete_lut']
    rec={}
    for seed in e.SEEDS:
        p=e.init(seed); ap=e.Adam(p,.005); scores={}
        for step in range(e.CHECK[-1]+1):
            if step:
                _,g=e.full_obj(p,ch,'distilled',teacher); ap.update(g)
            if step in e.CHECK:
                _,q=e.forward(p,e.X); h=q.argmax(1)
                scores[str(step)]=dict(fit_agreement=float(ch.p[h==teacher].sum(dtype='f8')),fit_CE_nats=float(-ch.p @ np.log(np.maximum(q[np.arange(e.V),teacher],1e-38))))
        rec[str(seed)]=scores
    e.jout(e.OUT/'21-published-imitation-fit-only.json',dict(context='Auxiliary direct capacity test requested separately. The old pooled teacher (historical attainable 0.025972 bits) saw overlapping historical TRAIN parents; its labels are not an independent gold standard. Only fit V frequencies and published map labels enter this fit. No development/audit agreement or face relevance measured; does not enter estimator comparison, model selection or predictions.',source_sha256=e.digest(source),fit_parents=4096,seconds=time.monotonic()-start,paired_seeds=rec))


if __name__=='__main__': run()
