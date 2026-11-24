"""Plot the archived validation contrasts and the frozen directional decoder (no refit)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', nargs='?', type=Path, default=Path('results/l20-operators/directional'))
    out = p.parse_args().directory
    with (out/'development-gains.csv').open() as fp:
        rows = list(csv.DictReader(fp))
    model = json.loads((out/'channel.json').read_text()); meta = json.loads((out/'freeze.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12,8.5), constrained_layout=True)
    labels = ['Direction only', 'Untrained H', 'Hand-built H', 'Barrier', 'Barrier + untrained H', 'Barrier + hand-built H']
    for ax, metric, title in zip(axes[0], ('logloss','direction_mse'), ('Bernoulli log-score gain (nats/event)', 'Within-bond direction MSE reduction')):
        for j, lag in enumerate((4096,16384)):
            dif, err = [], []
            for base in ['direction','random','simple','barrier','barrier+random','barrier+simple']:
                mod = 'barrier+learned' if base.startswith('barrier') else 'learned'
                r, = [r for r in rows if r['metric']==metric and r['split']=='validation' and r['base']==base and r['model']==mod and int(r['lag'])==lag]
                gain = float(r['gain']); dif.append(gain); err.append(float(r['hi'])-gain)
            ax.errorbar(dif, np.arange(6)+.17*(2*j-1), xerr=err, fmt='o', ms=4, capsize=2, label=f'{lag:,} sweeps')
        ax.axvline(0, color='0.5', linewidth=1); ax.set_yticks(np.arange(6), labels if ax==axes[0,0] else ['']*6)
        ax.axhline(2.5, color='.85', linewidth=1); ax.invert_yaxis(); ax.set_xlabel(title); ax.legend(fontsize=8)
        ax.ticklabel_format(style='sci',axis='x',scilimits=(-2,2)); ax.grid(axis='x',alpha=.2)
    for ax, lag in zip(axes[1], (4096,16384)):
        table=model[str(lag)]; pr=expit(logit(table['direction']) + np.array(table['learned']))
        delta = 100*(pr - pr.mean(axis=1, keepdims=True))
        bound = max(abs(delta.min()),abs(delta.max()),1e-7)
        im=ax.imshow(delta, cmap='RdBu_r', vmin=-bound,vmax=bound,aspect='auto')
        for k,row in enumerate(pr):
            for d,pred in enumerate(row):
                ax.text(d,k,f'{100*pred:.1f}',ha='center',va='center',fontsize=9,color='white' if abs(delta[k,d])>.6*bound else 'black')
        ax.set_xticks(range(6), ['−u','+u','−v','+v','−w','+w']); ax.set_yticks(range(4),[f'H={k}  (n={cnt})' for k,cnt in enumerate(meta['learned_leaf_counts']['fit'])])
        ax.set_title(f'{lag:,} sweeps: fitted probability (%)',fontsize=11); ax.set_xlabel('Canonical face; +u points toward higher species')
        fig.colorbar(im, ax=ax, shrink=.85, label='Difference from symbol mean (pp)')
    fig.suptitle('Directional predictive compression · cold archived parents\nTop: learned code vs comparator; 16 withheld parents, paired 95% t intervals',fontsize=14)
    fig.savefig(out/'development.png', dpi=155); plt.close(fig)

if __name__=='__main__':
    main()
