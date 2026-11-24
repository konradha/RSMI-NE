"""Plot the registered analytic physical comparisons, using split receipts only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2]/'results/l20-autoresearch/tranche2'


def receipt(num, split):
    return json.loads((OUT/f'{num}-{split}.json').read_text())


def plot():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.0), layout='constrained')
    colors = ['#357494', '#bf542c']; rungs = ['256', '1024', '4096']
    for j, split in enumerate(['development', 'audit']):
        rows = receipt(19, split)['scores_by_n_fit']; xs = np.arange(3) + (j-.5)*.13
        vals = [rows[r]['relative_improvement'] for r in rungs]; mean = np.array([v['estimate'] for v in vals]); lo, hi = np.array([v['parent95'] for v in vals]).T
        ax.errorbar(xs, 100*mean, yerr=100*np.stack([mean-lo, hi-mean]), color=colors[j], marker='o', capsize=3, label=split.capitalize(), ls='none')
    ax.axhline(0, lw=.8, color='0.5'); ax.set_xticks(np.arange(3), rungs); ax.legend(frameon=False, loc='lower right')
    ax.set(xlabel='Number of fit parents', ylabel='Reduction in Dirichlet quotient (%)', title='Adding the frozen code to the physical basis')
    ax.text(.03, .96, 'Useful gain: 5% (well above scale)', transform=ax.transAxes, va='top', fontsize=9)
    g = receipt(20, 'audit'); totals = g['totals']; eligible = totals['eligible']; xx = np.arange(2)
    raw = [totals[z+'_count']/eligible for z in ['open', 'close']]; bx.scatter(xx-.09, raw, marker='s', color='0.45', label='Unweighted counterfactual')
    vals = [g[z+'_flux_per_eligible'] for z in ['opening', 'closing']]; mean = np.array([v['estimate'] for v in vals]); lo, hi = np.array([v['parent95'] for v in vals]).T
    bx.errorbar(xx+.09, mean, yerr=np.stack([mean-lo, hi-mean]), color=colors[1], marker='o', capsize=3, ls='none', label='Weighted by acceptance of first move')
    bx.axhline(.001, color='0.35', lw=.9, ls='--', label='Material opening rate'); bx.set_xticks(xx, ['Opens', 'Closes']); bx.set_xlim(-.3, 1.3); bx.set_yscale('log'); bx.set_ylim(3e-6, 1)
    bx.set(ylabel='Probability per eligible pair', title='The first move rarely passes its own gate'); bx.legend(frameon=False, fontsize=8, loc='upper right')
    for q in (ax, bx):
        q.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Ideal local Metropolis exchange · β = 3.05 · independent native TRAIN parents', fontsize=12)
    fig.savefig(OUT/'19-20-physical.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    plot()
