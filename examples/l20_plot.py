"""Plot and tabulate the complete L20 first pass: python examples/l20_plot.py RESULTS."""
import argparse
import csv
from itertools import combinations, permutations, product
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


BUFFERS = (0, 1, 2)
SEEDS = {'learned': (11, 29, 47), 'uniform': (11,), 'staggered': (11,)}
LABEL = {'learned': 'Learned', 'uniform': 'Uniform / composition', 'staggered': 'Checkerboard'}
COLOR = {'learned': '#176593', 'uniform': '#bf531c', 'staggered': '#078365'}
N_BATCH = 32
BATCH_SIZE = 128
T95 = 2.0395134463964077  # two-sided 95% Student-t critical value, 31 df
METRICS = ('paired', 'shuffled', 'paired_minus_shuffled')
CHECKER = (1 - 2 * (np.indices((2, 2, 2)).sum(axis=0) % 2)).ravel()  # + at z=y=x=0
CI_NOTE = ('Pointwise 95% t intervals over 32 matched batches of 128 test parents; '
           'anchors and learned seeds averaged before inference. Conditional on the fitted models.')


def load_fits(root):
    fits, betas = {}, {}
    expected = {(i, b, method, seed) for i in range(10) for b in BUFFERS
                for method, seeds in SEEDS.items() for seed in seeds}
    for path in sorted((root / 'fits').glob('*.json')):
        fit = json.loads(path.read_text())
        i, b, method, seed = key = tuple(fit[k] for k in ('index', 'buffer', 'method', 'seed'))
        if key in fits or key not in expected:
            raise ValueError(f'{path}: duplicate or unexpected fit {key}')
        beta = float(fit['beta'])
        if not np.isfinite(beta) or betas.setdefault(i, beta) != beta:
            raise ValueError(f'{path}: inconsistent beta for index {i}')
        batches = np.array([[r['paired'], r['shuffled']] for r in fit['test_batches']], dtype=float)
        w = np.asarray(fit['weights'], dtype=float)
        if batches.shape != (N_BATCH, 2) or w.shape != (8,):
            raise ValueError(f'{path}: expected 32 matched test batches and 8 zyx weights')
        if not np.isfinite(batches).all() or not np.isfinite(w).all():
            raise ValueError(f'{path}: non-finite scores or weights')
        fits[key] = (batches, w)
    missing = expected.difference(fits)
    if missing:
        raise ValueError(f'Expected 150 distinct fits; missing {len(missing)} (first: {min(missing)})')
    if len(set(betas.values())) != 10:
        raise ValueError('Expected ten distinct inverse temperatures')
    return fits, betas


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(scores):
    return dict(zip(METRICS, (scores[:, 0], scores[:, 1], scores[:, 0] - scores[:, 1])))


def stats(values):
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(N_BATCH)
    return dict(mean_nats=mean, se_nats=se, low95_nats=mean - T95 * se, high95_nats=mean + T95 * se)


def summarize(root, fits, betas, order):
    summary, contrasts = {}, {}
    for i in order:
        for b in BUFFERS:
            scores = {}
            cell = dict(index=i, beta=betas[i], buffer=b, test_batches=N_BATCH,
                        test_parents=N_BATCH * BATCH_SIZE)
            for method, seeds in SEEDS.items():
                # Each seed saw the same test parents, in the same batches. Do not stack batches.
                by_seed = [fits[i, b, method, seed][0] for seed in seeds]
                scores[method] = metrics(np.mean(by_seed, axis=0))
                for metric, values in scores[method].items():
                    seed_values = [metrics(s)[metric].mean() for s in by_seed]
                    summary[i, b, method, metric] = dict(cell, method=method, seeds=len(seeds), metric=metric,
                                                         **stats(values), seed_min_nats=min(seed_values),
                                                         seed_max_nats=max(seed_values))
            for fixed in ('uniform', 'staggered'):
                for metric in METRICS:
                    diff = scores['learned'][metric] - scores[fixed][metric]
                    contrasts[i, b, fixed, metric] = dict(cell, contrast=f'learned_minus_{fixed}', metric=metric,
                                                         **stats(diff))
    write_csv(root / 'summary.csv', list(summary.values()))
    write_csv(root / 'comparisons.csv', list(contrasts.values()))
    return summary, contrasts


def curve(ax, x, rows, label, color, ls='-'):
    ax.fill_between(x, [r['low95_nats'] for r in rows], [r['high95_nats'] for r in rows],
                    color=color, alpha=.13, linewidth=0)
    ax.plot(x, [r['mean_nats'] for r in rows], ls, color=color, label=label, marker='o', markersize=3)


def zero_line(ax):
    ax.axhline(0, color='.55', linewidth=.7, zorder=0)
    ax.spines[['right', 'top']].set_visible(False)


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.15, facecolor='white')
    plt.close(fig)


def plot_signal(root, summary, betas, order):
    x = [betas[i] for i in order]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.5), sharex=True)
    fig.subplots_adjust(wspace=.34, hspace=.24, bottom=.17, top=.86)
    for j, b in enumerate(BUFFERS):
        top, bottom = axes[:, j]
        top.set_title(f'Buffer b = {b}')
        for ax in (top, bottom):
            zero_line(ax)
        for metric, ax, label, color, ls in (
                ('paired', top, 'Paired', COLOR['learned'], '-'),
                ('shuffled', top, 'Shuffled control', '#a04a31', '--'),
                ('paired_minus_shuffled', bottom, 'Paired − shuffled', '#4e467c', '-')):
            curve(ax, x, [summary[i, b, 'learned', metric] for i in order], label, color, ls)
        for ax in (top, bottom):
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(-.002, lo), max(.002, hi))
        bottom.set_xlabel('Inverse temperature β  →  colder')
    axes[0, 0].set_ylabel('Held-out InfoNCE (nats)')
    axes[1, 0].set_ylabel('Paired − shuffled (nats)')
    axes[0, 0].legend(frameon=False)
    fig.suptitle('L20 lattice glass  |  Learned ternary coarse graining  ·  separate vertical scales', fontsize=15)
    fig.text(.065, .035, CI_NOTE, fontsize=9)
    save(fig, root / 'signal.png')


def plot_comparators(root, summary, contrasts, betas, order):
    x = [betas[i] for i in order]
    cold = order[-1]  # protocol endpoint, not selected by score
    fig = plt.figure(figsize=(13.3, 8.1))
    fig.subplots_adjust(top=.86, bottom=.16, hspace=.56, wspace=.18)
    grid = fig.add_gridspec(2, 1)
    top = grid[0].subgridspec(1, 3)
    bottom = grid[1].subgridspec(1, 2, wspace=.27)
    first = None
    for j, b in enumerate(BUFFERS):
        ax = fig.add_subplot(top[j], sharey=first)
        if first is None:
            first = ax
            ax.set_ylabel('Held-out paired bound (nats)')
        else:
            ax.tick_params(labelleft=False)
        zero_line(ax)
        for method, ls in zip(SEEDS, ('-', '--', '-.')):
            curve(ax, x, [summary[i, b, method, 'paired'] for i in order], LABEL[method], COLOR[method], ls)
        ax.set(title=f'Buffer b = {b}', xlabel='Inverse temperature β')
    first.legend(frameon=False, fontsize=9)
    for j, (metric, title) in enumerate((('paired', 'Paired bound'), ('paired_minus_shuffled', 'Paired − shuffled'))):
        ax = fig.add_subplot(bottom[j])
        zero_line(ax)
        for fixed, shift, marker in (('uniform', -.07, 'o'), ('staggered', .07, 's')):
            rows = [contrasts[cold, b, fixed, metric] for b in BUFFERS]
            ax.errorbar(np.asarray(BUFFERS) + shift, [r['mean_nats'] for r in rows],
                        yerr=[T95 * r['se_nats'] for r in rows], fmt=marker, capsize=4,
                        color=COLOR[fixed], label=f'Learned − {LABEL[fixed].lower()}')
        ax.set_xticks(BUFFERS)
        ax.set(xlabel='Buffer b', ylabel='Matched difference (nats)', xlim=(-.35, 2.35),
               title=f'Coldest β = {betas[cold]:g}  |  {title}')
        if j == 0:
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Learned and fixed operators on identical test parents', fontsize=15)
    fig.text(.07, .06, CI_NOTE, fontsize=9)
    fig.text(.07, .035, 'Both fixed templates are reported separately. No test-selected winner or multiplicity correction; differences include critic fit.', fontsize=9)
    save(fig, root / 'comparators.png')


def unit(w):
    norm = np.linalg.norm(w)
    # A zero filter has no orientation; leave angles undefined in both CSVs and plots.
    return w / norm if norm else np.full(8, np.nan)


def symmetries():
    cube = np.arange(8).reshape(2, 2, 2)
    return np.array([cube.transpose(p)[sl].ravel() for p in permutations(range(3))
                     for sl in product((slice(None), slice(None, None, -1)), repeat=3)])


def filter_tables(root, fits, betas, order):
    overlaps, stability = [], []
    group = symmetries()
    uniform, checker = unit(np.ones(8)), unit(CHECKER)
    for i in order:
        for b in BUFFERS:
            cell = dict(index=i, beta=betas[i], buffer=b)
            directions = {}
            for seed in SEEDS['learned']:
                w = fits[i, b, 'learned', seed][1]
                v = directions[seed] = unit(w)
                u, c = np.dot(v, uniform), np.dot(v, checker)
                overlaps.append(dict(cell, seed=seed, norm=np.linalg.norm(w), uniform_cos=u, checker_cos=c,
                                     abs_uniform_cos=abs(u), abs_checker_cos=abs(c)))
            for a, d in combinations(SEEDS['learned'], 2):
                va, vd = directions[a], directions[d]
                same = va @ vd
                aligned = vd[group] @ va
                stability.append(dict(cell, seed_a=a, seed_b=d, signed_cos=same, abs_cos=abs(same),
                                      max_signed_cos_48=aligned.max(), max_abs_cos_48=np.abs(aligned).max()))
    write_csv(root / 'filter_overlaps.csv', overlaps)
    write_csv(root / 'filter_stability.csv', stability)
    return overlaps, stability


def plot_filters(root, fits, betas, order, overlaps, stability):
    cold = order[-1]
    seeds = SEEDS['learned']
    b_map = 1
    maps = [fits[cold, b_map, 'learned', seed][1].reshape(2, 2, 2) for seed in seeds]
    limit = max(np.max(np.abs(w)) for w in maps) or 1.
    fig = plt.figure(figsize=(13.2, 10.8))
    fig.subplots_adjust(top=.90, bottom=.15, hspace=.53, wspace=.17, left=.08, right=.95)
    grid = fig.add_gridspec(3, 3, height_ratios=(.9, 1, 1))
    map_axes = []
    for j, (seed, cube) in enumerate(zip(seeds, maps)):
        planes = grid[0, j].subgridspec(1, 2, wspace=.16)
        for z in (0, 1):
            ax = fig.add_subplot(planes[z])
            map_axes.append(ax)
            im = ax.imshow(cube[z], cmap='RdBu_r', vmin=-limit, vmax=limit, origin='upper')
            ax.set_title(f'Seed {seed} · z={z}', fontsize=10)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel('x')
            if z == 0:
                ax.set_ylabel('y')
            for y, x in product(range(2), repeat=2):
                val = cube[z, y, x]
                ax.text(x, y, f'{val:+.2g}', ha='center', va='center', fontsize=10,
                        color='white' if abs(val) > .55 * limit else 'black')
    fig.colorbar(im, ax=map_axes, shrink=.8, pad=.015, label='Fitted weight')
    x = np.array([betas[i] for i in order])
    # Horizontal displacements distinguish points without affecting their measured ordinates.
    dx = .10 * np.min(np.diff(x))
    for j, b in enumerate(BUFFERS):
        ax = fig.add_subplot(grid[1, j])
        ax.set_title(f'Template overlaps · b={b}')
        for key, method, marker, offset in (('abs_uniform_cos', 'uniform', 'o', -dx),
                                             ('abs_checker_cos', 'staggered', 's', dx)):
            # Draw each fit, not a CI treating the three seeds as a statistical sample.
            rows = [r for r in overlaps if r['buffer'] == b]
            ax.scatter([r['beta'] + offset for r in rows], [r[key] for r in rows], marker=marker,
                       color=COLOR[method], s=15, alpha=.6, label=LABEL[method])
        if j == 0:
            ax.set_ylabel('|cosine| to template')
            ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(-.06, 1.06)
        pair_ax = fig.add_subplot(grid[2, j])
        pair_ax.set_title(f'Seed-pair stability · b={b}')
        rows = [r for r in stability if r['buffer'] == b]
        for key, label, color, marker, offset in (('abs_cos', 'Same orientation', '#555555', 'o', -dx),
                                                 ('max_abs_cos_48', 'Best of 48 cube symmetries', '#7741a2', '^', dx)):
            pair_ax.scatter([r['beta'] + offset for r in rows], [r[key] for r in rows], color=color,
                            marker=marker, alpha=.6, s=15, label=label)
        pair_ax.set(xlabel='Inverse temperature β', ylim=(-.06, 1.06))
        if j == 0:
            pair_ax.set_ylabel('Seed-pair |cosine|')
            pair_ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'Learned weight geometry  |  Signed 2×2×2 filters at coldest β={betas[cold]:g}, b={b_map}', fontsize=15)
    fig.text(.07, .070, 'Top: as fitted, shared scale, no sign flips or alignment. Middle: all three seeds. Bottom: all three seed pairs. β offsets for legibility.', fontsize=9)
    fig.text(.07, .046, 'Absolute cosine describes weight shape, not sign equivalence of the categorical channel. Maximizing over rotations and reflections can inflate similarity.', fontsize=9)
    fig.text(.07, .023, 'Overlaps are descriptive, not evidence for a phase or order parameter. Signed, unaligned overlaps are also tabulated.', fontsize=9)
    save(fig, root / 'filters.png')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path, help='directory containing fits/*.json; figures and CSVs are written here')
    root = p.parse_args().results
    fits, betas = load_fits(root)
    order = sorted(betas, key=betas.get)
    summary, contrasts = summarize(root, fits, betas, order)
    overlaps, stability = filter_tables(root, fits, betas, order)
    plot_signal(root, summary, betas, order)
    plot_comparators(root, summary, contrasts, betas, order)
    plot_filters(root, fits, betas, order, overlaps, stability)


if __name__ == '__main__':
    main()
