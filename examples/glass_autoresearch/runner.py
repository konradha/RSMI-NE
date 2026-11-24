"""Schedule finite compression experiments on released endpoints (Linux, local CPU)."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time


LANES = ('rg', 'information', 'learn')
HERE = Path(__file__).resolve().parent
MIN_ROUNDS = 2
CHECKPOINT = 30 * 60
MAX_JOB = 2 * 3600
GRACE = 5
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec='seconds')


@contextmanager
def replacement(path):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '-')
    try:
        with os.fdopen(fd, 'wb') as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def save(path, value):
    with replacement(path) as f:
        f.write((json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + '\n').encode())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_tree(x):
    if isinstance(x, float) and not math.isfinite(x):
        raise ValueError('non-finite JSON number')
    if isinstance(x, (list, dict)):
        for v in x.values() if isinstance(x, dict) else x:
            finite_tree(v)


def unique_keys(pairs):
    d = dict(pairs)
    if len(d) != len(pairs):
        raise ValueError('duplicate JSON keys')
    return d


def text(x):
    return isinstance(x, str) and bool(x.strip()) and x.strip().lower() not in {
        'todo', 'tbd', 'placeholder', 'stub', 'n/a', 'none', 'unknown', 'mock'}


def has_count(x):
    if isinstance(x, (list, dict)):
        return any(has_count(v) for v in (x.values() if isinstance(x, dict) else x))
    return type(x) is int and x > 0


def concrete(x):
    if isinstance(x, (dict, list)):
        return any(concrete(v) for v in (x.values() if isinstance(x, dict) else x))
    return text(x) or type(x) in (int, float)


def receipt(path, mode, n):
    d = json.loads(path.read_bytes(), object_pairs_hook=unique_keys)
    finite_tree(d)
    if not isinstance(d, dict) or (d.get('mode'), d.get('round')) != (mode, n) or type(d['round']) is not int:
        raise ValueError('receipt has wrong mode or round')
    if d.get('status') not in ('ok', 'inconclusive'):
        raise ValueError('invalid scientific status')
    for key in ('metric', 'hypothesis'):
        if not text(d.get(key)):
            raise ValueError(f'missing substantive {key}')
    for key in ('selection_basis', 'data', 'params', 'counts', 'provenance'):
        v = d.get(key)
        if not (isinstance(v, (str, dict, list)) and concrete(v)):
            raise ValueError(f'missing substantive {key}')
    if not has_count(d['counts']):
        raise ValueError('no positive observation/work count')
    x = d['estimate']
    if x is None:
        if d['status'] != 'inconclusive':
            raise ValueError('null estimate must be inconclusive')
        if not any(text(d.get(k)) for k in ('reason', 'null_reason', 'estimate_reason')):
            raise ValueError('null estimate requires explanation')
    elif type(x) not in (float, int) or not math.isfinite(x):
        raise ValueError('estimate must be a finite number or explained null')
    if type(d.get('search_exhausted', False)) is not bool:
        raise ValueError('search_exhausted must be boolean')
    return {k: d[k] for k in ('round', 'metric', 'estimate', 'status')} | {
        'search_exhausted': d.get('search_exhausted', False), 'sha256': digest(path)}


def receipt_path(root, mode, n):
    return root / ('confirmation.json' if mode == 'confirm' else f'round-{n:04d}.json')


def worker(attempt, lease_fd):
    """Independent of scheduler lifetime. GNU timeout also inherits the lease."""
    spec = json.loads((attempt / 'request.json').read_bytes())
    now = time.time()
    seconds = spec['stop_epoch'] - now - GRACE
    rc = 124
    try:
        if seconds > 0:
            cmd = [spec['timeout_bin'], '--kill-after=' + str(GRACE), f'{seconds:.3f}', *spec['argv']]
            rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, pass_fds=(lease_fd,), check=False).returncode
    except OSError as e:
        print(f'EXEC FAILURE: {e}', file=sys.stderr, flush=True)
        rc = 127
    save(attempt / 'exit.json', {'returncode': rc, 'end_epoch': time.time()})
    return rc


def lock(f):
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


class Campaign:
    def __init__(self, a):
        self.a = a
        self.meta = a.work / 'scheduler'
        self.meta.mkdir(exist_ok=True, parents=True)
        self.guard = (self.meta / 'runner.lock').open('a+b')
        if not lock(self.guard):
            raise ValueError('another scheduler owns this work directory')
        a.out.mkdir(exist_ok=True, parents=True)
        self.public_guard = os.open(a.out, os.O_RDONLY | os.O_DIRECTORY)
        if not lock(self.public_guard):
            raise ValueError('another scheduler owns this public results directory')
        self.statefile = self.meta / 'campaign.json'
        self.deadline = a.start_epoch + a.hours * 3600
        self.reserve = a.reserve_minutes * 60
        self.cut = self.deadline - self.reserve
        self.config = {'version': 1, 'data': str(a.data), 'release_manifest_sha256': digest(a.data / 'SHA256SUMS'),
                       'work': str(a.work), 'out': str(a.out), 'start_epoch': a.start_epoch,
                       'deadline_epoch': self.deadline, 'reserve_seconds': self.reserve,
                       'lanes': list(a.lanes), 'min_discoveries': MIN_ROUNDS}
        for lane in a.lanes:
            (a.work / lane).mkdir(exist_ok=True, parents=True)
            (a.out / lane).mkdir(exist_ok=True, parents=True)
            (self.meta / lane).mkdir(exist_ok=True)
        if self.statefile.exists():
            old = json.loads(self.statefile.read_bytes())
            if old['config'] != self.config:
                raise ValueError('campaign identity differs: use original paths, clock, reserve and lane set')
            self.lanes = old['lanes']
        else:
            if (a.out / 'status.json').exists():
                raise ValueError('public campaign already exists: restore original private work directory')
            self.lanes = {l: {'done': [], 'active': None, 'error': None, 'confirmation': None,
                              'freeze': None, 'closed': None, 'attempts': 0} for l in a.lanes}
            self.persist()
        self.children = {}
        self.stopping = False
        self.last_phase = None
        self.last_bucket = None
        self.dirty_phase = False

    def persist(self):
        save(self.statefile, {'config': self.config, 'lanes': self.lanes})

    def fail(self, lane, why, job=None):
        s = self.lanes[lane]
        s['error'] = {'reason': why, 'job': job, 'epoch': time.time()}
        if job:
            save(self.meta / lane / job['id'] / 'failure.json', s['error'])
        s['active'] = None
        self.persist()
        print(f'FAIL {lane}: {why}' + (f"; private logs {self.meta / lane / job['id']}" if job else ''), flush=True)
        self.dirty_phase = True

    def recover(self):
        for lane, s in self.lanes.items():
            root = self.a.out / lane
            try:
                expected = set()
                for mode, rows in [('discover', s['done']), ('confirm', [s['confirmation']] if s['confirmation'] else [])]:
                    for row in rows:
                        path = receipt_path(root, mode, row['round'])
                        if digest(path) != row['sha256']:
                            raise ValueError(f'accepted receipt changed: {path.name}')
                        expected.add(path)
                job = s['active'] or (s['error'] or {}).get('job')
                if job:
                    expected.add(receipt_path(root, job['mode'], job['round']))
                unknown = (set(root.glob('round-*.json')) | set(root.glob('confirmation.json'))) - expected
                if unknown:
                    raise ValueError(f'unowned receipt: {min(unknown).name}; restore matching private state')
                if lane in self.a.retry_discovery:
                    if time.time() >= self.cut:
                        raise ValueError('cannot reopen discovery after the selection cutoff')
                    if s['active'] or not s['error'] or not job or job['mode'] != 'discover' or s['freeze']:
                        raise ValueError('retry only applies to a quarantined, unfrozen discovery')
                    if receipt_path(root, 'discover', job['round']).exists():
                        raise ValueError('cannot retry with existing receipt; inspect original attempt and preserve evidence')
                    s['error'] = None
                    self.persist()
            except (OSError, ValueError) as e:
                if s['active']:
                    # Drain its bounded worker but never release results from a compromised prefix.
                    s['error'] = {'reason': str(e), 'epoch': time.time(), 'job': s['active']}
                    self.persist()
                    print(f'FAIL {lane}: {e}; draining active worker', flush=True)
                    continue
                self.fail(lane, str(e), (s['error'] or {}).get('job'))

    def launch(self, lane, mode, n, stop, why):
        s = self.lanes[lane]
        root = self.a.out / lane
        if receipt_path(root, mode, n).exists():
            self.fail(lane, 'refusing to overwrite existing scientific receipt')
            return
        lease = (self.meta / lane / 'active.lock').open('a+b')
        try:
            if not lock(lease):
                raise RuntimeError('untracked running worker; refusing overlapping writes')
            script = HERE / (lane + '.py')
            argv = [self.a.python, str(script), '--mode', mode, '--round', str(n), '--data', str(self.a.data),
                    '--work', str(self.a.work / lane), '--out', str(root)]
            # Allocate a new journal even after a crash before the intent was checkpointed.
            while True:
                s['attempts'] += 1
                ident = f"{s['attempts']:05d}-{mode}-{n:04d}"
                attempt = self.meta / lane / ident
                try:
                    attempt.mkdir()
                    break
                except FileExistsError:
                    continue
            start = time.time()
            job = {'id': ident, 'round': n, 'mode': mode, 'start_epoch': start, 'stop_epoch': stop}
            if mode == 'confirm':
                if s['freeze'] is not None:
                    raise RuntimeError('confirmation ticket already spent')
                s['freeze'] = {'last_discovery': n, 'epoch': start, 'reason': why,
                               'prefix_sha256': hashlib.sha256(''.join(r['sha256'] for r in s['done']).encode()).hexdigest()}
                self.dirty_phase = True
            s['active'] = job
            self.persist()  # write-ahead ticket: never repeat an ambiguous test peek
            try:
                save(attempt / 'request.json', dict(job, argv=argv, script_sha256=digest(script),
                     timeout_bin=self.a.timeout_bin, release_manifest_sha256=self.config['release_manifest_sha256'],
                     selection_freeze=s['freeze'], cwd=str(HERE.parent.parent)))
                env = os.environ | {k: '1' for k in THREADS} | {
                    'CUDA_VISIBLE_DEVICES': '', 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'}
                with (attempt / 'stdout.log').open('xb') as out, (attempt / 'stderr.log').open('xb') as err:
                    self.children[lane] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_worker',
                        str(attempt), str(lease.fileno())], cwd=HERE.parent.parent, stdin=subprocess.DEVNULL,
                        stdout=out, stderr=err, pass_fds=(lease.fileno(),), env=env, start_new_session=True)
                print(f'START {lane} {mode} {n} until {utc(stop)}', flush=True)
            except (OSError, ValueError) as e:
                self.fail(lane, f'launch: {e}', job)
        finally:
            lease.close()  # the worker/timeout now own the lease

    def collect(self, lane):
        s = self.lanes[lane]
        job = s['active']
        proc = self.children.get(lane)
        if proc is not None and proc.poll() is None:
            return
        with (self.meta / lane / 'active.lock').open('a+b') as lease:
            if not lock(lease):
                return  # adopted running worker, including orphaned GNU timeout
            self.children.pop(lane, None)
            attempt = self.meta / lane / job['id']
            try:
                if s['error']:
                    raise ValueError(s['error']['reason'])
                exitfile = attempt / 'exit.json'
                if exitfile.exists():
                    ex = json.loads(exitfile.read_bytes())
                    if ex['returncode'] != 0:
                        raise ValueError(f"exit {ex['returncode']} (124/137 usually timeout)")
                    end = ex['end_epoch']
                    if end > job['stop_epoch']:
                        raise ValueError('worker finished after its hard deadline')
                else:
                    # Even a parseable receipt is not proof of clean exit after host loss.
                    raise ValueError('no durable worker exit; inspect interruption; no automatic replay')
                row = receipt(receipt_path(self.a.out / lane, job['mode'], job['round']), job['mode'], job['round'])
                row['elapsed_seconds'] = round(end - job['start_epoch'], 3)
                if job['mode'] == 'confirm':
                    s['confirmation'] = row
                    self.dirty_phase = True
                else:
                    s['done'].append(row)
                    if len(s['done']) == 1:
                        self.dirty_phase = True
                s['active'] = None
                self.persist()
                print(f"DONE {lane} {job['mode']} {job['round']} {row['status']} {row['estimate']}", flush=True)
            except (OSError, KeyError, ValueError, OverflowError, TypeError) as e:
                self.fail(lane, str(e), job)

    def next_job(self, lane, now):
        s = self.lanes[lane]
        if s['active'] or s['error'] or s['closed'] or s['confirmation']:
            return None
        rows = s['done']
        n = len(rows)
        estimate = 1.6 * max((r['elapsed_seconds'] for r in rows[-3:]), default=0) + 15
        exhausted = bool(rows and rows[-1]['search_exhausted'])
        if now >= self.cut - GRACE - 1 or exhausted or n >= MIN_ROUNDS and now + estimate >= self.cut:
            if n < MIN_ROUNDS:
                s['closed'] = f'not confirmed: {n} completed discoveries (minimum {MIN_ROUNDS})'
                self.persist()
                self.dirty_phase = True
                return None
            why = ('search budget exhausted' if exhausted else 'confirmation reserve' if now >= self.cut
                   else 'predicted next unit would overrun discovery allocation')
            return (0, n, lane, 'confirm', n - 1, min(self.deadline, now + MAX_JOB), why)
        return (1, n, lane, 'discover', n, min(now + MAX_JOB, self.cut), 'validation search')

    def dashboard(self, phase, now):
        rows = []
        views = {}
        for lane, s in self.lanes.items():
            ds = s['done']
            last = ds[-1] if ds else {}
            co = s['confirmation'] or {}
            state = ('failed' if s['error'] else 'confirmed' if co else 'not confirmed' if s['closed'] else
                     s['active']['mode'] if s['active'] else 'idle')
            views[lane] = {'state': state, 'completed_discoveries': len(ds), 'last_discovery': last or None,
                           'active': s['active'], 'freeze': s['freeze'], 'confirmation': co or None,
                           'failure': s['error'], 'closure': s['closed']}
            rows.append({'lane': lane, 'state': state, 'discoveries': len(ds), 'latest_val_metric': last.get('metric'),
                         'latest_val_estimate': last.get('estimate'), 'confirmation_metric': co.get('metric'),
                         'confirmation_estimate': co.get('estimate'), 'scientific_status': co.get('status')})
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        with replacement(self.a.out / 'leaderboard.csv') as f:
            f.write(buf.getvalue().encode())
        from matplotlib import use
        use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(rows), 1, figsize=(8, 2.45 * len(rows) + 1), squeeze=False, layout='constrained')
        for ax, (lane, s) in zip(axes[:, 0], self.lanes.items()):
            for metric in sorted({r['metric'] for r in s['done']}):
                pts = [r for r in s['done'] if r['metric'] == metric and r['estimate'] is not None]
                if pts:
                    ax.plot([r['round'] for r in pts], [r['estimate'] for r in pts], '.-', lw=1,
                            label='validation: ' + metric[:70])
            co = s['confirmation']
            if co and co['estimate'] is not None:
                ax.scatter(co['round'], co['estimate'], c='tab:red', marker='D', label='frozen test: ' + co['metric'][:70])
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc='best', fontsize=7)
            else:
                ax.text(.5, .5, 'No finite estimates reported', ha='center', transform=ax.transAxes)
            ax.set_title(f"{lane} — {views[lane]['state']} ({len(s['done'])} completed units)", loc='left', fontsize=10)
            ax.set_ylabel('lane estimate')
            ax.grid(alpha=.2)
        axes[-1, 0].set_xlabel('Discovery round (test marker at frozen prefix end)')
        fig.suptitle(f'Frozen-parent compression search | {phase}\n{utc(now)}', fontsize=12)
        try:
            with replacement(self.a.out / 'status.png') as f:
                fig.savefig(f, format='png', dpi=125)
        finally:
            plt.close(fig)
        save(self.a.out / 'status.json', {'phase': phase, 'as_of_utc': utc(now), 'start_utc': utc(self.a.start_epoch),
             'deadline_utc': utc(self.deadline), 'discovery_cutoff_utc': utc(self.cut), 'min_discoveries': MIN_ROUNDS,
             'release_manifest_sha256': self.config['release_manifest_sha256'], 'lanes': views})
        print(f'PUBLISH {utc(now)} {phase}: {self.a.out} (status.json, leaderboard.csv, status.png and lane artifacts)', flush=True)
        self.last_phase = phase
        self.last_bucket = math.floor((now - self.a.start_epoch) / CHECKPOINT)
        self.dirty_phase = False

    def run(self):
        self.recover()
        def stop(signum, frame):
            self.stopping = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        while True:
            now = time.time()
            for lane, s in self.lanes.items():
                if s['active']:
                    self.collect(lane)
            active = sum(s['active'] is not None for s in self.lanes.values())
            terminal = False
            if self.stopping:
                phase, terminal = 'scheduler detached; bounded workers may remain', True
            elif now < self.a.start_epoch:
                phase = 'awaiting scheduled start'
            elif now >= self.deadline - GRACE - 1:
                # No child is admitted once it cannot fit a term/kill grace period.
                phase = 'deadline; draining' if active else 'deadline reached'
                terminal = active == 0 or now > self.deadline + GRACE
                if terminal:
                    for s in self.lanes.values():
                        if not (s['active'] or s['error'] or s['closed'] or s['confirmation']):
                            s['closed'] = 'deadline; not confirmed'
                    self.persist()
            else:
                candidates = sorted(j for lane in self.lanes if (j := self.next_job(lane, now)) is not None)
                for _, _, lane, mode, n, stop_epoch, why in candidates:
                    if active >= self.a.max_workers:
                        break
                    self.launch(lane, mode, n, stop_epoch, why)
                    active += self.lanes[lane]['active'] is not None
                if not active:
                    phase, terminal = 'finished; no schedulable work', True
                elif any(s['active'] and s['active']['mode'] == 'confirm' for s in self.lanes.values()):
                    phase = 'confirmation / draining discovery'
                else:
                    phase = 'discovery' if now < self.cut else 'draining discovery'
            if self.dirty_phase or phase != self.last_phase or math.floor((now - self.a.start_epoch) / CHECKPOINT) != self.last_bucket:
                self.dashboard(phase, now)
            if terminal:
                return int(any(s['error'] for s in self.lanes.values()) or (not self.stopping and active > 0))
            time.sleep(min(2, max(.1, self.deadline - time.time())))


def main():
    if sys.argv[1:2] == ['_worker']:
        return worker(Path(sys.argv[2]), int(sys.argv[3]))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start-epoch', type=int, required=True)
    p.add_argument('--hours', type=float, default=10)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lanes', nargs='+', default=list(LANES), help='space- or comma-separated subset of rg,information,learn')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--reserve-minutes', type=float, default=45)
    p.add_argument('--max-workers', type=int, default=5)
    p.add_argument('--retry-discovery', action='append', choices=LANES, default=[], help='explicitly re-arm a failed discovery only')
    a = p.parse_args()
    lanes = [x for arg in a.lanes for x in re.split(',', arg)]
    if not lanes or set(lanes) - set(LANES) or len(lanes) != len(set(lanes)):
        p.error('lanes must be distinct members of rg,information,learn')
    a.lanes = [x for x in LANES if x in lanes]
    if set(a.retry_discovery) - set(a.lanes):
        p.error('retry lane must be enabled')
    if not (math.isfinite(a.hours) and 0 < a.reserve_minutes < 60 * a.hours and a.max_workers > 0):
        p.error('require finite positive hours, 0 < reserve < duration, and positive max-workers')
    a.data, a.work, a.out = [x.expanduser().resolve() for x in (a.data, a.work, a.out)]
    for x, y in ((a.work, a.out), (a.data, a.work), (a.data, a.out)):
        if x == y or x in y.parents or y in x.parents:
            p.error('data, work and out must be separate non-nested directories')
    a.python = shutil.which(os.path.expanduser(a.python))
    a.timeout_bin = shutil.which('timeout')
    if not a.python or not a.timeout_bin:
        p.error('Python executable and GNU timeout required')
    a.python = os.path.abspath(a.python)  # preserve virtualenv symlink
    return Campaign(a).run()


if __name__ == '__main__':
    sys.exit(main())
