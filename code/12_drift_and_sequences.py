# Step-cap study of the projection rule on the trained reference network, and sequences of several retained tasks.
# Run from inside code/ after 03_train_task_A.py:  python3 12_drift_and_sequences.py
# Writes E12_drift_and_sequences.json.
import numpy as np, json
from network_tools import *

out = {}

# ---------------- (a) step cap, path length and accumulated drift on the reference network ----------------
N, E = grid(5); net = Net(N, E)
gA = np.load('gA.npy')
cA, fA = net.e(4), net.e(24); tA = net.y(gA, cA, fA)            # task A is fitted exactly at gA
cB, fB = net.e(4), net.e(12); target = 1.03 * net.y(gA, cB, fB)
LA = lambda g: 0.5 * float((net.y(g, cA, fA) - tA)[0] ** 2)
rows = []
for tau in (0.1, 0.05, 0.025, 0.0125):
    g, info = project(net, gA, [(cA, fA, tA)], cB, fB, target, tau=tau)
    gc, infoc = project(net, gA, [(cA, fA, tA)], cB, fB, target, tau=tau, correct=True)
    rows.append(dict(tau=tau, iters=info['iters'], path=info['path'], status=info['status'],
                     new_loss_ratio=info['loss_ratio'], F_A=LA(g), F_over_tau2_path2=LA(g) / (tau * info['path']) ** 2,
                     clipped=info['clipped'], F_A_corrected=LA(gc), iters_corrected=infoc['iters'],
                     new_loss_ratio_corrected=infoc['loss_ratio']))
    print(rows[-1])
out['step_cap'] = rows

# ---------------- (b) sequences of scalar tasks on the 5x5 grid ----------------
# Each target is (1 +/- DELTA) times the response of the initial network, with a random sign.
KMAX, NSEQ, DELTA = 6, 8, 0.03
def retained_loss(g, tasks): return sum(0.5 * float((net.y(g, c, f) - t)[0] ** 2) for c, f, t in tasks)
seqs = []
for seed in range(NSEQ):
    rng = np.random.default_rng(2000 + seed)
    g0 = rng.uniform(0.75, 1.25, net.M)
    pairs = []
    while len(pairs) < KMAX:
        a, b = (int(v) for v in rng.choice(net.free, 2, replace=False))
        if (a, b) not in pairs and (b, a) not in pairs: pairs.append((a, b))     # no repeats, no reciprocal duplicates
    # the task list, targets included, is fixed here and is the same for the three methods
    signs = rng.choice([-1.0, 1.0], KMAX)
    tasks = [(net.e(b), net.e(a), (1 + sg * DELTA) * net.y(g0, net.e(b), net.e(a))) for (a, b), sg in zip(pairs, signs)]
    state = {m: dict(g=g0.copy(), tasks=[]) for m in ('plain', 'projection', 'projection+correction')}
    rec = {m: [] for m in state}
    for k, (c, f, tgt) in enumerate(tasks, start=1):
        for m, st in state.items():
            g = st['g']; info = None; y0 = net.y(g, c, f); r0 = float((y0 - tgt)[0])
            if k == 1 or m == 'plain':
                gn = g.copy()
                for s in np.linspace(0.2, 1.0, 5): gn = train(net, gn, c, f, y0 + s * (tgt - y0))
                rf = float((net.y(gn, c, f) - tgt)[0]); fit = (rf / r0) ** 2
                status = 'converged' if fitted(rf, r0, tgt[0]) else 'max_iter'
                gain0 = float('nan'); clipped = int(np.sum((gn <= GMIN + 1e-9) | (gn >= GMAX - 1e-9)))
            else:
                gn, info = project(net, g, st['tasks'], c, f, tgt, correct=(m == 'projection+correction'))
                fit, status, gain0, clipped = info['loss_ratio'], info['status'], info['gain0'], info['clipped']
            before = retained_loss(gn, st['tasks'])
            st['g'] = gn; st['tasks'] = st['tasks'] + [(c, f, tgt)]
            rec[m].append(dict(K=k, retained_loss=before, new_fit=fit, status=status, gain0=gain0, clipped=clipped,
                               requested=abs(r0) / abs(float(y0[0])), iters=info['iters'] if info else None,
                               moved=float(np.linalg.norm(gn - g0))))
    seqs.append(dict(seed=seed, pairs=pairs, signs=[float(v) for v in signs], targets=[float(t[0]) for _, _, t in tasks], records=rec))
out['sequences'] = seqs
METHODS = ('plain', 'projection', 'projection+correction')
# a sequence is regular if every method fits every task and no conductance reaches a bound
def regular(s): return all(r['status'] == 'converged' and r['clipped'] == 0 for m in METHODS for r in s['records'][m])
reg = [s for s in seqs if regular(s)]
out['regular_seeds'] = [s['seed'] for s in reg]
summary = {}
for m in METHODS:
    summary[m] = []
    for k in range(2, KMAX + 1):
        v = np.array([s['records'][m][k - 1]['retained_loss'] for s in reg])
        rq = [s['records'][m][k - 1]['requested'] for s in reg]
        row = dict(K=k, retained_loss_median=float(np.median(v)), retained_loss_max=float(v.max()),
                   requested_range=[float(min(rq)), float(max(rq))])
        if m != 'plain':
            row['min_gain0'] = float(min(s['records'][m][k - 1]['gain0'] for s in reg))
            row['iters_median'] = float(np.median([s['records'][m][k - 1]['iters'] for s in reg]))
        summary[m].append(row); print(m, row)
out['sequence_summary'] = summary
print('regular sequences:', out['regular_seeds'])
json.dump(out, open('E12_drift_and_sequences.json', 'w'), indent=1, default=float)
