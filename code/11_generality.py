# Tests across graph families, sizes and random seeds, and the restricted-access rank.
# Run from inside code/:  python3 11_generality.py
# Writes E11_generality.json and E11_pairs.npy (one row per task pair; columns listed in COLS).
import numpy as np, json, itertools
import scipy.stats as st
from network_tools import *


STATUS = ['converged', 'no_gain', 'stalled', 'max_iter']


def study(name, build, seeds, n_pairs=6):
    rows = []; reach = []; rankok = []; cond = []; Ms = []
    for seed in seeds:
        N, E = build(seed)
        net = Net(N, E); rng = np.random.default_rng(1000 + seed); Ms.append(net.M)
        g0 = rng.uniform(0.75, 1.25, net.M)
        # attainable rank, and rank growth for scalar tasks with Gaussian forcing and observation vectors
        d = net.rank_T(); assert d == net.rank_formula()
        Li = np.linalg.inv(net.Lap(g0)); P = net.Bf.T @ Li
        R = np.array([-(P @ rng.normal(size=net.Nf)) * (P @ rng.normal(size=net.Nf)) for _ in range(d)])
        sv = np.linalg.svd(R, compute_uv=False)
        rankok.append(all(nrank(R[:k]) == k for k in (1, d // 2, d - 1, d)))
        cond.append(float(sv[-1] / sv[0]))
        # task A: one source node, one observed node
        a_in, a_out = rng.choice(net.free, 2, replace=False)
        fA, cA = net.e(a_in), net.e(a_out)
        tA = 1.02 * net.y(g0, cA, fA); gA = train(net, g0, cA, fA, tA)
        JA = net.J(gA, cA, fA)[0]; nJA2 = float(JA @ JA)
        LA = lambda g: 0.5 * float((net.y(g, cA, fA) - tA)[0] ** 2)
        for k in range(n_pairs):
            while True:
                b_in, b_out = rng.choice(net.free, 2, replace=False)
                if (b_in, b_out) != (a_in, a_out): break
            recip = int((b_in, b_out) == (a_out, a_in))      # same response as task A, by reciprocity
            fB, cB = net.e(b_in), net.e(b_out)
            JB = net.J(gA, cB, fB)[0]
            Gam = float((JA @ JB) ** 2 / (nJA2 * (JB @ JB)))
            base = net.y(gA, cB, fB); target = base * 1.03
            LB = lambda g: float((net.y(g, cB, fB) - target)[0] ** 2 / (base[0] * 0.03) ** 2)
            # plain training by continuation up to a 3% correction
            g = gA.copy()
            for s in np.linspace(0.2, 1.0, 5): g = train(net, g, cB, fB, base * (1 + 0.03 * s))
            dg = g - gA; Fp = LA(g) - LA(gA); nd = float(np.linalg.norm(dg))
            err = abs(0.5 * float(JA @ dg) ** 2 / Fp - 1.0) if Fp > 0 else np.inf
            gp, info = project(net, gA, [(cA, fA, tA)], cB, fB, target)
            Fq = LA(gp) - LA(gA)
            rows.append((Gam, Fp / (0.5 * nd ** 2), nJA2 * Gam, nd, Fp, Fq, float(np.linalg.norm(gp - gA)),
                         int(np.sum((g <= GMIN + 1e-9) | (g >= GMAX - 1e-9))), info['clipped'],
                         LB(g), info['loss_ratio'], STATUS.index(info['status']), info['iters'], info['gain0'],
                         err, recip, net.M, seed, k))
            if k == 0:
                # reach of the quadratic law: continuation in amplitude up to a 30% correction
                g = gA.copy(); prev = None; hit = None
                for amp in 0.03 * np.logspace(-2, 1, 25):
                    g = train(net, g, cB, fB, base * (1 + amp))
                    dgc = g - gA; Fm = LA(g) - LA(gA); Fpred = 0.5 * float(JA @ dgc) ** 2
                    if Fm <= 0: continue
                    er = abs(Fpred / Fm - 1.0); rot = angle(JA, net.J(g, cA, fA)[0]); nrm = float(np.linalg.norm(dgc))
                    if er > 0.10 and hit is None:
                        if prev is None: hit = (rot, nrm)
                        else:
                            t = (0.10 - prev[0]) / (er - prev[0])
                            hit = (prev[1] + t * (rot - prev[1]), prev[2] + t * (nrm - prev[2]))
                    prev = (er, rot, nrm)
                reach.append(hit if hit is not None else (float('nan'), float('nan')))
    a = np.array(rows, dtype=float); reach = np.array(reach)
    conv = a[:, 11] == 0                                  # projection fitted the new task
    local = (a[:, 14] <= 0.10) & (a[:, 7] == 0)           # quadratic law within 10% for the plain update, no bound active
    q = lambda v: [float(np.nanpercentile(v, p)) for p in (25, 50, 75)]
    gain = a[conv, 4] / a[conv, 5]
    res = dict(family=name, N=N, M=[int(min(Ms)), int(max(Ms))], d_equals_M=True, n_networks=len(seeds), n_pairs=len(a),
               rank_growth_ok=bool(all(rankok)), sigma_ratio_at_saturation=q(np.array(cond)),
               rotation_at_10pct_deg=q(reach[:, 0]), reach_found=int(np.sum(~np.isnan(reach[:, 0]))),
               plain_max_loss_ratio=float(a[:, 9].max()),
               projection_converged=int(conv.sum()),
               projection_status={STATUS[i]: int(np.sum(a[:, 11] == i)) for i in range(4)},
               projection_max_loss_ratio_converged=float(a[conv, 10].max()),
               projection_iters=q(a[conv, 12]),
               gain_converged=q(gain), projection_better_converged=int(np.sum(a[conv, 5] < a[conv, 4])),
               min_gain0=float(a[:, 13].min()),
               spearman_local=float(st.spearmanr(a[local, 0], a[local, 1]).statistic), n_local=int(local.sum()),
               median_ratio_to_first_order=float(np.median(a[local, 1] / a[local, 2])),
               bounds_active_plain=int(np.sum(a[:, 7] > 0)), bounds_active_projection=int(np.sum(a[:, 8] > 0)))
    return res, np.c_[a, np.full(len(a), FAMILIES.index(name))]


FAMILIES = ['square grid 4x4', 'square grid 5x5', 'square grid 6x6', 'square grid 8x8',
            'triangular lattice 5x5', 'small-world N=30']
BUILD = {'square grid 4x4': lambda s: grid(4), 'square grid 5x5': lambda s: grid(5),
         'square grid 6x6': lambda s: grid(6), 'square grid 8x8': lambda s: grid(8),
         'triangular lattice 5x5': lambda s: triangular(5),
         'small-world N=30': lambda s: small_world(30, 4, 0.2, s)}

COLS = ['Gamma', 'F_plain/(0.5|dg|^2)', '|J_A|^2 Gamma', '|dg|_plain', 'F_plain', 'F_projected', '|dg|_projected',
        'bounds_plain', 'clipped_projection', 'plain_loss_ratio', 'projection_loss_ratio', 'status', 'iterations',
        'initial_projected_gain', 'quadratic_law_error', 'reciprocal_pair', 'M', 'seed', 'pair', 'family']
out = {'families': [], 'columns': COLS, 'status_codes': STATUS, 'rank_tolerance': RTOL, 'gain_threshold': ETA}; allpairs = []
for name in FAMILIES:
    res, a = study(name, BUILD[name], seeds=range(8))
    out['families'].append(res); allpairs.append(a)
    print(json.dumps(res))
np.save('E11_pairs.npy', np.vstack(allpairs))

# ---------------- restricted access on the 5x5 grid ----------------
N, E = grid(5); net = Net(N, E)
boundary = [v for v in net.free if v // 5 in (0, 4) or v % 5 in (0, 4)]
cases = [('all free nodes', net.free, net.free),
         ('boundary nodes', boundary, boundary),
         ('three corners', [4, 20, 24], [4, 20, 24]),
         ('one source, all observed', [24], net.free),
         ('S={20,24}, O={4,6,12}', [20, 24], [4, 6, 12]),
         ('S={24}, O={4}', [24], [4])]
ra = []
for label, S, O in cases:
    ranks = []
    for seed in range(5):
        g = np.random.default_rng(50 + seed).uniform(0.75, 1.25, net.M)
        ranks.append(net.rank_restricted(g, S, O))
    # rank growth of random tasks with forcings on S and observations on O
    g = np.random.default_rng(50).uniform(0.75, 1.25, net.M); rng = np.random.default_rng(7)
    Li = np.linalg.inv(net.Lap(g)); P = net.Bf.T @ Li; rows = []
    for _ in range(ranks[0] + 5):
        f = sum(rng.normal() * net.e(v) for v in S); c = sum(rng.normal() * net.e(v) for v in O)
        rows.append(-(P @ f) * (P @ c))
    sv = np.linalg.svd(np.array(rows), compute_uv=False)
    same = set(S) == set(O)
    bound = len(S) * (len(S) + 1) // 2 if same else len(S) * len(O)
    ra.append(dict(case=label, nS=len(S), nO=len(O), bound=min(bound, net.M), ranks=ranks,
                   random_task_rank=int((sv > RTOL * sv[0]).sum())))
    print(ra[-1])
out['restricted_access'] = ra

# ---------------- full observation: shared-row bound ----------------
def full_obs_ranks(net, g, Kmax, seed):
    rng = np.random.default_rng(seed); P = net.Bf.T @ np.linalg.inv(net.Lap(g)); rows = []; rk = []
    for _ in range(Kmax):
        f = rng.normal(size=net.Nf)
        for c in np.eye(net.Nf): rows.append(-(P @ f) * (P @ c))
        rk.append(nrank(np.array(rows)))
    return rk
fo = []
for label, (Nn, Ee) in [('complete graph K6', (6, list(itertools.combinations(range(6), 2)))),
                        ('complete graph K8', (8, list(itertools.combinations(range(8), 2)))),
                        ('square grid 4x4', grid(4))]:
    nt = Net(Nn, Ee); g = np.random.default_rng(3).uniform(0.75, 1.25, nt.M); d = nt.rank_T()
    Kmax = min(nt.Nf, 6)
    fo.append(dict(graph=label, Nf=nt.Nf, d=d, ranks=full_obs_ranks(nt, g, Kmax, 11),
                   bound=[min(K * nt.Nf - K * (K - 1) // 2, d) for K in range(1, Kmax + 1)]))
    print(fo[-1])
out['full_observation'] = fo
json.dump(out, open('E11_generality.json', 'w'), indent=1)
