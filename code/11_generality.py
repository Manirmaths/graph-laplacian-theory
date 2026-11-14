# Tests across graph families, sizes and random seeds, and the restricted-access rank.
# Run from inside code/:  python3 11_generality.py
# Writes E11_generality.json and E11_pairs.npy.
import numpy as np, json, itertools
from scipy.optimize import least_squares
import scipy.stats as st

GMIN, GMAX = 0.35, 2.50
LS = dict(bounds=(GMIN, GMAX), xtol=1e-15, ftol=1e-15, gtol=1e-15)


# ---------------- graphs ----------------
def grid(n, m=None):
    m = m or n
    idx = lambda r, c: r * m + c
    E = []
    for r in range(n):
        for c in range(m):
            if c + 1 < m: E.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < n: E.append((idx(r, c), idx(r + 1, c)))
    return n * m, E


def triangular(n):
    N, E = grid(n)
    idx = lambda r, c: r * n + c
    E += [(idx(r, c), idx(r + 1, c + 1)) for r in range(n - 1) for c in range(n - 1)]
    return N, E


def small_world(N, k, p, seed):
    # ring lattice with k neighbours, each edge rewired with probability p; kept simple and connected
    rng = np.random.default_rng(seed)
    while True:
        S = set()
        for i in range(N):
            for d in range(1, k // 2 + 1):
                j = (i + d) % N
                if rng.random() < p:
                    j = int(rng.integers(N))
                    while j == i or (min(i, j), max(i, j)) in S: j = int(rng.integers(N))
                S.add((min(i, j), max(i, j)))
        E = sorted(S)
        seen = {0}; stack = [0]
        adj = {i: [] for i in range(N)}
        for a, b in E: adj[a].append(b); adj[b].append(a)
        while stack:
            for w in adj[stack.pop()]:
                if w not in seen: seen.add(w); stack.append(w)
        if len(seen) == N: return N, E


class Net:
    def __init__(self, N, edges, ground=(0,)):
        self.N, self.edges, self.M = N, edges, len(edges)
        B = np.zeros((N, self.M))
        for k, (i, j) in enumerate(edges): B[i, k] = 1.0; B[j, k] = -1.0
        self.ground = list(ground)
        self.free = [i for i in range(N) if i not in self.ground]
        self.Nf = len(self.free); self.Bf = B[self.free, :]
        self.pos = {v: i for i, v in enumerate(self.free)}

    def Lap(self, g): return (self.Bf * g) @ self.Bf.T
    def e(self, node):
        v = np.zeros(self.Nf); v[self.pos[node]] = 1.0; return v
    def y(self, g, c, f): return np.array([c @ np.linalg.solve(self.Lap(g), f)])
    def J(self, g, c, f):
        L = self.Lap(g); u = np.linalg.solve(L, f); w = np.linalg.solve(L, c)
        return (-(self.Bf.T @ u) * (self.Bf.T @ w))[None, :]

    def rank_T(self):
        cols = [np.outer(b, b).ravel() for b in self.Bf.T]
        return int(np.linalg.matrix_rank(np.array(cols).T, tol=1e-9))

    def rank_formula(self):
        G = set(self.ground)
        m0 = sum(1 for a, b in self.edges if a in G and b in G)
        deg0 = {v: 0 for v in self.free}
        for a, b in self.edges:
            if (a in G) != (b in G): deg0[b if a in G else a] += 1
        return self.M - m0 - sum(max(k - 1, 0) for k in deg0.values())

    def rank_restricted(self, g, S, O):
        # rank of h -> E_S^T L^{-1} T(h) L^{-1} E_O
        Li = np.linalg.inv(self.Lap(g))
        PS = Li[[self.pos[v] for v in S], :] @ self.Bf       # |S| x M
        PO = Li[[self.pos[v] for v in O], :] @ self.Bf       # |O| x M
        A = np.einsum('se,oe->soe', PS, PO).reshape(-1, self.M)
        sv = np.linalg.svd(A, compute_uv=False)
        return int((sv > 1e-9 * sv[0]).sum())


def angle(a, b):
    c = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, c))))


def train(net, g_start, c, f, target):
    return least_squares(lambda q: net.y(q, c, f) - target, g_start, jac=lambda q: net.J(q, c, f), **LS).x


def project(net, gA, cA, fA, cB, fB, target, tau=0.05, iters=400):
    g = gA.copy(); clipped = 0
    for _ in range(iters):
        JA = net.J(g, cA, fA); Pi = np.eye(net.M) - np.linalg.pinv(JA) @ JA
        r = net.y(g, cB, fB) - target
        if abs(r[0]) < 1e-14: break
        s = -Pi @ np.linalg.pinv(net.J(g, cB, fB) @ Pi) @ r
        nrm = np.linalg.norm(s)
        if nrm > tau: s *= tau / nrm
        clipped = max(clipped, int(np.sum((g + s < GMIN) | (g + s > GMAX))))
        g = np.clip(g + s, GMIN, GMAX)
    return g, clipped


def study(name, build, seeds, n_pairs=6):
    rows = []; reach = []; rankok = []; cond = []
    for seed in seeds:
        N, E = build(seed)
        net = Net(N, E); rng = np.random.default_rng(1000 + seed)
        g0 = rng.uniform(0.75, 1.25, net.M)
        # attainable rank and rank growth for random scalar tasks
        d = net.rank_T(); assert d == net.rank_formula()
        Li = np.linalg.inv(net.Lap(g0)); P = net.Bf.T @ Li
        R = np.array([-(P @ rng.normal(size=net.Nf)) * (P @ rng.normal(size=net.Nf)) for _ in range(d)])
        sv = np.linalg.svd(R, compute_uv=False)
        rankok.append(bool(np.all([np.linalg.matrix_rank(R[:k], tol=1e-11) == k for k in (1, d // 2, d - 1, d)])))
        cond.append(float(sv[-1] / sv[0]))
        # task A
        a_in, a_out = rng.choice(net.free, 2, replace=False)
        fA, cA = net.e(a_in), net.e(a_out)
        tA = 1.02 * net.y(g0, cA, fA); gA = train(net, g0, cA, fA, tA)
        JA = net.J(gA, cA, fA)[0]; nJA2 = float(JA @ JA)
        LA = lambda g: 0.5 * float((net.y(g, cA, fA) - tA)[0] ** 2)
        for k in range(n_pairs):
            while True:
                b_in, b_out = rng.choice(net.free, 2, replace=False)
                if (b_in, b_out) != (a_in, a_out): break
            fB, cB = net.e(b_in), net.e(b_out)
            JB = net.J(gA, cB, fB)[0]
            Gam = float((JA @ JB) ** 2 / (nJA2 * (JB @ JB)))
            base = net.y(gA, cB, fB)
            # plain training by continuation up to a 3% correction
            g = gA.copy()
            for s in np.linspace(0.2, 1.0, 5): g = train(net, g, cB, fB, base * (1 + 0.03 * s))
            dg = g - gA; Fp = LA(g) - LA(gA); nd = float(np.linalg.norm(dg))
            gp, clipped = project(net, gA, cA, fA, cB, fB, base * 1.03)
            Fq = LA(gp) - LA(gA)
            rows.append((Gam, Fp / (0.5 * nd ** 2), nJA2 * Gam, nd, Fp, Fq, float(np.linalg.norm(gp - gA)),
                         int(np.sum((g <= GMIN + 1e-9) | (g >= GMAX - 1e-9))), clipped))
            if k == 0:
                # reach of the quadratic law: continuation in amplitude up to a 30% correction
                g = gA.copy(); prev = None; hit = None
                for amp in 0.03 * np.logspace(-2, 1, 25):
                    g = train(net, g, cB, fB, base * (1 + amp))
                    dgc = g - gA; Fm = LA(g) - LA(gA); Fpred = 0.5 * float(JA @ dgc) ** 2
                    if Fm <= 0: continue
                    err = abs(Fpred / Fm - 1.0); rot = angle(JA, net.J(g, cA, fA)[0]); nrm = float(np.linalg.norm(dgc))
                    if err > 0.10 and hit is None:
                        if prev is None: hit = (rot, nrm)
                        else:
                            t = (0.10 - prev[0]) / (err - prev[0])
                            hit = (prev[1] + t * (rot - prev[1]), prev[2] + t * (nrm - prev[2]))
                    prev = (err, rot, nrm)
                reach.append(hit if hit is not None else (float('nan'), float('nan')))
    a = np.array(rows); reach = np.array(reach)
    ok = (a[:, 3] <= 0.105) & (a[:, 7] == 0)
    q = lambda v: [float(np.nanpercentile(v, p)) for p in (25, 50, 75)]
    gain = a[:, 4] / a[:, 5]
    res = dict(family=name, N=N, M=net.M, d=d, n_networks=len(seeds), n_pairs=len(a),
               rank_growth_ok=bool(all(rankok)), sigma_ratio_at_saturation=q(np.array(cond)),
               rotation_at_10pct_deg=q(reach[:, 0]), norm_at_10pct=q(reach[:, 1]),
               reach_found=int(np.sum(~np.isnan(reach[:, 0]))),
               F_plain=q(a[:, 4]), F_projected=q(a[:, 5]), gain=q(gain),
               frac_projection_better=float(np.mean(a[:, 5] < a[:, 4])),
               norm_plain=q(a[:, 3]), norm_projected=q(a[:, 6]),
               spearman_all=float(st.spearmanr(a[:, 0], a[:, 1]).statistic),
               spearman_local=float(st.spearmanr(a[ok, 0], a[ok, 1]).statistic), n_local=int(ok.sum()),
               median_ratio_to_first_order=float(np.median(a[ok, 1] / a[ok, 2])),
               bounds_active_plain=int(np.sum(a[:, 7] > 0)), bounds_active_projection=int(np.sum(a[:, 8] > 0)))
    return res, np.c_[a, np.full(len(a), FAMILIES.index(name))]


FAMILIES = ['square grid 4x4', 'square grid 5x5', 'square grid 6x6', 'square grid 8x8',
            'triangular lattice 5x5', 'small-world N=30']
BUILD = {'square grid 4x4': lambda s: grid(4), 'square grid 5x5': lambda s: grid(5),
         'square grid 6x6': lambda s: grid(6), 'square grid 8x8': lambda s: grid(8),
         'triangular lattice 5x5': lambda s: triangular(5),
         'small-world N=30': lambda s: small_world(30, 4, 0.2, s)}

out = {'families': []}; allpairs = []
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
                   random_task_rank=int((sv > 1e-9 * sv[0]).sum())))
    print(ra[-1])
out['restricted_access'] = ra

# ---------------- full observation: shared-row bound ----------------
def full_obs_ranks(net, g, Kmax, seed):
    rng = np.random.default_rng(seed); P = net.Bf.T @ np.linalg.inv(net.Lap(g)); rows = []; rk = []
    for _ in range(Kmax):
        f = rng.normal(size=net.Nf)
        for c in np.eye(net.Nf): rows.append(-(P @ f) * (P @ c))
        sv = np.linalg.svd(np.array(rows), compute_uv=False); rk.append(int((sv > 1e-9 * sv[0]).sum()))
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
