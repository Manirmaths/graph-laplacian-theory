# Shared tools for 11_generality.py and 12_drift_and_sequences.py:
# graph generators, the reduced-Laplacian network class, training, and the safeguarded projection rule.
import numpy as np
from scipy.optimize import least_squares

GMIN, GMAX = 0.35, 2.50
LS = dict(bounds=(GMIN, GMAX), xtol=1e-15, ftol=1e-15, gtol=1e-15)
RTOL = 1e-9          # relative singular-value tolerance used for every numerical rank
ETA = 1e-2           # projected-gain threshold, relative to ||J_B||
CONV = 1e-12         # a residual r of a task with target t counts as fitted when |r| <= CONV |r_start| + ATOL |t|
ATOL = 1e-13


def nrank(A):
    sv = np.linalg.svd(np.atleast_2d(A), compute_uv=False)
    return int((sv > RTOL * sv[0]).sum()) if sv[0] > 0 else 0


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
    # ring lattice with k neighbours, each edge rewired with probability p; simple, connected, N*k/2 edges
    rng = np.random.default_rng(seed)
    while True:
        S = set()
        for i in range(N):
            for d in range(1, k // 2 + 1):
                j = (i + d) % N
                if rng.random() < p or (min(i, j), max(i, j)) in S:
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
        if len(seen) == N and len(E) == N * k // 2: return N, E


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
        return nrank(np.array(cols).T)

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
        return int((sv > RTOL * sv[0]).sum())


def angle(a, b):
    c = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, c))))


def fitted(r, r_start, t):
    return abs(r) <= CONV * abs(r_start) + ATOL * abs(t)


def train(net, g_start, c, f, target):
    return least_squares(lambda q: net.y(q, c, f) - target, g_start, jac=lambda q: net.J(q, c, f), **LS).x


def project(net, gA, retained, cB, fB, target, tau=0.05, iters=400, correct=False):
    """Relinearised null-space projection with a projected-gain test and backtracking.
    correct=True takes the joint first-order step s = n + Pi z, with n = -J_K^+ R_K the Gauss-Newton correction of the
    retained residual and z the minimum-norm solution of J_B Pi z = -(r_B + J_B n); convergence then also requires
    every retained residual to pass the same test as the new one.
    retained: list of (c, f, target) for the tasks to be kept. Returns g and a dict of diagnostics.
    status: 'converged', 'no_gain' (projected gain below ETA*||J_B||), 'stalled' (no acceptable step), 'max_iter'."""
    g = gA.copy(); clipped = 0; path = 0.0; status = 'max_iter'; it = 0; gain0 = None
    r0 = abs(float((net.y(g, cB, fB) - target)[0]))
    for it in range(1, iters + 1):
        JK = np.vstack([net.J(g, c, f) for c, f, _ in retained])
        Pi = np.eye(net.M) - np.linalg.pinv(JK) @ JK
        r = net.y(g, cB, fB) - target
        RK = np.concatenate([net.y(g, c, f) - t for c, f, t in retained])
        done = fitted(r[0], r0, target[0])
        if correct: done = done and all(fitted(RK[i], 0.0, retained[i][2][0]) for i in range(len(retained)))
        if done: status = 'converged'; it -= 1; break
        JB = net.J(g, cB, fB); b = JB @ Pi
        gain = float(np.linalg.norm(b) / np.linalg.norm(JB))
        if gain0 is None: gain0 = gain
        if gain < ETA: status = 'no_gain'; it -= 1; break
        b = b.ravel()
        n = -np.linalg.pinv(JK) @ RK if correct else np.zeros(net.M)
        s = n - b * float(r[0] + (JB @ n)[0]) / float(b @ b)   # n plus the minimum-norm solution in range(Pi)
        nrm = np.linalg.norm(s)
        if nrm > tau: s *= tau / nrm
        def merit(q):                                    # new-task residual; with correction, the stacked residual
            m = float((net.y(q, cB, fB) - target)[0]) ** 2
            if correct: m += sum(float((net.y(q, c, f) - t)[0]) ** 2 for c, f, t in retained)
            return m
        m0 = merit(g); ok = False
        for _ in range(30):                              # backtracking
            gn = np.clip(g + s, GMIN, GMAX)
            if merit(gn) < m0: ok = True; break
            s *= 0.5
        if not ok: status = 'stalled'; it -= 1; break
        clipped = max(clipped, int(np.sum((g + s < GMIN) | (g + s > GMAX))))
        path += float(np.linalg.norm(gn - g)); g = gn
    rf = abs(float((net.y(g, cB, fB) - target)[0]))
    if status == 'max_iter' and fitted(rf, r0, target[0]) and not correct: status = 'converged'
    return g, dict(status=status, iters=it, clipped=clipped, path=path, gain0=gain0,
                   loss_ratio=(rf / r0) ** 2 if r0 > 0 else 0.0)
