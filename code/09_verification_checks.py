# Additional numerical checks reported in the manuscript. Run from inside code/:  python3 09_verification_checks.py
# (A) manufactured anchoring test of Sec. VII D, including the O(1/lambda) convergence
# (B) analytic quartic coefficient (1/8)||D^2 y_A[v,v]||^2 on the reference network
# (C) complex-step check of the analytic Jacobian, and the finite-difference sweep
# (D) attainable rank for multi-output tasks: grid versus a full-observation counterexample
# (E) the low outlier in the Gamma experiment
# (F) collinearity of null-space rotation and ||dg|| along the E1 continuation branch
import numpy as np, json
exec(open('01_network.py').read())
from scipy.optimize import least_squares
out = {}

# ---------- reference network (not trained) ----------
rng = np.random.default_rng(260613); gR = rng.uniform(0.75, 1.25, size=M)
fR = forcing(24); CA = obs([4]); CB = obs([4, 20, 12])
JA = J(gR, CA, fR); JB = J(gR, CB, fR); yA0 = y(gR, CA, fR)
PiA = np.eye(M) - np.linalg.pinv(JA) @ JA
def FA(dg): return 0.5 * float((y(gR + dg, CA, fR) - yA0)[0] ** 2)

# ---------- (A) manufactured anchoring test ----------
_, S, Vt = np.linalg.svd(JB @ PiA); vs = Vt[0]; sig = S[0]; qs = JB @ vs / sig
k = 1 + int(np.argmax(np.abs(qs[1:])))
if qs[k] < 0: vs, qs = -vs, -qs                      # sign convention stated in the paper
delta = 5e-3; dyB = delta * (np.array([1.0, 0, 0]) + qs)
yBs = y(gR, CB, fR) + dyB; rB = y(gR, CB, fR) - yBs; LB0 = 0.5 * float(rB @ rB)
dsafe = -PiA @ np.linalg.pinv(JB @ PiA) @ rB
A = dict(sigma_s=sig, q_s=qs.tolist(), JA_vs=float(np.linalg.norm(JA @ vs)),
         norm_dsafe=float(np.linalg.norm(dsafe)), rank_JBPiA=int(np.linalg.matrix_rank(JB @ PiA)))
lams = np.logspace(1, 6, 11); uni = []; sen = []
for lam in lams:
    for MA, store in [(np.eye(M), uni), (JA.T @ JA, sen)]:
        dg = -np.linalg.pinv(JB.T @ JB + lam * MA) @ (JB.T @ rB)
        LBn = 0.5 * float(np.sum((y(gR + dg, CB, fR) - yBs) ** 2)) / LB0
        LBl = 0.5 * float(np.sum((rB + JB @ dg) ** 2)) / LB0
        store.append((lam, np.linalg.norm(dg), FA(dg), LBn, LBl,
                      np.linalg.norm(dg - dsafe) / np.linalg.norm(dsafe)))
uni = np.array(uni); sen = np.array(sen)
A['uniform_slope_norm'] = float(np.polyfit(np.log(lams[4:]), np.log(uni[4:, 1]), 1)[0])
A['uniform_slope_F'] = float(np.polyfit(np.log(lams[4:]), np.log(uni[4:, 2]), 1)[0])
i5 = int(np.argmin(np.abs(lams - 1e5)))
A['uniform_at_1e5'] = dict(norm=uni[i5, 1], F=uni[i5, 2], LB_ratio=uni[i5, 3])
A['sens_at_1e5'] = dict(lin_ratio=sen[i5, 4], nonlin_ratio=sen[i5, 3], closed_form=0.5 * (1e5 / (1 + 1e5)) ** 2)
A['sens_reldist'] = {('%.0e' % l): float(sen[np.argmin(np.abs(lams - l)), 5]) for l in (1e2, 1e4, 1e6)}
A['sens_reldist_slope'] = float(np.polyfit(np.log(lams), np.log(sen[:, 5]), 1)[0])
A['JA_dg_lam1e2'] = float((JA @ (-np.linalg.pinv(JB.T @ JB + 1e2 * JA.T @ JA) @ (JB.T @ rB)))[0])
out['A_manufactured'] = A
np.save('E9_manufactured.npy', np.c_[uni, sen])

# ---------- (B) analytic quartic coefficient along the unit safe direction v_s ----------
L = Lap(gR); u = np.linalg.solve(L, fR)
Lv = Bf @ np.diag(vs) @ Bf.T
D2 = 2.0 * CA @ np.linalg.solve(L, Lv @ np.linalg.solve(L, Lv @ u))
coef = 0.125 * float(D2 @ D2)
ratios = {('%.1e' % e): FA(e * vs) / e ** 4 for e in (4e-2, 2e-2, 1e-2, 5e-3, 2.5e-3)}
out['B_quartic'] = dict(coef_analytic=coef, D2yvv=float(D2[0]), ratios=ratios,
                        Lv_u_norm=float(np.linalg.norm(Lv @ u)))
# slopes over the plotted window of Fig. 2 / Fig. 3
V = np.linalg.svd(JA)[2]; vpar, vperp = V[0], V[-1]
eps = np.logspace(-4.5, -1, 26)
Fs = np.array([FA(e * vpar) for e in eps]); Fq = 0.5 * (JA @ vpar)[0] ** 2 * eps ** 2
Fn = np.array([FA(e * vperp) for e in eps]); m = eps <= 10 ** -1.5
out['B_slopes'] = dict(sensitive=float(np.polyfit(np.log(eps[m]), np.log(Fs[m]), 1)[0]),
                       remainder=float(np.polyfit(np.log(eps[m]), np.log(np.abs(Fs - Fq)[m]), 1)[0]),
                       null=float(np.polyfit(np.log(eps[m & (Fn > 1e-30)]), np.log(Fn[m & (Fn > 1e-30)]), 1)[0]),
                       JA_vperp=float(np.linalg.norm(JA @ vperp)),
                       ratio_max=float(np.max(Fs / Fq)), ratio_dev_eps_le_10m2p5=float(np.max(np.abs(Fs / Fq - 1)[eps <= 10 ** -2.5])),
                       window='eps in [10^-4.5, 10^-1.5]')

# ---------- (C) complex-step and finite differences ----------
def y_c(g): return CB @ np.linalg.solve(Bf @ np.diag(g) @ Bf.T, fR.astype(complex))
h = 1e-30; Jcs = np.zeros_like(JB)
for e in range(M):
    d = np.zeros(M, complex); d[e] = 1j * h
    Jcs[:, e] = np.imag(y_c(gR + d)) / h
fd_eps = np.logspace(-9, -1, 49); fd_err = []
for ee in fd_eps:
    Jfd = np.zeros_like(JB)
    for e in range(M):
        d = np.zeros(M); d[e] = ee
        Jfd[:, e] = (y(gR + d, CB, fR) - y(gR - d, CB, fR)) / (2 * ee)
    fd_err.append(np.linalg.norm(JB - Jfd) / np.linalg.norm(JB))
fd_err = np.array(fd_err); ib = int(np.argmin(fd_err))
out['C_jacobian'] = dict(complex_step_relerr=float(np.linalg.norm(JB - Jcs) / np.linalg.norm(JB)),
                         fd_best_eps=float(fd_eps[ib]), fd_best_err=float(fd_err[ib]))
np.save('E10_jacobian.npy', np.c_[fd_eps, fd_err])

# ---------- (D) attainable rank for p-output tasks ----------
def rank_growth(Bfx, gx, p, K, seed):
    r = np.random.default_rng(seed); Nfx = Bfx.shape[0]
    P = Bfx.T @ np.linalg.inv(Bfx @ np.diag(gx) @ Bfx.T); rows = []; rk = []
    for _ in range(K):
        ff = r.normal(size=Nfx)
        Cx = np.eye(Nfx) if p == Nfx else r.normal(size=(p, Nfx))
        for c in Cx: rows.append(-(P @ ff) * (P @ c))
        sv = np.linalg.svd(np.array(rows), compute_uv=False); rk.append(int((sv > 1e-9 * sv[0]).sum()))
    return rk
D = {}
for p in (2, 3, 5, 8, 12, 24):
    rk = rank_growth(Bf, gR, p, int(np.ceil(40 / p)) + 1, 7)
    D['grid_p%d' % p] = dict(rank=rk, predicted=[min((i + 1) * p, 40) for i in range(len(rk))])
# complete graph K_6, one ground: N_f = 5, M = d = 15
n6 = 6; e6 = [(i, j) for i in range(n6) for j in range(i + 1, n6)]
B6 = np.zeros((n6, len(e6)))
for kk, (i, j) in enumerate(e6): B6[i, kk] = 1; B6[j, kk] = -1
Bf6 = B6[1:, :]; g6 = np.random.default_rng(3).uniform(0.75, 1.25, len(e6))
for p in (4, 5):
    D['K6_p%d' % p] = dict(rank=rank_growth(Bf6, g6, p, 3, 11), predicted=[min(i * p, 15) for i in (1, 2, 3)],
                           bound_full_obs=[min(K * 5 - K * (K - 1) // 2, 15) for K in (1, 2, 3)])
out['D_rank'] = D

# ---------- (E) Gamma experiment: identify the low outlier ----------
exec(open('03_train_task_A.py').read())
QA = np.linalg.pinv(JA) @ JA; nJA2 = float(np.sum(JA ** 2)); pairs = []
for fn in [4, 6, 8, 16, 18, 20, 22, 24, 12, 2, 10]:
    for on in [4, 12, 20, 6, 18]:
        Cb = obs([on]); fb = forcing(fn); Jb = J(gA, Cb, fb)
        if np.linalg.norm(Jb) < 1e-12: continue
        pairs.append((float(np.trace(QA @ (np.linalg.pinv(Jb) @ Jb))), fn, on))
a = np.load('E7b.npy'); o = int(np.argmin(a[:, 1])); Gout = a[o, 0]
G, fn, on = min(pairs, key=lambda t: abs(t[0] - Gout))
Cb = obs([on]); fb = forcing(fn); base = y(gA, Cb, fb); g = gA.copy()
for s in np.linspace(0.2, 1.0, 6):
    yt = base * (1.0 + 0.03 * s)
    g = least_squares(lambda q: y(q, Cb, fb) - yt, g, jac=lambda q: J(q, Cb, fb), **LS).x
dg = g - gA; JbA = J(gA, Cb, fb)[0]
cos2_dg_JA = float((JA[0] @ dg) ** 2 / (nJA2 * (dg @ dg)))
out['E_gamma'] = dict(n_pairs=len(a), spearman=None, outlier=dict(Gamma=G, inject=fn, observe=on, ratio=float(a[o, 1]),
                      norm_dg=float(np.linalg.norm(dg)), active_bounds=int(np.sum((g <= GMIN + 1e-9) | (g >= GMAX - 1e-9))),
                      cos2_dg_JA=cos2_dg_JA, cos2_dg_JB=float((JbA @ dg) ** 2 / ((JbA @ JbA) * (dg @ dg))),
                      F_A=float(loss(g, CA, fA, yA_t) - LA0)),
                      norm_JA_sq=nJA2, median_ratio_over_firstorder=float(np.median(a[:, 1] / (nJA2 * a[:, 0]))))
import scipy.stats as st
out['E_gamma']['spearman'] = float(st.spearmanr(a[:, 0], a[:, 1]).statistic)
out['E_gamma']['pearson'] = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
srt = a[np.argsort(a[:, 0])]; out['E_gamma']['adjacent_inversions'] = int(np.sum(np.diff(srt[:, 1]) < 0))
keep = np.arange(len(a)) != o
out['E_gamma']['spearman_without_outlier'] = float(st.spearmanr(a[keep, 0], a[keep, 1]).statistic)

# ---------- (F) E1 branch ----------
E1 = np.load('E1.npy'); big = E1[:, 1] > 0.01
out['F_E1'] = dict(rotation_deg_per_unit_dg=(E1[big, 2] / E1[big, 1]).round(2).tolist())

json.dump(out, open('E9_checks.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
