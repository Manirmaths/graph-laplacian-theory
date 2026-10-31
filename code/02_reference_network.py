import numpy as np
np.set_printoptions(precision=10, suppress=False)
rng = np.random.default_rng(260613)

# ---- 5x5 grid, node 0 grounded ----
n=5; N=n*n
def idx(r,c): return r*n+c
edges=[]
for r in range(n):
    for c in range(n):
        if c+1<n: edges.append((idx(r,c), idx(r,c+1)))
        if r+1<n: edges.append((idx(r,c), idx(r+1,c)))
M=len(edges); print("N",N,"M",M)

B=np.zeros((N,M))
for k,(i,j) in enumerate(edges):
    B[i,k]=1.0; B[j,k]=-1.0
ground=[0]
free=[i for i in range(N) if i not in ground]
Bf=B[free,:]; Nf=len(free); print("Nf",Nf, "Bf rank", np.linalg.matrix_rank(Bf))

def Lap(g): return Bf@np.diag(g)@Bf.T
pos=lambda node: free.index(node)

f=np.zeros(Nf); f[pos(24)]=1.0            # unit current in at node 24, out through ground
cA=np.zeros(Nf); cA[pos(4)]=1.0
CB=np.zeros((3,Nf))
for row,nd in enumerate([4,20,12]): CB[row,pos(nd)]=1.0
CA=cA.reshape(1,-1)

# --- reference conductances ---
gA = rng.uniform(0.75,1.25,size=M)
L=Lap(gA); uA=np.linalg.solve(L,f)
print("\n[check] y_A(g_A) =", CA@uA)
gu=np.ones(M); print("[check] y_A with g = 1 on every edge =", CA@np.linalg.solve(Lap(gu),f))

# ---- 1. sensitivity identity vs finite differences ----
def yfun(g,C): return C@np.linalg.solve(Lap(g),f)
def Jexact(g,C):
    L=Lap(g); u=np.linalg.solve(L,f)
    theta=Bf.T@u                      # forward voltage drops
    return -(C@np.linalg.solve(L,Bf))*theta   # broadcast over columns
JA=Jexact(gA,CA); JB=Jexact(gA,CB)
eps=2e-5; Jfd=np.zeros_like(JB)
for e in range(M):
    d=np.zeros(M); d[e]=eps
    Jfd[:,e]=(yfun(gA+d,CB)-yfun(gA-d,CB))/(2*eps)
print("\n[1] rel err analytic vs centred FD (J_B): %.3e"%(np.linalg.norm(JB-Jfd)/np.linalg.norm(JB)))

# ---- 2. maximal attainable rank of physical sensitivity rows ----
P = Bf.T@np.linalg.inv(Lap(gA))       # rows rho(f,c) = -(P f) * (P c)
rows=[]
for _ in range(200):
    ff=rng.normal(size=Nf); cc=rng.normal(size=Nf)
    rows.append(-(P@ff)*(P@cc))
R=np.array(rows)
print("\n[2] rank of 200 random scalar sensitivity rows:", np.linalg.matrix_rank(R, tol=1e-9*np.linalg.norm(R)))
# rank of h -> B_f diag(h) B_f^T
Kmat=np.array([np.outer(Bf[:,e],Bf[:,e]).ravel() for e in range(M)]).T   # Nf^2 x M
print("[2] rank of the family {b_e b_e^T}:", np.linalg.matrix_rank(Kmat))
# two adjacent grounded nodes {0,1}
g2=[0,1]; free2=[i for i in range(N) if i not in g2]
Bf2=B[free2,:]
K2=np.array([np.outer(Bf2[:,e],Bf2[:,e]).ravel() for e in range(M)]).T
print("[2] grounded set {0,1}: rank =", np.linalg.matrix_rank(K2), "of M =",M)

# ---- 3. scaling exponents ----
U,S,Vt=np.linalg.svd(JA)
vpar=Vt[0]                                   # in range(J_A^T)
Z=Vt[np.abs(S if S.size==Vt.shape[0] else np.r_[S,np.zeros(Vt.shape[0]-S.size)])<1e-12] if False else Vt[1:]
vperp=Vt[-1]                                 # null direction (J_A is 1 x M)
print("\n[3] ||J_A v_perp|| = %.3e   ||J_A v_par|| = %.6f"%(np.linalg.norm(JA@vperp), np.linalg.norm(JA@vpar)))
yAstar=float(yfun(gA,CA)[0])
def F(dg): return 0.5*float((yfun(gA+dg,CA)[0]-yAstar)**2)
def slope(v):
    es=np.logspace(-4,-2.3,12); Fs=np.array([F(e*v) for e in es])
    m=Fs>0; return np.polyfit(np.log(es[m]),np.log(Fs[m]),1)[0]
print("[3] fitted slope, sensitive dir : %.4f"%slope(vpar))
print("[3] fitted slope, null dir      : %.4f"%slope(vperp))
es=np.logspace(-4,-2.3,12)
rem=np.array([abs(F(e*vpar)-0.5*float((JA@(e*vpar))[0]**2)) for e in es])
print("[3] fitted slope, Taylor remainder: %.4f"%np.polyfit(np.log(es),np.log(rem),1)[0])

# ---- 4. anchoring limits ----
PiA=np.eye(M)-np.linalg.pinv(JA)@JA
Us,Ss,Vts=np.linalg.svd(JB@PiA)
vs=Vts[0]; sig=Ss[0]; qs=(JB@vs)/sig
print("\n[4] ||J_A v_s|| = %.3e   sigma_s = %.10f   q_s = %s"%(np.linalg.norm(JA@vs), sig, np.round(qs,10)))
delta=5e-3; dyB=delta*(np.array([1.0,0,0])+qs)
yBstar=yfun(gA,CB)+dyB; rB=yfun(gA,CB)-yBstar
LB0=0.5*float(rB@rB)
for name,MA in [("uniform  M=I",np.eye(M)),("sensitivity M=J^T J",JA.T@JA)]:
    lam=1e5; G=JB.T@JB+lam*MA
    dg=-np.linalg.pinv(G)@(JB.T@rB)
    LB=0.5*float(np.sum((yfun(gA+dg,CB)-yBstar)**2))
    print("    %-20s ||dg||=%.6e   F_A=%.3e   L_B/L_B(gA)=%.6f"%(name,np.linalg.norm(dg),F(dg),LB/LB0))
print("    constructed safe update ||dg_s|| = %.10e"%(delta/sig))

# ---- 5. current vs sensitivity ----
theta=Bf.T@uA; I=gA*theta; s=np.abs(JA).ravel()
print("\n[5] Pearson |I| vs sensitivity: %.4f"%np.corrcoef(np.abs(I),s)[0,1])
from scipy.stats import spearmanr
print("[5] Spearman: %.4f"%spearmanr(np.abs(I),s).statistic)
k=8; topI=set(np.argsort(-np.abs(I))[:k]); topS=set(np.argsort(-s)[:k])
print("[5] top-%d by current vs top-%d by sensitivity: overlap %d/%d"%(k,k,len(topI&topS),k))
