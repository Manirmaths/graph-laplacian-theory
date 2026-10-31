exec(open('03_train_task_A.py').read())
import scipy.stats as st
fb=forcing(12); Cb=obs([4]); base=y(gA,Cb,fb)
print("=== E1  reach of the local quadratic law (continuation, warm-started) ===")
print("   amplitude   ||dg||   rotation   F_actual     F_quad      pred/actual")
E1=[]; g=gA.copy(); prev=0.0
for s in np.concatenate([np.logspace(-3.2,-0.6,9),np.linspace(0.3,1.0,6)]):
    yt=base*(1.0+0.06*s)
    g=least_squares(lambda q:y(q,Cb,fb)-yt,g,jac=lambda q:J(q,Cb,fb),**LS).x
    dg=g-gA; F=loss(g,CA,fA,yA_t)-LA0; Fq=0.5*float((JA@dg)[0]**2)
    Jt=J(g,CA,fA); rot=np.degrees(np.arcsin(min(1.0,np.linalg.norm(PiA-(np.eye(M)-np.linalg.pinv(Jt)@Jt),2))))
    E1.append((s,np.linalg.norm(dg),rot,F,Fq,Fq/F))
    print("   %8.5f  %8.5f  %6.2f  %.4e  %.4e   %6.3f"%E1[-1])
E1=np.array(E1); np.save('E1.npy',E1)
ok=E1[np.abs(E1[:,5]-1)<0.10]
print("   within 10%%: ||dg|| <= %.4f (rotation %.2f deg).  At full amplitude pred/actual=%.2f, rotation %.1f deg"%(
      ok[-1,1],ok[-1,2],E1[-1,5],E1[-1,2]))

print("\n=== E2  anchoring on the trained network, full nonlinear objective ===")
yB_t=base*1.03; LB0=loss(gA,Cb,fb,yB_t)
for name in ["uniform","sensitivity"]:
    rows=[]; g=gA.copy()
    for lam in np.logspace(-3,6,13):
        sq=np.sqrt(lam)
        if name=="uniform": res=lambda q:np.r_[y(q,Cb,fb)-yB_t,sq*(q-gA)]; jac=lambda q:np.vstack([J(q,Cb,fb),sq*np.eye(M)])
        else:               res=lambda q:np.r_[y(q,Cb,fb)-yB_t,sq*(JA@(q-gA))]; jac=lambda q:np.vstack([J(q,Cb,fb),sq*JA])
        g=least_squares(res,g,jac=jac,**LS).x
        rows.append((lam,np.linalg.norm(g-gA),loss(g,CA,fA,yA_t)-LA0,loss(g,Cb,fb,yB_t)/LB0))
    a=np.array(rows); np.save('E2_%s.npy'%name,a)
    print("  %-12s lam=1e6: ||dg||=%.5f  F_A=%.3e  L_B/L_B0=%.5f"%(name,*a[-1,1:]))
E8=np.load('E8.npy'); print("  projection   (no lambda): ||dg||=%.5f  F_A=%.3e  L_B/L_B0=%.2e"%(E8[4],E8[3],E8[5]))

print("\n=== E3  max attainable rank = rank{b_e b_e^T} ===")
E3=[]
for lab,gr in [("single ground (node 0)",[0]),("nodes 0,1 - adjacent",[0,1]),("nodes 0,2 - share free node 1",[0,2]),
               ("nodes 0,24 - opposite corners",[0,24]),("entire top row, nodes 0-4",[0,1,2,3,4])]:
    fr=[i for i in range(N) if i not in gr]; Bx=B[fr,:]
    K=np.array([np.outer(Bx[:,e],Bx[:,e]).ravel() for e in range(M)]).T
    both=sum(1 for (i,j) in edges if i in gr and j in gr)
    twin=sum(1 for v in fr if sum(1 for (i,j) in edges if (i==v and j in gr) or (j==v and i in gr))>=2)
    r=int(np.linalg.matrix_rank(K)); E3.append((lab,r,both,twin))
    print("   %-32s rank=%2d/%d  M-rank=%d   [both-ends grounded=%d, free nodes w/ >=2 ground edges=%d]"%(lab,r,M,M-r,both,twin))
import json; json.dump(E3,open('E3.json','w'))

print("\n=== E4  current as an importance proxy (at trained g_A) ===")
u=np.linalg.solve(Lap(gA),fA); I=gA*(Bf.T@u); s_=np.abs(JA).ravel()
print("   Pearson=%.4f   Spearman=%.4f"%(np.corrcoef(np.abs(I),s_)[0,1],st.spearmanr(np.abs(I),s_).statistic))
for k in (5,8,10): print("   top-%2d agree: %d/%d"%(k,len(set(np.argsort(-np.abs(I))[:k])&set(np.argsort(-s_)[:k])),k))
np.save('E4.npy',np.c_[np.abs(I),s_])

print("\n=== E5  trained network: quartic ratio along the safe direction; anchored vs safe update ===")
JB=J(gA,Cb,fb); S=np.linalg.svd(JB@PiA); vs=S[2][0]; sig=S[1][0]; E5=[]
for d in [4e-3,2e-3,1e-3,5e-4]:
    dg=(d/sig)*vs; F=loss(gA+dg,CA,fA,yA_t)-LA0; E5.append((d,F,F/d**4)); print("   delta=%.1e  F_A=%.5e  F_A/delta^4=%.5f"%E5[-1])
np.save('E5.npy',np.array(E5))
rB=y(gA,Cb,fb)-yB_t; dsafe=-PiA@np.linalg.pinv(JB@PiA)@rB
for lam in [1e2,1e4,1e6]:
    dl=-np.linalg.pinv(JB.T@JB+lam*(JA.T@JA))@(JB.T@rB)
    print("   lam=%.0e  ||dg_lam - dg_safe|| / ||dg_safe|| = %.4e"%(lam,np.linalg.norm(dl-dsafe)/np.linalg.norm(dsafe)))

print("\n=== E6  rank growth and conditioning of the stacked Jacobian ===")
P=Bf.T@np.linalg.inv(Lap(gA)); rows=[];rk=[];cond=[]
for k in range(1,61):
    ff=rng.normal(size=Nf); cc=rng.normal(size=Nf)
    rows.append(-(P@(ff/np.linalg.norm(ff)))*(P@(cc/np.linalg.norm(cc))))
    sv=np.linalg.svd(np.array(rows),compute_uv=False)
    rk.append(int(np.linalg.matrix_rank(np.array(rows),tol=1e-9*sv[0]))); cond.append(sv[-1]/sv[0])
np.save('E6.npy',np.c_[np.arange(1,61),rk,cond])
i40=int(np.argmax(np.array(rk)>=40))
print("   rank reaches M=40 at task %d; sigma_min/sigma_max there = %.2e ; at 60 tasks = %.2e"%(i40+1,cond[i40],cond[-1]))
