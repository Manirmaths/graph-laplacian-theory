exec(open('03_train_task_A.py').read())
QA=np.linalg.pinv(JA)@JA
# scalar task-B family, rank controlled: Gamma vs forgetting per unit squared update
print("=== E7b  scalar task B only (rank J_B = 1), so output dimension is controlled ===")
print("   Gamma    f@node obs   ||dg||    F_A          F_A/(0.5||dg||^2)")
rows=[]
for fn in [4,6,8,16,18,20,22,24,12,2,10]:
    for on in [4,12,20,6,18]:
        Cb=obs([on]); fb=forcing(fn); Jb=J(gA,Cb,fb)
        if np.linalg.norm(Jb)<1e-12: continue
        G=float(np.trace(QA@(np.linalg.pinv(Jb)@Jb)))
        base=y(gA,Cb,fb); g=gA.copy()
        for s in np.linspace(0.2,1.0,6):
            yt=base*(1.0+0.03*s)
            g=least_squares(lambda q:y(q,Cb,fb)-yt,g,jac=lambda q:J(q,Cb,fb),**LS).x
        dg=g-gA; F=loss(g,CA,fA,yA_t)-LA0; nr=np.linalg.norm(dg)
        rows.append((G,F/(0.5*nr**2),nr,F,fn,on))
rows.sort()
for r in rows[::3]: print("   %.4f   f@%-2d o%-3d  %8.5f  %+.4e   %.4e"%(r[0],r[4],r[5],r[2],r[3],r[1]))
import scipy.stats as st
a=np.array([[r[0],r[1]] for r in rows])
print("   Spearman(Gamma, forgetting per unit squared update) = %.3f  over %d scalar task pairs"%(
      st.spearmanr(a[:,0],a[:,1]).statistic,len(rows)))
np.save('E7b.npy',a)
np.save('E7b_full.npy',np.array([r[:6] for r in rows]))   # Gamma, ratio, ||dg||, F_A, inject node, observed node

# ---------------- E8: plain sequential training vs null-space projection ----------------
print("\n=== E8  plain sequential training vs projected (null-space) training ===")
fb=forcing(12); Cb=obs([4]); base=y(gA,Cb,fb); yB_t=base*1.03      # Gamma(A,B) = 0.926
LB0=loss(gA,Cb,fb,yB_t)
gp=least_squares(lambda q:y(q,Cb,fb)-yB_t,gA,jac=lambda q:J(q,Cb,fb),**LS).x
print("  plain:      L_B/L_B0=%.3e   forgetting F_A=%.4e   ||dg||=%.5f"%(
      loss(gp,Cb,fb,yB_t)/LB0, loss(gp,CA,fA,yA_t)-LA0, np.linalg.norm(gp-gA)))
# projected Gauss-Newton: relinearise J_A at the current point each step
g=gA.copy()
for it in range(400):
    Jt=J(g,CA,fA); Pi=np.eye(M)-np.linalg.pinv(Jt)@Jt
    r=y(g,Cb,fb)-yB_t; Jb=J(g,Cb,fb)
    step=-Pi@np.linalg.pinv(Jb@Pi)@r
    nrm=np.linalg.norm(step)
    if nrm>0.05: step*=0.05/nrm
    g=np.clip(g+step,GMIN,GMAX)
print("  projected:  L_B/L_B0=%.3e   forgetting F_A=%.4e   ||dg||=%.5f"%(
      loss(g,Cb,fb,yB_t)/LB0, loss(g,CA,fA,yA_t)-LA0, np.linalg.norm(g-gA)))
print("  forgetting ratio, plain / projected: %.0f"%(
      (loss(gp,CA,fA,yA_t)-LA0)/max(loss(g,CA,fA,yA_t)-LA0,1e-30)))
# first-order safe update alone (single step, no relinearisation)
r0=y(gA,Cb,fb)-yB_t; Jb0=J(gA,Cb,fb); PiA0=np.eye(M)-np.linalg.pinv(JA)@JA
d1=-PiA0@np.linalg.pinv(Jb0@PiA0)@r0
print("  one-step safe update: L_B/L_B0=%.4f  F_A=%.3e  ||dg||=%.5f"%(
      loss(gA+d1,Cb,fb,yB_t)/LB0, loss(gA+d1,CA,fA,yA_t)-LA0, np.linalg.norm(d1)))
np.save('E8.npy',np.array([loss(gp,CA,fA,yA_t)-LA0,np.linalg.norm(gp-gA),loss(gp,Cb,fb,yB_t)/LB0,
                           loss(g,CA,fA,yA_t)-LA0,np.linalg.norm(g-gA),loss(g,Cb,fb,yB_t)/LB0]))
