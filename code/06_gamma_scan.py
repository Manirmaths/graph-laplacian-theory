exec(open('03_train_task_A.py').read())
QA=np.linalg.pinv(JA)@JA          # projector onto rowspace(J_A)
print("Gamma(A,B) for candidate task B (forcing node / observed nodes):")
cands=[]
for fn in [20,24,12,4,6]:
    for ob in [[4],[12],[4,12],[20,12],[6,18],[4,20,12]]:
        Jb=J(gA,obs(ob),forcing(fn))
        if np.linalg.matrix_rank(Jb)<len(ob): continue
        QB=np.linalg.pinv(Jb)@Jb
        G=np.trace(QA@QB)/np.sqrt(np.linalg.matrix_rank(JA)*np.linalg.matrix_rank(Jb))
        cands.append((G,fn,tuple(ob)))
for G,fn,ob in sorted(cands,reverse=True):
    print("   Gamma=%.4f   f at node %-3d observe %s"%(G,fn,list(ob)))
