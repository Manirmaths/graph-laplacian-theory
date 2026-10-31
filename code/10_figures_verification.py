# Figures for the verification checks and the trained-network comparisons. Run after 09_verification_checks.py, from inside code/.
#   figures/figure_sensitivity_identity.pdf  complex-step and finite-difference check of Eq. (jacobian)
#   figures/figure_quadratic_law.pdf          (a) ratio to the quadratic law, (b) Taylor remainder
#   figures/figure_pathway_maps.pdf           forward, current, adjoint and sensitivity maps
#   figures/figure_gamma_forgetting.pdf       Gamma vs forgetting, coloured by update size
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, json
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,
  'ytick.labelsize':8,'figure.dpi':160,'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.5,
  'lines.linewidth':1.4,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
C0,C1,C2,C3='#1f4e79','#c1502e','#2e7d5b','#7a5c99'
exec(open('01_network.py').read())
chk=json.load(open('E9_checks.json'))

# ---- sensitivity identity ----
E10=np.load('E10_jacobian.npy')
fig,ax=plt.subplots(figsize=(3.6,2.7))
ax.loglog(E10[:,0],E10[:,1],'o-',color=C0,ms=3,label='centred finite differences')
ax.axhline(chk['C_jacobian']['complex_step_relerr'],color=C1,ls='--',lw=1.2,label='complex step')
ib=int(np.argmin(E10[:,1])); ax.loglog([E10[ib,0]],[E10[ib,1]],'o',mfc='none',mec=C1,ms=9)
ax.set_xlabel(r'step size $\varepsilon$'); ax.set_ylabel('relative error in $J_B$')
ax.legend(frameon=False,loc='upper left',fontsize=7.5)
fig.savefig('figures/figure_sensitivity_identity.pdf'); plt.close(fig)

# ---- quadratic law, two panels ----
rng=np.random.default_rng(260613); gR=rng.uniform(0.75,1.25,size=M)
fR=forcing(24); CA=obs([4]); JA=J(gR,CA,fR); yA0=y(gR,CA,fR)
def FA(dg): return 0.5*float((y(gR+dg,CA,fR)-yA0)[0]**2)
vpar=np.linalg.svd(JA)[2][0]; eps=np.logspace(-4.5,-1,26)
Fs=np.array([FA(e*vpar) for e in eps]); Fq=0.5*(JA@vpar)[0]**2*eps**2; rem=np.abs(Fs-Fq)
fig,axs=plt.subplots(1,2,figsize=(6.8,2.6))
axs[0].semilogx(eps,Fs/Fq,'o-',color=C0,ms=3); axs[0].axhline(1,color='0.4',ls='--',lw=1)
axs[0].set_xlabel(r'perturbation size $\varepsilon$'); axs[0].set_ylabel(r'$F_A\,/\,\frac{1}{2}\|J_A\Delta g\|^2$')
axs[0].set_title('(a) ratio to the quadratic law',fontsize=8.5,loc='left')
axs[1].loglog(eps,rem,'o-',color=C0,ms=3,label='measured remainder')
axs[1].loglog(eps,rem[12]*(eps/eps[12])**3,'--',color='0.45',lw=1,label='slope 3')
axs[1].set_xlabel(r'perturbation size $\varepsilon$'); axs[1].set_ylabel(r'$|F_A-\frac{1}{2}\|J_A\Delta g\|^2|$')
axs[1].set_title('(b) Taylor remainder',fontsize=8.5,loc='left'); axs[1].legend(frameon=False)
fig.tight_layout(); fig.savefig('figures/figure_quadratic_law.pdf'); plt.close(fig)

# ---- pathway maps ----
L=Lap(gR); u=np.linalg.solve(L,fR); w=np.linalg.solve(L,CA[0])
theta=Bf.T@u; phi=Bf.T@w
fields=[(np.abs(theta),r'(a) forward voltage drop $|\theta_{A,e}|$'),
        (np.abs(gR*theta),r'(b) forward current $|I_{A,e}|$'),
        (np.abs(phi),r'(c) adjoint voltage drop $|\phi_{A,e}|$'),
        (np.abs(theta*phi),r'(d) sensitivity $|\partial y_A/\partial g_e|$')]
xy={i:(i%5,-(i//5)) for i in range(N)}; WMAX=7.0
fig,axs=plt.subplots(2,2,figsize=(6.8,7.0))
for ax,(val,title) in zip(axs.ravel(),fields):
    ax.grid(False); ax.set_aspect('equal'); ax.axis('off')
    vmax=val.max()
    for k,(i,j) in enumerate(edges):
        (x1,y1),(x2,y2)=xy[i],xy[j]
        ax.plot([x1,x2],[y1,y2],'-',color=C0,lw=0.4+WMAX*val[k]/vmax,solid_capstyle='round',zorder=1)
    for i in range(N):
        ax.plot(*xy[i],'o',color='0.55',ms=3.5,zorder=2)
    ax.plot(*xy[0],'s',color='k',ms=9,zorder=3); ax.plot(*xy[24],'^',color=C1,ms=10,zorder=3)
    ax.plot(*xy[4],'D',color=C2,ms=8,zorder=3)
    ax.set_title(title,fontsize=8.5,loc='left')
    # width scale: 25%, 50%, 100% of the panel maximum
    for r,frac in enumerate((1.0,0.5,0.25)):
        yy=-4.75-0.32*r
        ax.plot([0.0,0.6],[yy,yy],'-',color=C0,lw=0.4+WMAX*frac,solid_capstyle='butt')
        ax.text(0.75,yy,'%.3g'%(frac*vmax),va='center',fontsize=7)
    ax.text(0.0,-4.45,'width scale',fontsize=7)
    ax.set_xlim(-0.4,4.4); ax.set_ylim(-5.55,0.4)
handles=[Line2D([],[],marker='^',color=C1,ls='',ms=9,label='source (node 24)'),
         Line2D([],[],marker='s',color='k',ls='',ms=8,label='ground (node 0)'),
         Line2D([],[],marker='D',color=C2,ls='',ms=7,label='observed (node 4)')]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=8)
fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig('figures/figure_pathway_maps.pdf'); plt.close(fig)

# ---- Gamma vs forgetting, with update size and first-order line ----
a=np.load('E7b_full.npy'); nJA2=chk['E_gamma']['norm_JA_sq']
fig,ax=plt.subplots(figsize=(3.9,2.9))
inr=a[:,2]<=0.105
ax.plot(a[inr,0],a[inr,1],'o',color=C0,ms=4,label=r'$\|\Delta g\|\leq0.105$')
ax.plot(a[~inr,0],a[~inr,1],'o',mfc='none',mec=C1,ms=6,mew=1.3,label=r'$\|\Delta g\|>0.105$')
gg=np.linspace(0.35,1.0,50); ax.plot(gg,nJA2*gg,'--',color='0.45',lw=1,label=r'first order: $\|J_A\|^2\Gamma$')
ax.set_xlabel(r'sensitivity-subspace overlap $\Gamma(A,B)$'); ax.set_ylabel(r'$F_A\,/\,\frac{1}{2}\|\Delta g\|^2$')
ax.legend(frameon=False,fontsize=7,loc='upper left')
fig.savefig('figures/figure_gamma_forgetting.pdf'); plt.close(fig)
# ---- current vs sensitivity on the reference network ----
import scipy.stats as st
Iref=np.abs(gR*theta); Sref=np.abs(theta*phi)
fig,ax=plt.subplots(figsize=(3.6,2.7))
ax.plot(Iref,Sref,'o',color=C0,ms=4,alpha=0.75)
tI=set(np.argsort(-Iref)[:8]); tS=set(np.argsort(-Sref)[:8]); mis=sorted(tI^tS)
ax.plot(Iref[mis],Sref[mis],'o',mfc='none',mec=C1,ms=9,mew=1.3,label='disagreement in top 8')
ax.set_xlabel(r'edge current $|I_{A,e}|$'); ax.set_ylabel(r'exact sensitivity $s_{A,e}$')
ax.legend(frameon=False,fontsize=7.5)
ax.set_title('Pearson %.3f, Spearman %.3f'%(np.corrcoef(Iref,Sref)[0,1],st.spearmanr(Iref,Sref).statistic),fontsize=8)
fig.savefig('figures/figure_current_vs_sensitivity.pdf'); plt.close(fig)
print('figures written')
