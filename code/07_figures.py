import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,
  'ytick.labelsize':8,'figure.dpi':160,'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.5,
  'lines.linewidth':1.4,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
C0,C1,C2,C3='#1f4e79','#c1502e','#2e7d5b','#7a5c99'
exec(open('01_network.py').read())
from scipy.optimize import least_squares

# ================= reference network (untrained g_A) =================
rng=np.random.default_rng(260613); gO=rng.uniform(0.75,1.25,size=M)
fO=forcing(24); CAo=obs([4]); CBo=obs([4,20,12])
yAo=y(gO,CAo,fO); JAo=J(gO,CAo,fO)
def Fo(dg): return 0.5*float((y(gO+dg,CAo,fO)-yAo)[0]**2)
V=np.linalg.svd(JAo)[2]; vpar,vperp=V[0],V[-1]
eps=np.logspace(-4.5,-1,26)
Fs=np.array([Fo(e*vpar) for e in eps]); Fq=np.array([0.5*float((JAo@(e*vpar))[0]**2) for e in eps])
rem=np.abs(Fs-Fq); Fn=np.array([Fo(e*vperp) for e in eps])

# ratio of measured forgetting to the quadratic law
fig,ax=plt.subplots(figsize=(3.4,2.6))
ax.semilogx(eps,Fs/Fq,'o-',color=C0,ms=3)
ax.axhline(1,color='0.4',ls='--',lw=1)
ax.set_xlabel(r'perturbation size $\varepsilon$'); ax.set_ylabel(r'$F_A\,/\,\frac{1}{2}\|J_A\Delta g\|^2$')
ax.set_title('ratio to the quadratic law',fontsize=9)
fig.savefig('figures/figure_quadratic_ratio.pdf'); plt.close(fig)

# Taylor remainder with slope-3 reference
fig,ax=plt.subplots(figsize=(3.4,2.6))
ax.loglog(eps,rem,'o-',color=C0,ms=3,label='measured remainder')
ax.loglog(eps,rem[12]*(eps/eps[12])**3,'--',color='0.45',lw=1,label=r'slope 3')
ax.set_xlabel(r'perturbation size $\varepsilon$'); ax.set_ylabel(r'$|F_A-\frac{1}{2}\|J_A\Delta g\|^2|$')
ax.legend(frameon=False); fig.savefig('figures/figure_quadratic_remainder.pdf'); plt.close(fig)

# null vs sensitive directions with slope references
fig,ax=plt.subplots(figsize=(3.7,2.8))
ax.loglog(eps,Fs,'o-',color=C1,ms=3,label='sensitive subspace')
ax.loglog(eps,Fn,'s-',color=C0,ms=3,label='null space')
ax.loglog(eps,Fs[10]*(eps/eps[10])**2,'--',color='0.45',lw=1,label='slope 2')
ax.loglog(eps,Fn[10]*(eps/eps[10])**4,':',color='0.45',lw=1.2,label='slope 4')
ax.set_xlabel(r'perturbation size $\varepsilon$'); ax.set_ylabel('old-task forgetting $F_A$')
ax.legend(frameon=False,loc='lower right'); fig.savefig('figures/figure_null_vs_sensitive.pdf'); plt.close(fig)

# rank growth and conditioning
E6=np.load('E6.npy')
fig,axs=plt.subplots(1,2,figsize=(6.6,2.9))
k=np.arange(1,61); axs[0].plot(k,np.minimum(k,40),'-',color=C0,label='independent: rank')
axs[0].plot(k,np.maximum(40-np.minimum(k,40),0),'-',color=C1,label='independent: plastic dim')
axs[0].plot(k,np.minimum(k,5)*0+5,'--',color=C2,label='redundant: rank')
axs[0].plot(k,k*0+35,':',color=C3,label='redundant: plastic dim')
axs[0].set_xlabel('number of retained scalar tasks'); axs[0].set_ylabel('dimension')
axs[0].legend(frameon=False,fontsize=7,ncol=2,loc='lower center',bbox_to_anchor=(0.5,1.0),columnspacing=1.0,handlelength=1.8)
axs[1].semilogy(E6[:,0],E6[:,2],'-',color=C0)
axs[1].axvline(40,color='0.5',ls='--',lw=1); axs[1].set_xlabel('number of retained scalar tasks')
axs[1].set_ylabel(r'$\sigma_{\min}/\sigma_{\max}$ of $\mathcal{J}_K$')
axs[1].set_title('conditioning of the retained constraints',fontsize=8)
fig.savefig('figures/figure_rank_capacity.pdf'); plt.close(fig)

# ================= trained network =================
E1=np.load('E1.npy'); E2u=np.load('E2_uniform.npy'); E2s=np.load('E2_sensitivity.npy'); E8=np.load('E8.npy')
E7=np.load('E7b.npy')

# reach of the local quadratic law under sequential training
fig,axs=plt.subplots(1,2,figsize=(6.6,2.7))
axs[0].loglog(E1[:,1],E1[:,3],'o-',color=C1,ms=3,label='measured forgetting')
axs[0].loglog(E1[:,1],E1[:,4],'--',color=C0,lw=1.2,label=r'$\frac{1}{2}\|J_A(g_A)\Delta g\|^2$')
axs[0].set_xlabel(r'$\|\Delta g\|$ after training task B'); axs[0].set_ylabel('$F_A$'); axs[0].legend(frameon=False)
a2=axs[1]; a2.semilogx(E1[:,1],E1[:,5],'o-',color=C1,ms=3)
a2.axhline(1,color='0.4',ls='--',lw=1); a2.axhspan(0.9,1.1,color='0.85',alpha=0.5,zorder=0)
a2.set_xlabel(r'$\|\Delta g\|$'); a2.set_ylabel('predicted / measured')
a3=a2.twinx(); a3.semilogx(E1[:,1],E1[:,2],'^:',color=C2,ms=3); a3.set_ylabel(r'rotation of $\ker J_A$ (deg)',color=C2)
a3.tick_params(axis='y',colors=C2); a3.grid(False)
fig.savefig('figures/figure_local_validity.pdf'); plt.close(fig)

# uniform anchoring, sensitivity anchoring and projection on the trained network
fig,ax=plt.subplots(figsize=(4.0,3.0))
ax.loglog(np.maximum(E2u[:,2],1e-20),E2u[:,3],'o-',color=C0,ms=3.5,label='uniform anchoring')
ax.loglog(np.maximum(E2s[:,2],1e-20),np.maximum(E2s[:,3],1e-12),'s-',color=C1,ms=3.5,label='sensitivity anchoring')
ax.loglog([E8[3]],[max(E8[5],1e-12)],'*',color=C2,ms=15,label='null-space projection')
ax.set_xlabel('old-task forgetting $F_A$'); ax.set_ylabel(r'new-task loss  $\mathcal{L}_B/\mathcal{L}_B(g_A)$')
ax.legend(frameon=False,loc='lower left',fontsize=7.5)
ax.set_title('retention-adaptation trade-off',fontsize=8)
fig.savefig('figures/figure_tradeoff_trained.pdf'); plt.close(fig)

# Gamma vs forgetting per unit squared update
fig,ax=plt.subplots(figsize=(3.7,2.7))
ax.plot(E7[:,0],E7[:,1],'o',color=C1,ms=4,alpha=0.8)
z=np.polyfit(E7[:,0],E7[:,1],1); xx=np.linspace(E7[:,0].min(),E7[:,0].max(),10)
ax.plot(xx,np.polyval(z,xx),'--',color='0.45',lw=1)
ax.set_xlabel(r'sensitivity-subspace overlap $\Gamma(A,B)$')
ax.set_ylabel(r'$F_A\,/\,\frac{1}{2}\|\Delta g\|^2$')
ax.set_title(r'55 scalar task pairs, Spearman $=0.95$',fontsize=8)
fig.savefig('figures/figure_gamma_forgetting.pdf'); plt.close(fig)

# edge current vs exact sensitivity
E4=np.load('E4.npy')
fig,ax=plt.subplots(figsize=(3.6,2.7))
ax.plot(E4[:,0],E4[:,1],'o',color=C0,ms=4,alpha=0.75)
tI=set(np.argsort(-E4[:,0])[:8]); tS=set(np.argsort(-E4[:,1])[:8])
mis=list(tI^tS); ax.plot(E4[mis,0],E4[mis,1],'o',mfc='none',mec=C1,ms=9,mew=1.3,label='disagreement in top 8')
ax.set_xlabel(r'edge current $|I_{A,e}|$'); ax.set_ylabel(r'exact sensitivity $s_{A,e}$')
ax.legend(frameon=False,fontsize=7.5); ax.set_title(r'Pearson $0.62$, Spearman $0.45$',fontsize=8)
fig.savefig('figures/figure_current_vs_sensitivity.pdf'); plt.close(fig)
print("wrote:")
import os
for f in sorted(os.listdir('figures')): print("  figures/"+f, "%.0f kB"%(os.path.getsize('figures/'+f)/1024))
