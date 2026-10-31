import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,
 'ytick.labelsize':8,'figure.dpi':160,'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.5,
 'lines.linewidth':1.4,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
C0,C1,C2='#1f4e79','#c1502e','#2e7d5b'
E1=np.load('E1.npy'); E2u=np.load('E2_uniform.npy'); E2s=np.load('E2_sensitivity.npy'); E8=np.load('E8.npy')

fig,ax=plt.subplots(figsize=(4.6,3.1))
ax.axhspan(0.9,1.1,color='0.88',alpha=0.7,zorder=0,label='10% accuracy band')
ax.semilogx(E1[:,1],E1[:,5],'o-',color=C1,ms=3.5,zorder=3,label='predicted / measured')
ax.axhline(1,color='0.35',ls='--',lw=1,zorder=2)
ax.set_xlabel(r'$\|\Delta g\|$ after training task B to convergence')
ax.set_ylabel('predicted / measured forgetting'); ax.set_ylim(0.84,1.12)
a2=ax.twinx(); a2.semilogx(E1[:,1],E1[:,2],'^:',color=C2,ms=3.5,zorder=3,label=r'rotation of $\ker J_A$')
a2.set_ylabel(r'rotation of $\ker J_A$ (degrees)',color=C2); a2.tick_params(axis='y',colors=C2); a2.grid(False)
h1,l1=ax.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,frameon=False,fontsize=7.5,ncol=3,loc='lower center',bbox_to_anchor=(0.5,1.0),columnspacing=1.2,handlelength=1.8)
fig.savefig('figures/figure_local_validity.pdf'); plt.close(fig)

fig,ax=plt.subplots(figsize=(4.3,3.1))
fl=1e-11
ax.loglog(E2u[:,2],np.maximum(E2u[:,3],fl),'o-',color=C0,ms=3.5,label=r'uniform anchoring, $\lambda$ sweep')
ax.loglog([E2s[0,2]],[fl],'s',color=C1,ms=8,label=r'sensitivity anchoring (all $\lambda$)')
ax.loglog([E8[3]],[fl],'*',color=C2,ms=17,label='null-space projection')
ax.set_ylim(3e-12,4); ax.set_xlabel('old-task forgetting $F_A$')
ax.set_ylabel(r'new-task loss  $\mathcal{L}_B/\mathcal{L}_B(g_A)$')
ax.legend(frameon=False,loc='center left',fontsize=7.5,bbox_to_anchor=(0.02,0.42))
fig.savefig('figures/figure_tradeoff_trained.pdf'); plt.close(fig)
print("figures written")
