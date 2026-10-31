exec(open('01_network.py').read())
from scipy.optimize import least_squares
rng=np.random.default_rng(260613)
g0=rng.uniform(0.75,1.25,size=M)
fA=forcing(24); CA=obs([4]); fB=forcing(20); CB=obs([4,12])
yA_t=1.02*y(g0,CA,fA)
LS=dict(bounds=(GMIN,GMAX),xtol=1e-15,ftol=1e-15,gtol=1e-15)
gA=least_squares(lambda g:y(g,CA,fA)-yA_t,g0,jac=lambda g:J(g,CA,fA),**LS).x
JA=J(gA,CA,fA); LA0=loss(gA,CA,fA,yA_t); PiA=np.eye(M)-np.linalg.pinv(JA)@JA
dyB=np.array([-0.03,0.04])*y(gA,CB,fB)
