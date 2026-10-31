import numpy as np
n=5; N=n*n
def idx(r,c): return r*n+c
edges=[]
for r in range(n):
    for c in range(n):
        if c+1<n: edges.append((idx(r,c), idx(r,c+1)))
        if r+1<n: edges.append((idx(r,c), idx(r+1,c)))
M=len(edges)
B=np.zeros((N,M))
for k,(i,j) in enumerate(edges): B[i,k]=1.0; B[j,k]=-1.0
GROUND=[0]; free=[i for i in range(N) if i not in GROUND]; Nf=len(free)
Bf=B[free,:]
def pos(node): return free.index(node)
def Lap(g): return Bf@np.diag(g)@Bf.T
def forcing(node):
    f=np.zeros(Nf); f[pos(node)]=1.0; return f
def obs(nodes):
    C=np.zeros((len(nodes),Nf))
    for r,nd in enumerate(nodes): C[r,pos(nd)]=1.0
    return C
def y(g,C,f): return C@np.linalg.solve(Lap(g),f)
def J(g,C,f):
    L=Lap(g); u=np.linalg.solve(L,f)
    return -(C@np.linalg.solve(L,Bf))*(Bf.T@u)
def loss(g,C,f,yt): r=y(g,C,f)-yt; return 0.5*float(r@r)
def grad(g,C,f,yt): return J(g,C,f).T@(y(g,C,f)-yt)
GMIN,GMAX=0.35,2.50
