from numpy import *
from numpy.linalg import *
from matplotlib.pyplot import *
def v(x):
    return stack([-x[0]-2*x[1]**2, x[0]*x[1]-x[1]**3])
#def v(x):
#    r = x[0]**2 + x[1]**2
#    return stack([x[1]+x[0]*r, - x[0] + x[1]*r])
#b = -1
#def v(x):
#    return stack([x[1], -x[0] + b*x[1] - b*x[0]*x[0]*x[1]])
ngr = 50
gr = linspace(-0.5, 0.5, ngr)
x_gr, y_gr = meshgrid(gr, gr, indexing='xy')
x = stack([x_gr, y_gr]).reshape(2,-1)
n = x.shape[1]
eps = 1.e-5
x_p_dx = stack([x[0] + eps, x[1]])
x_m_dx = stack([x[0] - eps, x[1]])
x_p_dy = stack([x[0], x[1] + eps])
x_m_dy = stack([x[0], x[1] - eps])
dv = zeros((2, 2, n))
dv[0] = (v(x_p_dx) - v(x_m_dx))/(2*eps)
dv[1] = (v(x_p_dy) - v(x_m_dy))/(2*eps)
dv = transpose(dv, (2,0,1))
eig_jac = zeros((n, 2))
for i in arange(n):
    eig_jac[i] = eigvals(dv[i])

eig_jac_gr = reshape(eig_jac.T, (ngr, ngr, 2))
fig, ax = subplots()
cax = ax.contourf(x_gr, y_gr, eig_jac_gr[:,:,0],levels=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
cbar = colorbar(cax, ax=ax)
cbar.ax.tick_params(labelsize=24)
ax.grid(True)
tight_layout()
show()

#print(dv)