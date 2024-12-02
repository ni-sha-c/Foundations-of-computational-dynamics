from numpy import *
s, r, b  = 10, 28, 8/3
dt = 0.005
def lorenz(p):
    x, y, z = p
    return stack([s*(y-x), x*(r-z) - y, x*y - b*z])
def step(p):
    return p + dt*lorenz(p)
def dstep(p):
    n, d = p.shape
    jac = zeros((n, d, d))
    for i in range(n):
        x, y, z = p[i]
        jac[i] = eye(d) + dt*stack([[-s, s, 0], [r-z, -1, -x], [y, x, -b]])
    return jac
def simulate(x0, n):
    orbit = zeros((n, 3))
    orbit[0] = x0
    for i in range(n-1):
        orbit[i+1] = step(orbit[i])
    return orbit



def blv(jac, dim):
    n = jac.shape[0]
    v0 = zeros((n, 3, dim))
    v0[0] = random.rand(3,dim)
    les = zeros(dim)
    for i in range(n-1):
        v0[i+1] = jac[i] @ v0[i]
        v0[i+1], r = linalg.qr(v0[i+1])
        les += log(abs(diag(r)))/(n*dt)
    return v0, les 

n = 10000
dim = 3
orbit = simulate(random.rand(3), 10000)
jac = dstep(orbit)
#rev_jac = zeros()
v0, les = blv(jac, 3)


