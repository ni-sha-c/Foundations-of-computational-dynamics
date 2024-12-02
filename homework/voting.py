from numpy import *
from scipy.stats import norm
from matplotlib.pyplot import *
m1, m2, s1, s2 = -1, 1, sqrt(0.5), sqrt(0.5)
o_l, o_r = 1, 1
# LJ potential
eps = 1.0
sigma = 1
def lj(r):
    return 48*eps*(sigma**12/r**13 - 0.5*sigma**6/r**7)


sqrtpi_inv = 1/sqrt(pi)
sqrt2pi_inv = sqrtpi_inv/sqrt(2)
dt = 0.001
def share_left(x):
    return 0.5*norm.cdf((x-m1)/s1) + \
    0.5*norm.cdf((x-m2)/s2)
def step(x):
    l, r, a, b = x
    y = (l + r)/2
    dshare_left = 1/s1*norm.pdf((y-m1)/2)/2 + \
    1/s2*norm.pdf((y-m2)/s2)/2
    dsl_dl = 0.5*dshare_left
    dsr_dr = -0.5*dshare_left
    F = lj(abs(r-l))
    a = dt*F
    b = -dt*F

    l = l + dt*o_l*dsl_dl + 0.5*dt*dt*a
    r = r + dt*o_r*dsr_dr + 0.5*dt*dt*b 

    an = lj(abs(r-l))
    bn = -an

    a = a + dt*(a + an)/2
    b = b + dt*(b + bn)/2

    l = l % (m1 - s1)
    l = l % (m2 + s2)
    r = r % (m2 + s2)
    r = r % (m1 - s1)
    return stack([l, r, a, b])
def simulate_visualize():
    ns = 5
    x0 = m1 + (m2-m1)*random.rand(4, ns)
    x0[2] = 0.0
    x0[3] = 0.0
    nt = 10000
    orbit = zeros((nt, 4, ns))
    orbit[0] = x0
    print("At time 0, Vote share of l:", share_left((x0[0] + x0[1])/2))
    for i in range(nt-1):
        orbit[i+1] = step(orbit[i])
    orbit = orbit.T
    fig = figure()
    ax = fig.add_subplot()
    
    for i in range(ns):
        ax.plot(x0[0,i], x0[1,i], "P", color="k", ms=5.0)
        ax.plot(orbit[i,0,:], orbit[i,1,:], lw=3.0)
    print("At time 1, l:", orbit[:,0,-1])
    print("At time 1, r:", orbit[:,1,-1])
    print("Vote share of l:", share_left((orbit[:,0,-1] + orbit[:,1,-1])/2))
    ax.set_xlabel("Left pos", fontsize=24)
    ax.set_ylabel("Right pos", fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    tight_layout()
    show()
    return orbit





