from numpy import *
from matplotlib.pyplot import *
from matplotlib import cm 
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, BoundaryNorm
theta = linspace(0, 2*pi, 500)
theta, phi = meshgrid(theta, theta, indexing='xy')
r, ra = 10, 1
cp = cos(phi)
sp = sin(phi)
ct = cos(theta)
st = sin(theta)
x = (r + ra*cp)*ct
y = (r + ra*cp)*st
z = ra*sp
# Plot T^2 embedded in R^3
fig = figure()
ax = fig.add_subplot(projection='3d')
ax.contourf(x, y, z, cmap='gray', alpha=0.5, levels=30)
ax.grid(True)
ax.axis('equal')
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.zaxis.set_tick_params(labelsize=24)
ax.set_xlabel('x',fontsize=24)
ax.set_ylabel('y',fontsize=24)
ax.set_zlabel('z',fontsize=24)
tight_layout()

# Make 2D figures
fig2 = figure()
ax2 = fig2.add_subplot()
ax2.contourf(x, y, zeros_like(x), cmap='gray', alpha=0.5, levels=30)
ax2.grid(True)
ax2.axis('equal')
ax2.xaxis.set_tick_params(labelsize=24)
ax2.yaxis.set_tick_params(labelsize=24)
ax2.set_xlabel('x',fontsize=24)
ax2.set_ylabel('y',fontsize=24)


def change_coords(w):
    t, p = w
    rpracp = r + ra*cos(p)
    return stack([rpracp*cos(t), rpracp*sin(t), ra*sin(p)])


def change_tangent_coords(w, v):
    t = w[0]
    p = w[1]
    st, ct = [sin(t), cos(t)]
    sp, cp = [sin(p), cos(p)]
    rpracp = r + ra*cp
    dxdt = -rpracp*st
    dydt = rpracp*ct
    dxdp = -ra*sp*ct
    dydp = -ra*sp*st
    dzdp = ra*cp
    return stack([v[0]*dxdt + v[1]*dxdp, v[0]*dydt + v[1]*dydp, v[1]*dzdp])

def grad_x(w):
    t, p = w
    # x = (r + ra*cp)*ct
    cp = cos(p)
    st = sin(t)
    rpracp = r + ra*cp
    return stack([-rpracp*st, -ra*sin(p)*cos(t)])


def grad_desc(w, eta):
    return w - eta*grad_x(w)


n_o = 100
n_time = 500
t = 2*pi*random.rand(n_o)
p = 2*pi*random.rand(n_o)
w = stack([t, p])
eta = 1.e-1
colors = colormaps.get_cmap('viridis')
norm = Normalize(vmin=0, vmax=n_time)  # Normalize the data
sm = cm.ScalarMappable(cmap=colors, norm=norm)
sm.set_array([])
#cbar = fig.colorbar(sm, ax=ax)
w_orbit = zeros((n_time, 2, n_o)) 
x = change_coords(w)
#ax.plot(x[0], x[1], x[2], 'P', ms=10, color='k')
ax2.plot(x[0], x[1], 'P', ms=10, color='k')

for n in range(1, n_time):
    w_orbit[n] = w
    w = grad_desc(w, eta)
    x = change_coords(w)
    #ax.plot(x[0], x[1], x[2], ".", color=colors(n))
    ax2.plot(x[0], x[1], ".", color=colors(n))

#cbar.set_label('Time', fontsize=20)
#cbar.ax.tick_params(labelsize=20)
cbar2 = fig.colorbar(sm, ax=ax2)
cbar2.set_label('Time', fontsize=20)
cbar2.ax.tick_params(labelsize=20)
tight_layout()



# Plot the gradient field in 2D
# interpolate and plot contours
fig_grad_1 = figure()
ax_grad_1 = fig_grad_1.add_subplot()
ngr = 1000
t_grid, p_grid = meshgrid(linspace(0,2*pi,ngr), linspace(0, 2*pi,ngr), indexing='xy')
gradx_grid = grad_x(stack([t_grid.flatten(), p_grid.flatten()])).T.reshape(ngr, ngr, 2)
xcomp = ax_grad_1.contourf(t_grid, p_grid, gradx_grid[:,:,0])
cbar_gxx = fig_grad_1.colorbar(xcomp, ax=ax_grad_1)
cbar_gxx.ax.tick_params(labelsize=24)

ax_grad_1.grid(True)
ax_grad_1.xaxis.set_tick_params(labelsize=24)
ax_grad_1.yaxis.set_tick_params(labelsize=24)
ax_grad_1.set_xlabel(r'$\theta$', fontsize=24)
ax_grad_1.set_ylabel(r'$\phi$', fontsize=24)
tight_layout()


fig_grad_2 = figure()
ax_grad_2 = fig_grad_2.add_subplot()
ycomp = ax_grad_2.contourf(t_grid, p_grid, gradx_grid[:,:,1])
cbar_gxx = fig_grad_2.colorbar(xcomp, ax=ax_grad_2)
cbar_gxx.ax.tick_params(labelsize=24)

ax_grad_2.grid(True)
ax_grad_2.xaxis.set_tick_params(labelsize=24)
ax_grad_2.yaxis.set_tick_params(labelsize=24)
ax_grad_2.set_xlabel(r'$\theta$', fontsize=24)
ax_grad_2.set_ylabel(r'$\phi$', fontsize=24)
tight_layout()

# Plot critical points, which are omega-limit sets
fig_crit = figure()
ax_crit = fig_crit.add_subplot()
colors = ['white', 'gray', 'black']
cmap = ListedColormap(colors)

# Define the boundaries for the discrete values
bounds = [0, 1, 2]  # Define the boundaries for each color
norm = BoundaryNorm(bounds, cmap.N)

close_to_critical1 = (abs(gradx_grid[:,:,0]) < 1.e-6)
close_to_critical2 = (abs(gradx_grid[:,:,1]) < 1.e-6)
close_to_critical = close_to_critical1.astype(int) + close_to_critical2.astype(int)
ccrit = ax_crit.contourf(t_grid, p_grid, close_to_critical, cmap=cmap, norm=norm)
ax_crit.plot(t_grid[close_to_critical1].flatten(), p_grid[close_to_critical1].flatten(), "P", color="k", ms=10)
cbar = colorbar(ccrit, ticks=[0,1,2], ax=ax_crit)
ax_crit.set_title('Critical points of x', fontsize=24)
cbar.ax.tick_params(labelsize=24)
show()

