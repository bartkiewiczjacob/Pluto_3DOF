import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution as opt
from scipy.optimize import NonlinearConstraint
from ambiance import Atmosphere
from pymap3d import enu2geodetic
plt.style.use('seaborn-poster')
lato = 0.1
lono = 0.1
alto = 0.0
burn_time = 120
wind_vel = [0,0,0]
S = np.square(0.7692)*np.pi/4
payload = 544
def ca(M):
    cd_ref = [0.3,0.26,0.4,0.55,0.47,0.36,0.28,0.24,0.21,0.20,0.20,0.21,0.22,0.23,0.24,0.25]
    M_ref = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5]
    f = interp1d(M_ref,cd_ref,bounds_error=False,fill_value=0.25)
    cax = f(M)
    return float(cax)
def hit_ground(t,y,omegas):
    lat, lon, alt = enu2geodetic(y[0], y[1], y[2], lato, lono, alto)
    return alt
def reach_apogee(t,y,omegas):
    return y[5]
hit_ground.terminal = True
hit_ground.direction = -1
reach_apogee.terminal = True
reach_apogee.direction = -1
def thrust_func(t,P):
    if t < burn_time:
        thrust = [26245,0,0]
    else:
        thrust = [0,0,0]
    return thrust
def mass_func(t):
    if t < burn_time:
        m_dot = [-8.8]
    else:
        m_dot = [0]
    return m_dot
def aero_func(M):
    cax = ca(M)
    coeffs = [cax,1.0,1.0]
    return coeffs
def omegas(t,opts):
    if t < 37.5:
        omegas = [0,opts[0],0]
    elif t < 75:
        omegas = [0,opts[1],0]
    elif t < 112.5:
        omegas = [0,opts[2],0]
    else:
        omegas = [0,opts[3],0]
    return omegas
def dynamics(t,x,opts):
    g = 9.8
    R = np.array([[np.cos(x[7])*np.cos(x[8]), -np.cos(x[7])*np.sin(x[8]), np.sin(x[7])],
              [np.cos(x[6])*np.sin(x[8])+np.cos(x[8])*np.sin(x[6])*np.sin(x[7]), np.cos(x[6])*np.cos(x[8])-np.sin(x[6])*np.sin(x[7])*np.sin(x[8]),-np.cos(x[7])*np.sin(x[6])],
              [np.sin(x[6])*np.sin(x[8])-np.cos(x[6])*np.cos(x[8])*np.sin(x[7]), np.cos(x[8])*np.sin(x[6])+np.cos(x[6])*np.sin(x[7])*np.sin(x[8]),np.cos(x[6])*np.cos(x[7])]])
    m_dot = mass_func(t)
    rates = omegas(t,opts)
    v = [x[3], x[4], x[5]]
    lat,lon,alt = enu2geodetic(x[0], x[1], x[2], lato, lono, alto)
    if (alt < 80000)&(alt > 0):
        atmos = Atmosphere(alt)
        rho = atmos.density
        P = atmos.pressure
        a = atmos.speed_of_sound
    else:
        rho = 0
        P = 0
        a = 282
    v_wind = np.subtract(wind_vel,v)
    V_inf = np.linalg.norm(v_wind,2)
    M = V_inf/a
    coeffs = aero_func(M)
    v_body = np.matmul(R,np.reshape(v_wind,(3,1))).reshape(3)
    qbar = 0.5*rho*np.square(v_body)*np.sign(v_body)
    force_drag = np.multiply(coeffs,qbar*S)
    force_thrust = thrust_func(t,P)
    force_gravity = np.matmul(R,np.reshape([0,0,-g*x[9]],(3,1))).reshape(3)
    forces = force_drag + force_thrust + force_gravity
    a = np.multiply(forces, 1 / x[9])
    v_dot = np.matmul(a,R)
    x_dot = np.concatenate([v,v_dot,rates,m_dot])
    return x_dot
def propagate(lato,lono,alto,payload,wind_vel,S,opts,t_max):
    t_eval = np.arange(0,t_max,0.1)
    xo = [0, 0, 0, 0, 0, 0, 0, np.pi / 2, 0, 8.8*burn_time + 450 + payload]
    sol = solve_ivp(dynamics, [0, t_max], xo, t_eval = t_eval,events=[hit_ground],args=[opts])
    return sol
def obj_func(x):
    sol = propagate(lato,lono,alto,payload,wind_vel,S,x,burn_time)
    return -sol.y[3,-1]
def final_alt(x):
    sol = propagate(lato,lono,alto,payload,wind_vel,S,x,burn_time)
    lat, lon, alt = enu2geodetic(sol.y[0], sol.y[1], sol.y[2], lato, lono, alto)
    return alt[-1]
nlc = NonlinearConstraint(final_alt,30000,31000)
rate_o = [-0.00392592, -0.00274876, -0.02599948, -0.05255883]
#rate_o = [0,0,0,0]
res = opt(obj_func,[(-0.1,0),(-0.1,0),(-0.1,0),(-0.1,0)],constraints=nlc,disp=True,atol=1E-6,polish=False,x0=rate_o,popsize=15,maxiter=25)
print(res.x)
rate_o = res.x
sol_o = propagate(lato,lono,alto,payload,wind_vel,S,rate_o,1000)
lat_o,lon_o,alt_o = enu2geodetic(sol_o.y[0], sol_o.y[1], sol_o.y[2], lato, lono, alto)
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111)
#plt.plot(sol_o.y[0],alt_o,label='Original')
vel = np.sqrt(np.square(sol_o.y[3])+np.square(sol_o.y[4]))
inds = np.arange(0,len(alt_o))
M = np.zeros(len(alt_o))
for i in inds:
    atmos = Atmosphere(alt_o[i])
    M[i] = vel[i]/atmos.speed_of_sound
plt.plot(sol_o.t,M,label='Optimized')
plt.xlabel('time[s]')
plt.ylabel('Mach')
plt.title('Burn Time = {}s, Payload = {}kg'.format(burn_time,payload))
#ax.set_aspect('equal', adjustable='box')
#ax.legend()
major_ticks = np.arange(0,max(sol_o.t),10)
minor_ticks = np.arange(0,max(sol_o.t),1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
plt.show()
