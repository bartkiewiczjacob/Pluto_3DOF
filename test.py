import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution as opt
from scipy.optimize import NonlinearConstraint
import ambiance as amb
import pymap3d as pm
plt.style.use('seaborn-poster')
lato = 0.1
lono = 0.1
alto = 0.0
payload = 0
wind_vel = [0,0,0]
S = np.square(0.7692)*np.pi/4
def ca(M):
    cd_ref = [0.3,0.26,0.4,0.55,0.47,0.36,0.28,0.24,0.21,0.20,0.20,0.21,0.22,0.23,0.24,0.25]
    M_ref = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5]
    f = interp1d(M_ref,cd_ref,bounds_error=False,fill_value=0.25)
    ca = f(M)
    return float(ca)
def hit_ground(t,y,omegas):
    lat, lon, alt = pm.enu2geodetic(y[0], y[1], y[2], lato, lono, alto)
    return alt
def reach_apogee(t,y,omegas):
    return y[5]
hit_ground.terminal = True
hit_ground.direction = -1
reach_apogee.terminal = True
reach_apogee.direction = -1
def thrust_func(t,P):
    if t < 150:
        thrust = [26245,0,0]
    else:
        thrust = [0,0,0]
    return thrust
def mass_func(t):
    if t < 150:
        m_dot = [-8.8]
    else:
        m_dot = [0]
    return m_dot
def aero_func(M):
    cax = ca(M)
    coeffs = [cax,1.0,1.0]
    return coeffs
def dynamics(t,x,omegas):
    g = 9.8
    R = np.array([[np.cos(x[7])*np.cos(x[8]), -np.cos(x[7])*np.sin(x[8]), np.sin(x[7])],
              [np.cos(x[6])*np.sin(x[8])+np.cos(x[8])*np.sin(x[6])*np.sin(x[7]), np.cos(x[6])*np.cos(x[8])-np.sin(x[6])*np.sin(x[7])*np.sin(x[8]),-np.cos(x[7])*np.sin(x[6])],
              [np.sin(x[6])*np.sin(x[8])-np.cos(x[6])*np.cos(x[8])*np.sin(x[7]), np.cos(x[8])*np.sin(x[6])+np.cos(x[6])*np.sin(x[7])*np.sin(x[8]),np.cos(x[6])*np.cos(x[7])]])
    m_dot = mass_func(t)
    v = [x[3], x[4], x[5]]
    lat,lon,alt = pm.enu2geodetic(x[0],x[1],x[2],lato,lono,alto)
    if (alt < 80000)&(alt > 0):
        atmos = amb.Atmosphere(alt)
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
    x_dot = np.concatenate([v,v_dot,omegas,m_dot])
    return x_dot
def propagate(lato,lono,alto,payload,wind_vel,S,omegas,t_max):
    t_eval = np.arange(0, t_max, 0.1)
    xo = [0, 0, 0, 0, 0, 0, 0, np.pi / 2, 0, 1860 + payload]
    sol = solve_ivp(dynamics, [0, t_max], xo, t_eval=t_eval, events=[hit_ground,reach_apogee],args=[omegas])
    return sol
def obj_func(x):
    sol = propagate(lato,lono,alto,100,wind_vel,S,[0,x[0],0],1000)
    return -sol.y[3,-1]
def final_alt(x):
    sol = propagate(lato,lono,alto,100,wind_vel,S,[0,x[0],0],1000)
    return sol.y[2,-1]
nlc = NonlinearConstraint(final_alt,100000,100001)
vo = obj_func([-0.01])
res = opt(obj_func,[(-0.1,0)],constraints=nlc,disp=True,atol=1E-6,polish=False,x0=-0.00675089,popsize=15,maxiter=100)
print(res.x)
sol = propagate(lato,lono,alto,100,wind_vel,S,[0,res.x[0],0],1000)
lat,lon,alt = pm.enu2geodetic(sol.y[0],sol.y[1],sol.y[2],lato,lono,alto)
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111)
plt.plot(sol.y[0],alt)
plt.xlabel('Range')
plt.ylabel('Altitude')
ax.set_aspect('equal', adjustable='box')
plt.grid()
plt.show()

