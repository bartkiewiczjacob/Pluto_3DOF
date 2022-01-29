import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fmin_slsqp as opt
import ambiance as amb
import pymap3d as pm
plt.style.use('seaborn-poster')
lato = 0.0
lono = 0.0
alto = 0.0
payload = 1000
xo = [0,0,0,0,0,0,0,np.pi/2,0,1860+payload]
wind_vel = [0,0,0]
S = np.square(0.7692)*np.pi/4
def thrust_func(t,P):
    if t < 120:
        thrust = [26245,0,0]
    else:
        thrust = [0,0,0]
    return thrust
def mass_func(t):
    if t < 120:
        m_dot = [-8.8]
    else:
        m_dot = [0]
    return m_dot
def aero_func(M):
    coeffs = [0.3,1.0,1.0]
    return coeffs
def dynamics(t,x):
    g = 9.8
    R = np.array([[np.cos(x[7])*np.cos(x[8]), -np.cos(x[7])*np.sin(x[8]), np.sin(x[7])],
              [np.cos(x[6])*np.sin(x[8])+np.cos(x[8])*np.sin(x[6])*np.sin(x[7]), np.cos(x[6])*np.cos(x[8])-np.sin(x[6])*np.sin(x[7])*np.sin(x[8]),-np.cos(x[7])*np.sin(x[6])],
              [np.sin(x[6])*np.sin(x[8])-np.cos(x[6])*np.cos(x[8])*np.sin(x[7]), np.cos(x[8])*np.sin(x[6])+np.cos(x[6])*np.sin(x[7])*np.sin(x[8]),np.cos(x[6])*np.cos(x[7])]])
    if t < 75:
        omegas = [0,0,0]
    else:
        omegas = [0,0.00,0]
    m_dot = mass_func(t)
    v = [x[3], x[4], x[5]]
    lat,lon,alt = pm.enu2geodetic(x[0],x[1],x[2],lato,lono,alto)
    if alt < 80000:
        atmos = amb.Atmosphere(alt)
        rho = atmos.density
        P = atmos.pressure
        a = atmos.speed_of_sound
    else:
        rho = 0
        P = 0
        a = 10000
    v_wind = np.subtract(wind_vel,v)
    V_inf = np.linalg.norm(v_wind)
    M = V_inf/a
    coeffs = aero_func(M)
    v_body = np.matmul(R,np.reshape(v_wind,(3,1))).reshape(3)
    qbar = 0.5*rho*np.square(v_body)
    force_drag = np.multiply(coeffs,qbar*S)
    force_thrust = thrust_func(t,P)
    force_gravity = np.matmul(R,np.reshape([0,0,-g*x[9]],(3,1))).reshape(3)
    forces = force_drag + force_thrust + force_gravity
    a = np.multiply(forces, 1 / x[9])
    v_dot = np.matmul(a,R)
    x_dot = np.concatenate([v,v_dot,omegas,m_dot])
    return x_dot
t_eval = np.arange(0, 300, 0.1)
sol = solve_ivp(dynamics, [0, 300], xo, t_eval=t_eval)
lat,lon,alt = pm.enu2geodetic(sol.y[0],sol.y[1],sol.y[2],lato,lono,alto)
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111)
plt.plot(sol.t,alt)
plt.xlabel('Range')
plt.ylabel('Altitude')
#ax.set_aspect('equal', adjustable='box')
plt.show()

