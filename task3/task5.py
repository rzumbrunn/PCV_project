# %%
import sys
sys.path.append('../')
sys.path.append('./PCV_project')
from plot_config import *

import numpy as np

def euler_sim(M, x0 ,dt, T, Perturbation=None):
    t_range = np.linspace(0, T, int(T/dt))[:-1]
    X = np.zeros((len(t_range)+1, 3), dtype=np.complex128)
    X[0] = x0
    # If statements outside the loop for performance!
    if Perturbation is not None:
        for i, t in enumerate(t_range):
            X[i+1] = X[i] + 1j * (M+Perturbation(t)) @ X[i] *dt
    else:
        for i, t in enumerate(t_range):
            X[i+1] = X[i] + 1j*dt*M @ X[i]

    return X

def rk4_sim(M, x0, dt, T, Perturbation=None):

    t_range = np.linspace(0, T, int(T/dt))[:-1]
    X = np.zeros((len(t_range)+1, 3), dtype=np.complex128)
    X[0] = x0
    if Perturbation is not None:
        for i, t in enumerate(t_range):
            k1 = 1j * (M+Perturbation(t)) @ X[i]
            k2 = 1j * (M+Perturbation(t+dt/2)) @ (X[i] + k1*dt/2)
            k3 = 1j * (M+Perturbation(t+dt/2)) @ (X[i] + k2*dt/2)
            k4 = 1j * (M+Perturbation(t+dt)) @ (X[i] + k3*dt)
            X[i+1] = X[i] + (k1 + 2*k2 + 2*k3 + k4)*dt/6
    else:
        for i, t in enumerate(t_range):
            k1 = 1j * M @ X[i]
            k2 = 1j * M @ (X[i] + k1*dt/2)
            k3 = 1j * M @ (X[i] + k2*dt/2)
            k4 = 1j * M @ (X[i] + k3*dt)
            X[i+1] = X[i] + (k1 + 2*k2 + 2*k3 + k4)*dt/6

    return X

def implicit_euler_sim(M,x0,dt , T, Perturbation = None):

    t_range = np.linspace(0, T, int(T/dt))[:-1]
    X = np.zeros((len(t_range)+1, 3), dtype=np.complex128)
    X[0] = x0
    if Perturbation is not None:
        for i, t in enumerate(t_range):
            X[i+1] = np.linalg.solve(np.eye(M.shape[0])-1j*dt*(M+Perturbation(t)), X[i])
    else:
        for i, t in enumerate(t_range):
            X[i+1] = np.linalg.solve(np.eye(M.shape[0])-1j*dt*M, X[i])


    return X

# %%
#constants
from scipy.constants import physical_constants as pc
aut = pc['atomic unit of time'][0]
E = 200*1e-6 / pc['atomic unit of energy'][0] #uJ


tau_p = 5*1e-9 / aut #ns
t_0 = 10*1e-9 / aut #ns
w_0 = 0.5*1e-3 / pc['atomic unit of length'][0] #mm

mu_i_phi = np.sqrt(0.1)*0.15324673 # au
mu_i_psi = np.sqrt(0.2)*0.09692173 #au

E_i_phi = 5000*1e2 * pc["atomic unit of length"][0]  #cm^-1
E_i_psi = 5000*1e2 * pc["atomic unit of length"][0] #cm^-1
Gamma_diss = 25*1e2 * pc["atomic unit of length"][0] #cm^-1

#


# %%
def intensity(t):
    return 2*E/(tau_p * np.pi* w_0**2)*2**(-4*((t-t_0)/(tau_p))**2)

def Omega_if(t, mu):
    return mu*np.sqrt(intensity(t))

# %%
import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([1, 0, 0])
dt = 0.000001e-9/aut
T = 50e-9/aut

def calc_dissoc_yield(omega, q):
    E_i = -omega
    E_phi = E_i_phi - omega
    E_psi = E_i_psi - omega + 1j* Gamma_diss/2
    g_phipsi = 0

    #laser
    laser = lambda t: np.array(
        [
            [0, np.conjugate(Omega_if(t,mu_i_phi)), np.conjugate(Omega_if(t,mu_i_psi))],
            [Omega_if(t,mu_i_phi), 0, 0],
            [Omega_if(t,mu_i_psi), 0, 0],
        ]
    )

    M = np.array(
        [
            [E_i, 0,0],
            [0, E_phi, np.conjugate(g_phipsi)],
            [0, g_phipsi, E_psi],
        ]
    )

    Perturbation = laser

    X = rk4_sim(M, x0, dt, T, Perturbation)

    q.put(X)


# %%
from multiprocessing import Process, Queue
import os
PROJECT_DIR = os.path.abspath('')
X_cache_dir = os.path.join('/scratch/islerd/x_cache')
os.makedirs(X_cache_dir, exist_ok=True)

threads = []
omega_range = np.linspace(-1.25*E_i_phi, 1.25*E_i_phi, 66)
# Dictionary with omega_range values as keys
dissoc_yields = {omega: None for omega in omega_range}
for omega in omega_range:
    q = Queue()
    p = Process(target=calc_dissoc_yield, args=(omega,q,))
    threads.append((p, q, omega))

num_threads_max = 22
num_threads_running = 0
running_threads = []


first = True
while len(threads) > 0:
    if num_threads_running >= num_threads_max or len(threads) == 0:
        # Wait until a thread finishes
        for thread in running_threads:
            # dissoc_yield = thread[1].get()
            X = thread[1].get()
            # Store X to disk
            np.savetxt(os.path.join(X_cache_dir, f'X_{str(thread[2]*1e10)}_dt_{round(dt)}.txt'), X)
            dissoc_yield = 1 - np.max(np.abs(X[-10000:-1,0])**2)
            print(f'Saved for omega={thread[2]}')
            thread[0].join()
            dissoc_yields[thread[2]] = dissoc_yield
            running_threads.remove(thread)
            num_threads_running -= 1
    else:
        for i in range(num_threads_max - num_threads_running):
            if len(threads) == 0:
                break
            thread = threads.pop(0)
            thread[0].start()
            print(f"Thread for omega={thread[2]} started")
            num_threads_running += 1
            running_threads.append(thread)

# Wait for the remaining threads to finish
for thread in running_threads:
    X = thread[1].get()
    # Store X to disk
    np.savetxt(os.path.join(X_cache_dir, f'X_{str(thread[2]*1e10)}_dt_{round(dt)}.txt'), X)
    dissoc_yield = 1 - np.max(np.abs(X[-10000:-1,0])**2)
    print(f'Saved for omega={thread[2]}')
    thread[0].join()
    dissoc_yields[thread[2]] = dissoc_yield

# %%
# Plot the results
plt.plot(omega_range, list(dissoc_yields.values()))
plt.savefig(os.path.join(PROJECT_DIR, 'fano.png'))
print('Finished successfully')
