# %%
import sys
sys.path.append('../')
sys.path.append('./PCV_project')
from plot_config import *

import numpy as np
def euler_sim(M, x0 ,dt, T, Perturbation=None):
    x = x0
    X = []
    for t in np.linspace(0, T, int(T/dt)):
        if Perturbation is not None:
            x = x + 1j * (M+Perturbation(t)) @ x *dt
        else:
            x = x + 1j*dt*M @ x
        X.append(x)

    return np.array(X, dtype=np.complex128)

def rk4_sim(M, x0, dt, T, Perturbation=None):

    x = x0
    X = []
    for t in np.linspace(0, T, int(T/dt)):
        if Perturbation is not None:
            k1 = 1j * (M+Perturbation(t)) @ x
            k2 = 1j * (M+Perturbation(t+dt/2)) @ (x + k1*dt/2)
            k3 = 1j * (M+Perturbation(t+dt/2)) @ (x + k2*dt/2)
            k4 = 1j * (M+Perturbation(t+dt)) @ (x + k3*dt)
        else:
            k1 = 1j * M @ x
            k2 = 1j * M @ (x + k1*dt/2)
            k3 = 1j * M @ (x + k2*dt/2)
            k4 = 1j * M @ (x + k3*dt)
        x = x + (k1 + 2*k2 + 2*k3 + k4)*dt/6
        X.append(x)

    return np.array(X, dtype=np.complex128)

def implicit_euler_sim(M,x0,dt , T, Perturbation = None):
    x = x0
    X = []
    for t in np.linspace(0, T, int(T/dt)):
        if Perturbation is not None:
            x = np.linalg.solve(np.eye(M.shape[0])-1j*dt*(M+Perturbation(t)), x)
        else:
            x = np.linalg.solve(np.eye(M.shape[0])-1j*dt*M, x)
        X.append(x)


    return np.array(X, dtype=np.complex128)

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

# %%

x0 = np.array([1, 0, 0])
dt = 0.000005e-9/aut
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
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
X_cache_dir = os.path.join('/scratch/islerd/x_cache')
os.makedirs(X_cache_dir, exist_ok=True)

threads = []
omega_range = np.linspace(-1.25*E_i_phi, 1.25*E_i_phi, 60)
# Dictionary with omega_range values as keys
dissoc_yields = {omega: None for omega in omega_range}
for omega in omega_range:
    q = Queue()
    p = Process(target=calc_dissoc_yield, args=(omega,q,))
    threads.append((p, q, omega))

num_threads_max = 20
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
