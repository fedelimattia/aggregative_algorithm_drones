import numpy as np
from drone import Drone

# Simulation params
stop_time = 5
dt = 0.01
TT = int(stop_time/dt) + 1

# Population
N = 5
N_Target = 1

# number of dimensions
ni = 3      # number of directional states
nx = 12     # number of states
ns = nx
m = 4       # thrusts

# initial agents position
x0 = np.random.uniform(-1.5, 1.5, (N, ni))

# Weight for distances
gamma_x_r = 10
gamma_sigma_r0 = 5
gamma_x_sigma = 1
gamma_x_r0 = 15
gamma_barrier = 5
gamma_barrier_intruder = 5
gamma_barrier_target = 5

# Simulation parameters
max_iters = TT
alpha = dt 

# Intruders parameters
static_intruders = False
opp_targ_dynamics = True

