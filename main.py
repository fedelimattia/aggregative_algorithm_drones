import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from utils import generate_connected_graph, generate_intruders_and_target, sigma, nabla_1li, nabla_2li, li, J, eq_state, update_drone_state, prepare_vis_vectors
from params import *
from drone import Drone
from visualization import animate_drones

# Intruders positions
r, r_0 = generate_intruders_and_target(max_iters, N, 3, min_dist=1, speed = 0.3)


# Fix their value as the final one:
if static_intruders:
    for t in range(max_iters):
        r_0[t, :] = r_0[max_iters-1, :]
        for i in range(N):
            r[t, i, :] = r[max_iters-1, i, :]


# Variables for the algorithm
x = np.zeros((max_iters, N, ni))
x[0, :, :] = x0

s = np.zeros((max_iters, N, ni))
s[0, :, :] = x0

v = np.zeros((max_iters, N, ni))
for i in range(N):
    v[0, i, :] = nabla_2li(x[0, :, :], i, s[0, i, :], r_0[0, :])

# Communication graph (connected, double stochastic, complete)
G, _, _, A = generate_connected_graph(N, p = 1)

# Total cost of the algorithm
cost = np.zeros(max_iters)

drones = [Drone(dt) for j in range(2*N + 1)]
x_star = np.zeros((max_iters, 2*N + 1, nx))
x_des = np.zeros((max_iters, 2*N + 1, nx))
thrusts = np.zeros((max_iters, 2*N + 1, m))

# Initialize the initial states
for j, drone in enumerate(drones):
    if j < N:
        x_star[0, j, :] = eq_state(x0[j, :], nx)
        drone.reset_state(x_star[0, j, :])
    elif j < 2*N:
        x_star[0, j, :] = eq_state(r[0, j-N, :], nx)
        drone.reset_state(x_star[0, j, :])
    else:
        x_star[0, j, :] = eq_state(r_0[0, :], nx)
        drone.reset_state(x_star[0, j, :])

# Aggregative algorithm
for k in range(max_iters-1):

    # STEP 1 #########################################
    # Update the positions of the intruders and target
    opponents_des = r[k+1, :, :]
    opponents = opponents_des
    target_des = r_0[k+1, :]
    target = target_des
    
    # Opponents (j from 0 to N-1)
    for j, drone in enumerate(drones[N:2*N]):
        if opp_targ_dynamics:
            xdes = eq_state(opponents_des[j, :], nx)
            x_des[k+1, N+j, :] = xdes
            x_star[k+1, N+j, :], thrusts[k, N+j, :] = update_drone_state(drone, xdes, dt)
            opponents[j, :] = x_star[k+1, N+j, :3]
        else:
            x_star[k+1, N+j, :] = eq_state(opponents_des[j, :], nx)

    # Target (single drone at index 2N)
    if opp_targ_dynamics:
        drone = drones[2*N]
        xdes = eq_state(target_des, nx)
        x_des[k+1, 2*N, :] = xdes
        x_star[k+1, 2*N, :], thrusts[k, 2*N, :] = update_drone_state(drone, xdes, dt)
        target = x_star[k+1, 2*N, :3]
    else:
        x_star[k+1, 2*N, :] = eq_state(target_des, nx)

            
    # STEP 2 #########################################
    # Local Gradient method using the derivative of li with s that estimates the barycenter locally
    for i in range(N):
        x[k+1, i, :] = x[k, i, :] - alpha * (nabla_1li(x[k, :, :], opponents, i, s[k, i, :], A, target) + v[k, i, :])

        # Update the estimates of s and v by adding the tracking error terms for each agent
        # First do the consensus step
        for j in range(N):
            s[k+1, i, :] += A[i, j] * s[k, j, :] 
            v[k+1, i, :] += A[i, j] * v[k, j, :]
        
        # Then update the estimates of s and v with the tracking error terms
        s[k+1, i, :] += x[k+1, i, :] - x[k, i, :]
        v[k+1, i, :] += nabla_2li(x[k+1, :, :], i, s[k+1, i, :], x_star[k+1, -1, :3]) - nabla_2li(x[k, :, :], i, s[k, i, :], x_star[k, -1, :3])

    # STEP 3 #########################################
    # Tracking the positions of the defending drones 
    for j, drone in enumerate(drones[:N]):
        # Tracking the optimal value computed by the algorithm
        xdes = eq_state(x[k+1, j, :], nx)
        x_des[k+1, j, :] = xdes
        
        x_star[k+1, j, :], thrusts[k, j, :] = update_drone_state(drone, xdes, dt)
        
    # STEP 4 #########################################
    # Update the cost
    cost[k] = J(x[k, :, :], opponents, target, A)
    print("Iteration:", k, "Cost:", cost[k])

    # STEP 5 #########################################
    # Check for convergence
    if (np.linalg.norm(x_star[k, :N, :3] - x[k, :, :]) < 1e-3) and k > 1:
        print("Converged at iteration", k)
        x_star = x_star[:k+1, :, :]
        x_des = x_des[:k+1, :, :]
        thrusts = thrusts[:k+1, :, :]

        break

x_des[:, N:, :] = x_star[:, N:, :]
animate_drones(x_star = x_star, xdes=x_des, thrusts = thrusts, 
               show_target=True,dt = dt, stop_time = stop_time, 
               colors = ["b"] * (N) + ["r"] * (N) + ["green"])
