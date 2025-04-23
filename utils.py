import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_random_graph(N, p, double_stochastic=False):
    erG = nx.erdos_renyi_graph(N, p)
    Adj, WeightedAdj = get_matrices_from_graph(erG, double_stochastic=double_stochastic)
    return erG, Adj, WeightedAdj

def get_neighbors(G, node):
    return list(G.neighbors(node))

def generate_connected_graph(N, p = 0.5):
    """
    Generate a random graph with given number of nodes and probability of connection, ensuring
    that the graph is connected
    
    Parameters
    ----------
    N : int
        Number of nodes
    p : float
        Probability of connection between each pair of nodes
    """
    
    G = nx.erdos_renyi_graph(N, p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(N, p)
    
    Adj, WeightedAdj = get_matrices_from_graph(G, double_stochastic=False)
    _, WeightedAdj_doubly_stochastic = get_matrices_from_graph(G, double_stochastic=True)
    return G, Adj, WeightedAdj, WeightedAdj_doubly_stochastic

# At each iteration create a new graph with just 1 edge which cycles in between different nodes
def path_at_iteration(N, i):    
    """
    Create a path graph with just one edge in between node i to node i+1 which cycles in between different nodes

    Parameters
    ----------
    N : int
        Number of nodes
    i : int
        Index of the edge to be added

    Returns
    -------
    G : networkx.Graph
        A path graph with just one edge which cycles in between different nodes
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edge(i, i+1)

    return G


def plot_graph(G):
    """
    Plot a graph using NetworkX and Matplotlib.

    Parameters
    ----------
    G : networkx.Graph
        The graph to be plotted.

    This function uses a spring layout to position the nodes and displays
    the graph with labels using Matplotlib.
    """

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()


def get_matrices_from_graph(G, double_stochastic=False):
    """
    Generate adjacency and weighted adjacency matrices from a graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph from which to generate the adjacency matrices.
    double_stochastic : bool, optional
        If True, generate a doubly stochastic weighted adjacency matrix.
        If False, generate a row stochastic weighted adjacency matrix.
        Default is False.

    Returns
    -------
    Adj : numpy.ndarray
        The adjacency matrix of the graph.
    WeightedAdj : numpy.ndarray
        The weighted adjacency matrix of the graph, either row or
        doubly stochastic based on the double_stochastic parameter.
    """

    Adj = nx.adjacency_matrix(G, weight="weight").toarray()
    N = Adj.shape[0]

    # Normalize it (Row stochastic)
    if not double_stochastic:
        WeightedAdj = Adj.astype(float) + np.eye(N)
        for i in range(N):
            WeightedAdj[i, :] = WeightedAdj[i, :] / np.sum(WeightedAdj[i, :])   

    # (Doubly stochastic)
    else:
        WeightedAdj = Adj.astype(float) + np.eye(N)
        while any(abs(np.sum(WeightedAdj, axis=1) - 1) > 1e-8) or any(abs(np.sum(WeightedAdj, axis=0) - 1) > 1e-8):
            for i in range(N):
                WeightedAdj[i, :] = WeightedAdj[i, :] / np.sum(WeightedAdj[i, :])
            for i in range(N):
                WeightedAdj[:, i] = WeightedAdj[:, i] / np.sum(WeightedAdj[:, i])
            
            WeightedAdj = np.abs(WeightedAdj)
        
    return Adj, WeightedAdj

def laplacian(Adj):
    return np.diag(np.sum(Adj, axis=1)) - Adj





# ALGORITHMS


def consensus_tv(X0, N, p, max_iters):
    # Initialize the state of the agents
    X = np.zeros((max_iters, N))
    X[0, :] = X0

    # Algorithm
    for k in range(max_iters-1):
        # At each iteration create a new graph
        erG, Adj, WeightedAdj = generate_random_graph(N, p)

        for i in range(N):
            # For each node, compute the neighbors
            N_i = list(nx.neighbors(erG, i))
            
            # Self contribution
            X[k+1, i] += WeightedAdj[i, i] * X[k, i]
            
            # Add the contributions of the neighbors
            for j in N_i:
                X[k+1, i] += WeightedAdj[i, j] * X[k, j]

    return X

    
def consensus_ti(X0, erG, max_iters):
    # Initialize the state of the agents
    N = X0.shape[0]
    X = np.zeros((max_iters, N))
    X[0, :] = X0

    Adj, WeightedAdj = get_matrices_from_graph(erG)

    # Algorithm
    for k in range(max_iters-1):
        for i in range(N):
            # For each node, compute the neighbors
            N_i = list(nx.neighbors(erG, i))
            
            # Self contribution
            X[k+1, i] += WeightedAdj[i, i] * X[k, i]
            
            # Add the contributions of the neighbors
            for j in N_i:
                X[k+1, i] += WeightedAdj[i, j] * X[k, j]

    return X






# OPTIMIZATION

def quadratic_cost_function(x, Q, q):
    f = 0.5 * x.T @ Q @ x + q.T @ x
    df = Q @ x + q
    
    return f, df, Q

def gradient_method_qp(Q, q, x0, max_iters, tol, alpha):
    d = x0.shape[0]
    xx = np.zeros((max_iters, d))

    xx[0, :] = x0
    for k in range(max_iters-1):
        f, df, Q = quadratic_cost_function(xx[k, :], Q, q)
        xx[k+1, :] = xx[k, :] - alpha * df

    return xx



################################
# UTILS FOR FORMATON CONTROL
################################

#
# Utils for Formation control
# Ivano Notarnicola
# Bologna, 08/04/2025
#
import numpy as np
import matplotlib.pyplot as plt


###############################################################################
#
# System dynamics
#
def formation_vect_field(xt, n_x, distances):
    N = len(xt) // n_x
    xt_ = xt.reshape((N, n_x))
    dxt_ = np.zeros((N, n_x))


    for i in range(N):
        N_i = np.where(distances[i] > 0)[0]

        x_i = xt_[i]
        for j in N_i:
            x_j = xt_[j]
            dxt_[i] += -(np.linalg.norm(x_i - x_j)**2 - distances[i, j]**2) * (x_i - x_j)

    dxt = dxt_.reshape(-1)
    return dxt


def inter_distance_error(XX, NN, n_x, distances, horizon):
    err = np.zeros((len(horizon), NN, NN))
    for tt in range(len(horizon)):
        xt = XX[tt].reshape((NN, n_x))

        for i in range (NN):
            N_i = np.where(distances[i] > 0)[0]
            x_i_t = xt[i]
            for j in N_i:
                x_j_t = xt[j]
                
                err[tt, i, j] = np.abs(np.linalg.norm(x_i_t - x_j_t) - distances[i, j])

    return err.reshape(len(horizon), -1)

def animation(XX, NN, n_x, horizon, Adj, ax, wait_time=0.0000001):
    axes_lim = (np.min(XX) - 1, np.max(XX) + 1)
    ax.set_xlim(axes_lim)
    ax.set_ylim(axes_lim)

    # plot the first frame as cross markers
    for ii in range(NN):
        ax.plot(
            XX[0, ii * n_x],
            XX[0, ii * n_x + 1],
            marker="x",
            markersize=15,
            fillstyle="full",
            color="tab:red",
        )



    for tt in range(len(horizon)):
        # plot 2d-trajectories
        ax.plot(
            XX[:, 0 : n_x * NN : n_x],
            XX[:, 1 : n_x * NN : n_x],
            color="tab:gray",
            linestyle="dashed",
            alpha=0.5,
        )

        # plot 2d-formation
        xx_tt = XX[tt].reshape((NN, n_x))

        for ii in range(NN):
            p_prev = xx_tt[ii]


            for jj in range(NN):
                if Adj[ii, jj] & (jj > ii):
                    p_curr = xx_tt[jj]
                    ax.plot(
                        [p_prev[0], p_curr[0]],
                        [p_prev[1], p_curr[1]],
                        linewidth=1,
                        color="steelblue",
                        linestyle="solid",
                    )
            
            
            ax.plot(
                p_prev[0],
                p_prev[1],
                marker="o",
                markersize=5,
                fillstyle="full",
                color="tab:red",
            )

        ax.set_xlim(axes_lim)
        ax.set_ylim(axes_lim)
        ax.axis("equal")
        ax.set_xlabel("first component")
        ax.set_ylabel("second component")
        ax.set_title(f"Simulation time = {horizon[tt]:.2f} s")
        plt.show(block=False)
        plt.pause(wait_time)
        ax.cla()
    # keep it at it last frame when it ends
    ax.plot(
        XX[:, 0 : n_x * NN : n_x],
        XX[:, 1 : n_x * NN : n_x],
        color="tab:gray",
        linestyle="dashed",
        alpha=0.5,
    )
    for ii in range(NN):
        p_prev = xx_tt[ii]
        ax.plot(
            p_prev[0],
            p_prev[1],
            marker="o",
            markersize=15,
            fillstyle="full",
            color="tab:red",
        )
        for jj in range(NN):
            if Adj[ii, jj] & (jj > ii):
                p_curr = xx_tt[jj]
                ax.plot(
                    [p_prev[0], p_curr[0]],
                    [p_prev[1], p_curr[1]],
                    linewidth=1,
                    color="steelblue",
                    linestyle="solid",
                )


def polygon_distance_matrix(NN, L=2, skipped_connections=0):
    assert NN >= 3, "The number of agents (NN) must be at least 3."

    R = L / (2 * np.sin(np.pi / NN))  # Radius of circumscribed circle
    # Precompute distances for steps from 0 to floor(NN/2) due to symmetry
    max_step = NN // 2
    distances_k = [0] + [2 * R * np.sin(k * np.pi / NN) for k in range(1, max_step + 1)]

    # Build the symmetric distance matrix
    distances = np.zeros((NN, NN))
    for i in range(NN):
        for j in range(NN):
            k = min((j - i) % NN, (i - j) % NN)  # Minimal circular step distance
            if k > skipped_connections:
                distances[i, j] = distances_k[k]

    #print(distances)
    
    Adj = distances > 0
    return distances, Adj


def distance_matrix_from_points(points, p = 0.2):
    distances = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(len(points)):
            if np.random.rand() <= p: distances[i, j] = np.linalg.norm(points[i] - points[j])

    Adj = distances > 0
    return distances, Adj

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

def sample_text_pixels(word, NN, font_size=100, font_properties=None):
    """
    Genera una lista di punti che campionano una parola in modo uniforme,
    e restituisce NN coppie (x, y) selezionate casualmente da questi punti.
    
    Args:
        word (str): La parola da campionare.
        NN (int): Numero di agenti (punti da restituire).
        font_size (int): Dimensione del font per il campionamento.
        font_properties (FontProperties): Proprietà del font (opzionale).
        
    Returns:
        np.array: Array di shape (NN, 2) con le coordinate x, y dei punti.
    """
    if font_properties is None:
        font_properties = FontProperties()
    
    # Crea un TextPath per la parola
    tp = TextPath((0, 0), word, size=font_size, prop=font_properties)
    
    # Estrai i vertici e i codici del percorso
    vertices = tp.vertices
    codes = tp.codes
    
    # Filtra i punti (escludi i codici di "moveto" e "closepoly")
    # e seleziona solo i punti dei segmenti di linea (codice 1 o 2)
    mask = (codes == 1) | (codes == 2)
    filtered_vertices = vertices[mask]
    
    # Se non ci sono punti validi, restituisci punti casuali attorno all'origine
    if len(filtered_vertices) == 0:
        return np.random.uniform(-1, 1, size=(NN, 2))
    
    # Campiona NN punti in modo uniforme dai vertici filtrati
    if len(filtered_vertices) >= NN:
        sampled_indices = np.random.choice(len(filtered_vertices), NN, replace=False)
        sampled_points = filtered_vertices[sampled_indices]
    else:
        # Se i punti sono meno di NN, ne seleziona alcuni con ripetizione
        sampled_indices = np.random.choice(len(filtered_vertices), NN, replace=True)
        sampled_points = filtered_vertices[sampled_indices]
    
    return sampled_points


# #######################
# Aggregative algorithm
# #######################
import matplotlib.animation
def animate_aggregative(x, r_0, r, sigma, N, A, max_iters, velocity = 10):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    def update(frame):
        frame = int(frame * velocity) % len(x)
        ax.clear()
        ax.scatter(x[frame, :, 0], x[frame, :, 1], x[frame, :, 2], c='b', marker='s', label='agents')
        ax.scatter(r_0[frame, 0], r_0[frame, 1], r_0[frame, 2], c='g', marker='^', label='target')
        ax.scatter(r[frame, :, 0], r[frame, :, 1], r[frame, :, 2], c='r', marker='o', label='intruders')
        ax.scatter(sigma(x[frame, :, :])[0], sigma(x[frame, :, :])[1], sigma(x[frame, :, :])[2], c='k', marker='*', label='barycenter')
        # Draw communication lines in between agents
        for i in range(N):
            for j in range(i, N):
                if A[i, j] > 0:
                    ax.plot([x[frame, i, 0], x[frame, j, 0]], [x[frame, i, 1], x[frame, j, 1]], [x[frame, i, 2], x[frame, j, 2]], 'k--', alpha=0.05)
        # Add a single legend entry for communication edges
        ax.plot([], [], [], 'k--', alpha=0.2, label='communication')
        ax.legend()

        ax.grid()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.set_zlim([-5, 5])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Iteration: {frame}")

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=int(max_iters // velocity), interval = 50, repeat=True)
    plt.show()



from noise import pnoise1

def random_walk(N, TT, dt, ns, ns_total, size, speed):
    
    xdes = np.zeros((N, ns_total, TT))
    base_freq = 0.1

    for j in range(N):
        seeds = np.random.uniform(-1000, 1000, ns)
        for i in range(ns):
            for t in range(TT):
                xdes[j, i, t] = size*pnoise1(dt*speed * t * base_freq + seeds[i], repeat=TT)

    return xdes


def generate_smooth_path(start_pos, steps, scale=0.01, step_size=2.0, seed=0.0):
    path = np.zeros((steps, len(start_pos)))
    path[0] = start_pos
    for t in range(1, steps):
        for d in range(len(start_pos)):
            delta = pnoise1(t * scale + seed + d) - pnoise1((t - 1) * scale + seed + d)
            path[t, d] = path[t - 1, d] + step_size * delta
    return path


def generate_intruders_and_target(max_iters, N, ni, speed, min_dist = 0.5):
    from params import dt
    while True:
        # New random seeds for variation each loop
        seeds = np.random.uniform(-1000, 1000, size=4)

        # Opponents' positions
        r = np.zeros((max_iters, N, ni))
        for i in range(N):
            start_pos = np.random.uniform(-2, 2, size=ni)
            r[:, i, :] = generate_smooth_path(start_pos, max_iters, scale=dt*speed, step_size=2, seed=seeds[2] * i)

        # Target's path
        target_start = np.random.uniform(-1, 1, size=ni)
        r_0 = generate_smooth_path(target_start, max_iters, scale=dt*speed, step_size=1.0, seed=seeds[0])

        # Ensure all opponents are at least 0.5 units away from the target at all times
        distances = np.linalg.norm(r - r_0[:, np.newaxis, :], axis=2)  # (max_iters, N)
        if np.all(distances >= min_dist):
            return r, r_0

from params import N, gamma_x_r, gamma_sigma_r0, gamma_x_sigma, gamma_barrier, gamma_barrier_intruder, gamma_x_r0, gamma_barrier_target

def sigma(x):
    return 1/N * np.sum(x, axis=0)
def li(x, r, i, sigma_x, A, r_0):
    x_i = x[i, :]
    r_i = r[i, :]
    epsilon = 1e-3

    distance_from_neighbor = 0
    distance_from_intruder = 0

    for j in range(N):
        x_j = x[j, :]
        r_j = r[j, :]

        if A[i, j] > 0 and i != j:
            d = np.linalg.norm(x_i - x_j)
            distance_from_neighbor -= gamma_barrier * np.log(d + epsilon)

        d_intr = np.linalg.norm(x_i - r_j)
        distance_from_intruder -= gamma_barrier_intruder * np.log(d_intr + epsilon)

    distance_from_opp = gamma_x_r * np.linalg.norm(x_i - r_i)**2
    distance_from_barycenter = gamma_x_sigma * np.linalg.norm(x_i - sigma_x)**2
    distance_from_target = gamma_x_r0 * np.linalg.norm(x_i - r_0)**2
    distance_target_barycenter = gamma_sigma_r0 * np.linalg.norm(sigma_x - r_0)**2
    barrier_x_target = -gamma_barrier_target * np.log(np.linalg.norm(x_i - r_0) + epsilon)

    return (
        distance_from_opp +
        distance_from_barycenter +
        distance_from_target +
        distance_target_barycenter +
        distance_from_neighbor +
        distance_from_intruder +
        barrier_x_target
    )

def nabla_1li(x, r, i, sigma_x, A, r_0):
    x_i = x[i, :]
    r_i = r[i, :]
    epsilon = 1e-3

    distance_from_opp = 2 * gamma_x_r * (x_i - r_i)
    distance_from_barycenter = 2 * gamma_x_sigma * (x_i - sigma_x)
    distance_from_target = 2 * gamma_x_r0 * (x_i - r_0)

    distance_from_neighbor = 0
    distance_from_intruder = 0

    for j in range(N):
        x_j = x[j, :]
        r_j = r[j, :]

        if A[i, j] > 0 and i != j:
            diff = x_i - x_j
            d = np.linalg.norm(diff)
            distance_from_neighbor -= gamma_barrier * diff / (d + epsilon)**2

        diff_intr = x_i - r_j
        d_intr = np.linalg.norm(diff_intr)
        distance_from_intruder -= gamma_barrier_intruder * diff_intr / (d_intr + epsilon)**2

    barrier_x_target = -gamma_barrier_target * (x_i - r_0) / (np.linalg.norm(x_i - r_0) + epsilon)**2

    return (
        distance_from_opp +
        distance_from_barycenter +
        distance_from_target +
        distance_from_neighbor +
        distance_from_intruder +
        barrier_x_target
    )

def nabla_2li(x, i, sigma_x, r_0):
    x_i = x[i, :]
    return (
        2 * gamma_sigma_r0 * (sigma_x - r_0) +
        2 * gamma_x_sigma * (sigma_x - x_i)
    )


# Total cost
def J(x, opp, tar, A):
    s = sigma(x)
    return np.sum([li(x, opp, i, s, A, tar) for i in range(N)])

def eq_state(pos, ns):
    return np.array([*pos, *np.zeros(ns - len(pos))])


def update_drone_state(drone, xdes, dt):
    u = drone.LQcontrol(drone.state, xdes)
    thrust = drone.forces_to_thrust(u)
    drone.state = drone.state + dt * drone.f(drone.state, u)
    return drone.state, thrust


def prepare_vis_vectors(x_star_team, x_star_opponents, x_star_target):
    """
    Prepare the vectors for visualization.
    """
    max_iters, N, nx = x_star_team.shape
    vis_vector = np.zeros((max_iters, 2*N + 1, nx))
    vis_vector[:, :N, :] = x_star_team
    vis_vector[:, N:2*N, :] = x_star_opponents
    vis_vector[:, -1, :] = x_star_target[:, 0, :]
    return vis_vector
