import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def draw_frame(pos, euler_angles, ax, length=0.3, linewidth=2, alpha=1.0):
    """
    Draws a body-fixed reference frame at a 3D position with given Euler angles.
    
    Parameters:
        pos: (x, y, z) position in world frame
        euler_angles: (phi, theta, psi) Euler angles in radians (ZYX order)
        ax: matplotlib 3D axis
        length: length of the arrows
        linewidth: thickness of the frame arrows
        alpha: transparency of the arrows
    """
    x, y, z = pos
    phi, theta, psi = euler_angles

    # Use scipy to create rotation matrix from Euler angles (ZYX convention)
    rotation = R.from_euler('ZYX', [psi, theta, phi])
    R_mat = rotation.as_matrix()

    # Unit axes in the body frame
    origin = np.array([x, y, z])
    x_axis = origin + R_mat[:, 0] * length
    y_axis = origin + R_mat[:, 1] * length
    z_axis = origin + R_mat[:, 2] * length

    # Plot each axis (red: x, green: y, blue: z)
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r', linewidth=linewidth, alpha=alpha)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g', linewidth=linewidth, alpha=alpha)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b', linewidth=linewidth, alpha=alpha)

    # Optional: draw the origin point
    ax.scatter(*origin, color='orange', s=20)



    # Rotation matrices
def rotation_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx  # ZYX order
def plot_drone(ax, position, rot_x, rot_y, rot_z, arm_length=0.5, color='blue',
               alpha=1.0, thrusts=None, thrust_scale=0.1, cache={}):
    """
    Efficiently plots a drone with minimal overhead, reusing artists if possible.

    Parameters:
        ax           -- 3D axis
        position     -- [x, y, z]
        rot_x, rot_y -- angles in radians
        thrusts      -- [Lf, Rf, Lr, Rr] or None
        cache        -- internal dictionary for persistent line objects (mutable default OK here)
    """

    # Rotation and transformation
    R = rotation_matrix(rot_x, rot_y, rot_z)
    body_frame = np.array([
        [0, 0, 0],                              # center
        [-arm_length,  arm_length, 0],          # Lf
        [ arm_length,  arm_length, 0],          # Rf
        [-arm_length, -arm_length, 0],          # Lr
        [ arm_length, -arm_length, 0],          # Rr
    ])
    world = (R @ body_frame.T).T + position

    # Initialize plots only once
    if 'arms' not in cache:
        cache['arms'] = []
        for i in range(1, 5):
            line, = ax.plot([], [], [], color='black', lw=2, alpha=alpha*0.7)
            cache['arms'].append(line)

        cache['center'], = ax.plot([], [], [], 'o', color=color, markersize=4, alpha=alpha)
        cache['motors'] = ax.scatter([], [], [], color='gray', s=10, alpha=alpha)

        if thrusts is not None:
            cache['thrusts'] = []
            for _ in range(4):
                tline, = ax.plot([], [], [], color='red', lw=2, alpha=0.7)
                cache['thrusts'].append(tline)

    # Update arms
    for i, line in enumerate(cache['arms']):
        line.set_data([world[0, 0], world[i+1, 0]],
                      [world[0, 1], world[i+1, 1]])
        line.set_3d_properties([world[0, 2], world[i+1, 2]])

    # Center point
    cache['center'].set_data([world[0, 0]], [world[0, 1]])
    cache['center'].set_3d_properties([world[0, 2]])

    # Motor points
    cache['motors']._offsets3d = (world[1:, 0], world[1:, 1], world[1:, 2])
    # Thrusts (if present)
    if thrusts is not None and 'thrusts' in cache:
        for i, (tline, thrust) in enumerate(zip(cache['thrusts'], thrusts)):
            start = world[i+1]
            end = start + (R @ np.array([0, 0, max(min(thrust_scale * thrust, 0.5), -0.5)]))
            tline.set_data([start[0], end[0]], [start[1], end[1]])
            tline.set_3d_properties([start[2], end[2]])


def animate_drones(x_star, dt, stop_time, thrusts = None, xdes = None, show_target = False, colors = None):
    TT = x_star.shape[0]
    N = x_star.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if colors is None: colors = [np.random.randint(0, 255, 3)/255 for _ in range(N)]
    actual_caches = [{} for _ in range(N)]
    desired_caches = [{} for _ in range(N)]

    # Set plot limits once
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)

    # # Remove the grid
    # ax.grid(False)

    # # Remove the axis ticks
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])


    # make the figure bigger
    fig.set_size_inches(15, 15)


    fps = 30

    for t in range(0, TT, int(TT * 2 / (fps*(dt * TT)))):
        # Instead of clearing, just remove the title and reset it
        ax.set_title(f"t = {t*dt:.2f} s")

        for j in range(N):
            # Desired position (faded)
            if show_target: plot_drone(
                ax,
                position=xdes[t, j, :3],
                rot_x=xdes[t, j, 4],
                rot_y=-xdes[t, j, 3],
                rot_z = xdes[t, j, 5],
                alpha=0.15,
                arm_length=0.2,
                color=colors[j],
                cache=desired_caches[j]
            )

            # Actual position with thrust
            plot_drone(
                ax,
                position=x_star[t, j, :3],
                rot_x = x_star[t, j, 3],
                rot_y = x_star[t, j, 4],
                rot_z = x_star[t, j, 5],
                thrusts=thrusts[t, j, :] if thrusts is not None else np.zeros(4),
                thrust_scale=0.3,
                arm_length=0.2,
                color=colors[j],
                cache=actual_caches[j]
            )

        plt.pause(1e-3)
    
    plt.show(block = True)
