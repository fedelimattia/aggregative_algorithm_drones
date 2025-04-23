import numpy as np
from scipy import linalg as la
from numpy import sin, cos

class Drone:
    def __init__(self, dt, M = 0.25, g = 9.81, L = 0.1, 
                 Jx = 2.5e-3, Jy = 2.5e-3, Jz = 5e-3, 
                 beta = 0.05, LQmatrices = None):
            
        # Model params
        self.M = M
        self.L = L
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.J = np.diag([self.Jx, self.Jy, self.Jz])
        self.g = g
        self.beta = beta
        self.ns = 12
        self.ni = 4
        self.state = np.zeros(self.ns)  # [x, y, z, theta_x, theta_y, vx, vy, vz, omega_x, omega_y]
        self.inv_M = 1.0 / self.M
        self.inv_J = np.linalg.inv(self.J)
        self.dt = dt
        self.time = 0
            
        # LQ gains
        if LQmatrices == None:
            self.Q = np.diag([2e1, 2e1, 1e1,
                              5e1, 5e1, 1e1,
                              1e-9, 1e-9, 1e-9,
                              1e0, 1e0, 1e-6])

            self.R = np.diag([1e-1, 1e0, 1e0, 1e0])
        else:
            self.Q = LQmatrices[0]
            self.R = LQmatrices[1]

        self.A, self.B = self.gradients(dt)
        self.P = la.solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.B.T @ self.P @ self.B + self.R) @ (self.B.T @ self.P @ self.A)


    def update_state(self, dt, u):
        self.state += dt * self.f(self.state, u)
        self.time += 1

    def f(self, state, u):
        x, y, z, theta_x, theta_y, theta_z, vx, vy, vz, omega_x, omega_y, omega_z = state
        T_vert, tau_x, tau_y, tau_z = u  # ignori yaw
        
        sin_theta_x = np.sin(theta_x)
        cos_theta_x = np.cos(theta_x)
        sin_theta_y = np.sin(theta_y)
        cos_theta_y = np.cos(theta_y)
        sin_theta_z = np.sin(theta_z)
        cos_theta_z = np.cos(theta_z)

        R = np.array([
            [
                cos_theta_y * cos_theta_z,
                sin_theta_x * sin_theta_y * cos_theta_z - cos_theta_x * sin_theta_z,
                cos_theta_x * sin_theta_y * cos_theta_z + sin_theta_x * sin_theta_z
            ],
            [
                cos_theta_y * sin_theta_z,
                sin_theta_x * sin_theta_y * sin_theta_z + cos_theta_x * cos_theta_z,
                cos_theta_x * sin_theta_y * sin_theta_z - sin_theta_x * cos_theta_z
            ],
            [
                -sin_theta_y,
                sin_theta_x * cos_theta_y,
                cos_theta_x * cos_theta_y
            ]
        ])

        thrust_inertial = R @ np.array([0, 0, T_vert])

        # Angular accelerations
        omega = np.array([omega_x, omega_y, omega_z])
        tau = np.array([tau_x, tau_y, tau_z])
        damping = -self.beta * omega

        omega_dot = self.inv_J @ (tau + damping - np.cross(omega, self.J @ omega))

        dxdt = np.empty(12)
        dxdt[0] = vx
        dxdt[1] = vy
        dxdt[2] = vz
        dxdt[3] = omega_x
        dxdt[4] = omega_y
        dxdt[5] = omega_z
        dxdt[6] = (thrust_inertial[0] - self.beta * vx) * self.inv_M
        dxdt[7] = (thrust_inertial[1] - self.beta * vy) * self.inv_M
        dxdt[8] = (thrust_inertial[2] - self.beta * vz - self.M * self.g) * self.inv_M
        dxdt[9:12] = omega_dot

        return dxdt

    def gradients(self, dt):
        A = np.zeros((self.ns, self.ns))
        B = np.zeros((self.ns, self.ni))

        # Consider the drone in an eq config: T = mg, theta_x = theta_y = 0
        T_vert = self.M * self.g

        theta_x = 0
        theta_y = 0
        theta_z = 0

        A[0, 6] = 1
        A[1, 7] = 1
        A[2, 8] = 1
        A[3, 9] = 1
        A[4, 10] = 1
        A[5, 11] = 1
        A[6, 3] = T_vert*self.inv_M*(-sin(theta_x)*sin(theta_y)*cos(theta_z) + sin(theta_z)*cos(theta_x))
        A[6, 4] = T_vert*self.inv_M*cos(theta_x)*cos(theta_y)*cos(theta_z)
        A[6, 5] = T_vert*self.inv_M*(sin(theta_x)*cos(theta_z) - sin(theta_y)*sin(theta_z)*cos(theta_x))
        A[6, 6] = -self.beta*self.inv_M
        A[7, 3] = -T_vert*self.inv_M*(sin(theta_x)*sin(theta_y)*sin(theta_z) + cos(theta_x)*cos(theta_z))
        A[7, 4] = T_vert*self.inv_M*sin(theta_z)*cos(theta_x)*cos(theta_y)
        A[7, 5] = T_vert*self.inv_M*(sin(theta_x)*sin(theta_z) + sin(theta_y)*cos(theta_x)*cos(theta_z))
        A[7, 7] = -self.beta*self.inv_M
        A[8, 3] = -T_vert*self.inv_M*sin(theta_x)*cos(theta_y)
        A[8, 4] = -T_vert*self.inv_M*sin(theta_y)*cos(theta_x)
        A[8, 8] = -self.beta*self.inv_M
        A[9, 9] = -self.beta/self.Jx
        A[10, 10] = -self.beta/self.Jy
        A[11, 11] = -self.beta/self.Jz
        B[6, 0] = self.inv_M*(sin(theta_x)*sin(theta_z) + sin(theta_y)*cos(theta_x)*cos(theta_z))
        B[7, 0] = self.inv_M*(-sin(theta_x)*cos(theta_z) + sin(theta_y)*sin(theta_z)*cos(theta_x))
        B[8, 0] = self.inv_M*cos(theta_x)*cos(theta_y)
        B[9, 1] = 1/self.Jx
        B[10, 2] = 1/self.Jy
        B[11, 3] = 1/self.Jz

        return A*dt + np.eye(self.ns), B*dt


    def reset_state(self, x0):
        self.state = x0
        self.time = 0

    def LQcontrol(self, state, xdes):
        # Equilibrium input (only T_vert = M * g to counter gravity)
        u_eq = np.array([self.M * self.g, 0, 0, 0])

        # Control input: u = -K(x - x_ref) + u_eq
        u = -self.K @ (state - xdes) + u_eq

        return u

    def simulation(self, x0, xdes, stop_time, control = "LQ"):
        TT = int(stop_time/self.dt) + 1
        evolution = np.zeros((self.ns, TT))
        inputs = np.zeros((self.ni, TT))
        thrusts = np.zeros((4, TT))

        # Initial conditions
        self.reset_state(x0)
        for t in range(TT):
            # Extract state
            evolution[:, t] = self.state

            # Control
            inputs[:, t] = np.array(self.PDcontrol(self.state, xdes[:, t]) if control == "PD" else self.LQcontrol(self.state, xdes[:, t]))
            thrusts[:, t] = self.forces_to_thrust(inputs[:, t])

            # Evolve
            self.update_state(self.dt, inputs[:, t])

        return evolution, inputs, thrusts
    
    def forces_to_thrust(self, u):
        T_vert, tau_x, tau_y, tau_z = u

        Lf = T_vert / 4 - tau_x / (2 * self.L) - tau_y / (2 * self.L)
        Rf = T_vert / 4 + tau_x / (2 * self.L) - tau_y / (2 * self.L)
        Lr = T_vert / 4 - tau_x / (2 * self.L) + tau_y / (2 * self.L)
        Rr = T_vert / 4 + tau_x / (2 * self.L) + tau_y / (2 * self.L)

        return np.array([Lf, Rf, Lr, Rr])
    